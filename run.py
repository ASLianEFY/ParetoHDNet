"""Portable entry points for the manuscript's common-selection protocol."""
from pathlib import Path
import argparse
import json
import os
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
KINDS = ('diffusion',)


def dataset(family):
    if family == 'synthetic':
        source = ROOT / 'data_cache/scenarios_uc_milp.pkl'
        oracle = ROOT / 'data_cache/oracle_test_uc_milp_epsilon_k101.pkl'
        with source.open('rb') as f:
            scenarios = pickle.load(f)['test']
        with oracle.open('rb') as f:
            reference = pickle.load(f)
        return dict(scenarios=scenarios, oracle=reference,
                    original_ids=list(range(len(scenarios))), infeasible_ids=[],
                    source=source, oracle_source=oracle)
    raise ValueError('Only the synthetic dataset is bundled; see README.md for external data links.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('train', 'validate', 'evaluate', 'verify', 'smoke-train'))
    p.add_argument('--kind', choices=KINDS, default='diffusion')
    p.add_argument('--seed', type=int, choices=(44,), default=44)
    p.add_argument('--checkpoint', type=Path, help='Use a newly trained checkpoint instead of the supplied selected weights')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--family', choices=('synthetic',), default='synthetic')
    p.add_argument('--budget', type=int, choices=(21, 100), default=100)
    p.add_argument('--limit', type=int, help='First N test days; omit for the full set')
    p.add_argument('--joint-lp', action='store_true', help='Also run optional LP, using the original CUDA evaluator')
    p.add_argument('--output', type=Path, help='New run output directory (never reference_results or checkpoints)')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.threads < 1 or args.epochs < 1 or (args.limit is not None and args.limit < 1):
        p.error('threads, epochs and limit must be positive')
    os.chdir(ROOT)
    import torch
    torch.set_num_threads(args.threads)
    if args.command == 'verify':
        import hashlib
        manifest = ROOT / 'SHA256SUMS'
        count = 0
        for line in manifest.read_text().splitlines():
            expected, name = line.split('  ', 1)
            actual = hashlib.sha256()
            with (ROOT / name).open('rb') as stream:
                for block in iter(lambda: stream.read(2**20), b''):
                    actual.update(block)
            if actual.hexdigest() != expected:
                raise ValueError(f'Checksum mismatch: {name}')
            count += 1
        from src.inference import load_policy
        model, normalizer, cfg, metadata = load_policy('diffusion', ROOT / 'checkpoints/diffusion_seed44.pt', device='cpu')
        assert metadata['training']['training_seed'] == 44
        assert metadata['best_epoch'] == 85
        print(json.dumps({'status': 'passed', 'files_verified': count,
                          'model': 'PGDNet', 'seed': 44, 'selected_epoch': 85}, indent=2))
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is unavailable; use --device cpu (joint LP entry point requires CUDA)')
    suffix = f'{args.kind}_seed{args.seed}'
    checkpoint = (args.checkpoint or ROOT / 'checkpoints' / f'{suffix}.pt').resolve()
    output = (args.output or ROOT / 'runs' / args.command).resolve()
    for protected in ('checkpoints', 'reference_results', 'data_cache', 'src', 'training_history', 'provenance'):
        directory = ROOT / protected
        if output == directory or directory in output.parents:
            p.error(f'output must not overwrite {protected}')
    if args.command == 'train':
        command = [sys.executable, '-m', 'src.train', '--kind', args.kind,
                   '--seed', str(args.seed), '--epochs', str(args.epochs), '--device', args.device,
                   '--threads', str(args.threads), '--micro-batch-size', '128',
                   '--validation-every', '5', '--output', str(output)]
        if args.resume:
            command.append('--resume')
        subprocess.run(command, cwd=ROOT, check=True)
        return
    from src.inference import atomic_json, evaluate_job, load_policy
    if args.kind == 'gru':
        torch.backends.cudnn.enabled = False
    if args.command in ('validate', 'smoke-train'):
        import numpy as np
        from src.utils import set_seed
        set_seed(42)
        model, normalizer, cfg, metadata = load_policy(args.kind, checkpoint, device=args.device)
        if args.command == 'validate':
            from src.train import pareto_validation
            with np.load(ROOT / 'data_cache/demos_val_uc_milp.npz') as f:
                demos = {key: f[key] for key in f.files}
            with (ROOT / 'data_cache/scenarios_uc_milp.pkl').open('rb') as f:
                scenarios = pickle.load(f)['val']
            metrics = pareto_validation(model, cfg, normalizer, scenarios, demos, args.device)
            expected = metadata['training']['best_validation']
            for key in ('hv', 'igd', 'feas'):
                np.testing.assert_allclose(metrics[key], expected[key], rtol=2e-5, atol=2e-6)
            report = {'status': 'passed', 'checkpoint': str(checkpoint),
                      'selected_epoch': metadata['best_epoch'], 'validation_days': 64,
                      'metrics': metrics, 'saved_best_validation': expected}
        else:
            from src.data.dataset import DemoDataset
            from torch.utils.data import DataLoader
            with np.load(ROOT / 'data_cache/demos_train_uc_milp.npz') as f:
                demos = {key: f[key][:2] for key in f.files}
            state, lam, action, raw = [x.to(args.device) for x in
                                     next(iter(DataLoader(DemoDataset(demos, normalizer), batch_size=2)))[:4]]
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'])
            optimizer.zero_grad(set_to_none=True)
            loss = (model.loss(state, raw, lam, action) if args.kind == 'diffusion'
                    else model.loss(state, lam, action))[0]
            assert torch.isfinite(loss), 'non-finite training loss'
            loss.backward()
            gradients = [v.grad for v in model.parameters() if v.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            optimizer.step()
            assert all(torch.isfinite(v).all() for v in model.parameters())
            report = {'status': 'passed', 'kind': args.kind, 'seed': args.seed,
                      'loss': float(loss.detach()), 'gradient_norm': float(grad_norm),
                      'optimizer_steps': 1, 'batch_size': 2, 'checkpoint_written': False,
                      'note': 'In-memory training-path test, not a retraining experiment.'}
        atomic_json(output / f'{suffix}.json', report)
        print(json.dumps(report, indent=2))
        return
    data = dataset(args.family)
    if args.limit is not None and args.limit > len(data['scenarios']):
        p.error('limit exceeds the dataset size')
    size = args.limit or len(data['scenarios'])
    output = output / f'{args.family}_k{args.budget}_n{size}'
    if args.joint_lp:
        if args.device != 'cuda' or args.budget != 100:
            p.error('original LP evaluator requires CUDA and budget 100')
        import src.evaluate_lp as transfer
        for key in ('scenarios', 'oracle', 'original_ids', 'dates'):
            if key in data:
                data[key] = data[key][:size]
        transfer.OUTPUT = output
        transfer.evaluate_one({'label': f'frozen_{suffix}', 'kind': args.kind,
                               'seed': args.seed, 'variant': 'frozen', 'checkpoint': checkpoint},
                              args.family, data)
    else:
        job = {'id': f'new_{suffix}_k{args.budget}m1_d10', 'kind': args.kind,
               'checkpoint': str(checkpoint), 'K_lambda': args.budget, 'M_samples': 1, 'K_infer': 10}
        result = evaluate_job(job, data['scenarios'], data['oracle'], output,
                             device=args.device, limit=args.limit,
                             oracle_path=data['oracle_source'], scenario_path=data['source'],
                             source_protocol={'family': args.family, 'reference': 'epsilon_constraint',
                                              'epsilon_grid_size': 101})
        print(json.dumps(result['aggregate'], indent=2))


if __name__ == '__main__':
    main()
