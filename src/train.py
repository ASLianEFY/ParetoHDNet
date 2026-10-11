"""Paired-seed training with common strict Pareto checkpoint selection.

Training-only normalization; 100 epochs; warmup/cosine AdamW; fixed validation
days/preferences/noise; independent strict feasibility audit. No test data is
used for training or selection. Checkpoints include optimizer and RNG state.
"""
from __future__ import annotations

import argparse
import copy
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.artifacts import checkpoint_metadata, require_compatible_sidecar, write_sidecar
from src.data.dataset import DemoDataset, StateNormalizer
from src.evaluate import method_metrics, normalize, scenario_bounds
from src.ies.batch_decode import decode_batch
from src.ies.independent_audit import PROTOCOL, audit_dispatch, objective_values
from src.inference import DEFAULT_OUTPUT, atomic_json
from src.models.paretohdnet import ParetoHDNet
from src.training_utils import _scheduled_lr, _rng_state, _restore_rng_state, _atomic_torch_save
from src.utils import CACHE_DIR, load_config, set_seed

MODELS = {'paretohdnet': ParetoHDNet}


@torch.no_grad()
def pareto_validation(model, cfg, norm, scenarios, demos, device):
    model.eval()
    grid = np.asarray(cfg['milp']['lambda_grid'])
    ids = np.linspace(0, len(scenarios) - 1, min(64, len(scenarios)), dtype=int)
    lam = torch.tensor(np.stack([grid, 1 - grid], -1), dtype=torch.float32, device=device)
    rows = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if str(device).startswith('cuda') else []):
        for index in ids:
            torch.manual_seed(12345 + int(index))
            sc = scenarios[index]
            st = norm(sc.state_grid(cfg['system']['battery']['soc_init'], cfg['system']['gastank']['soc_init']))
            state = torch.tensor(st, device=device)[None].repeat(len(grid), 1, 1)
            actions = model.sample(state, lam, M=2, K_infer=20).cpu().numpy().reshape(-1, cfg['horizon']['T'], 9)
            values, on = decode_batch(actions, sc, cfg)
            feasible = sum(audit_dispatch(values, on, sc, cfg).values()) <= cfg['milp']['feasibility_tol']
            points = objective_values(values, on, sc, cfg)
            section = slice(index * len(grid), (index + 1) * len(grid))
            reference = np.stack([demos['f1'][section], demos['f2'][section]], -1)
            ideal, span, pf = scenario_bounds(reference)
            row = method_metrics(normalize(points[feasible], ideal, span), normalize(pf, ideal, span))
            row['feas'] = float(feasible.mean())
            rows.append(row)
    return {key: float(np.mean([row[key] for row in rows])) for key in ('hv', 'igd', 'feas')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=MODELS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--micro-batch-size', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT / 'checkpoints')
    parser.add_argument('--validation-every', type=int, default=5)
    parser.add_argument('--width', type=int, default=None)
    args = parser.parse_args()
    if min(args.epochs, args.micro_batch_size, args.threads, args.validation_every) < 1:
        parser.error('numeric arguments must be positive')
    cfg = load_config()
    torch.set_num_threads(args.threads)
    if args.kind == 'gru':
        torch.backends.cudnn.enabled = False
    set_seed(args.seed)
    demos = {}
    for split in ('train', 'val'):
        path = CACHE_DIR / f'demos_{split}_uc_milp.npz'
        require_compatible_sidecar(path, cfg, artifact_type='demonstrations', fields={'split': split})
        with np.load(path) as archive:
            demos[split] = {k: archive[k] for k in archive.files}
    path = CACHE_DIR / 'scenarios_uc_milp.pkl'
    require_compatible_sidecar(path, cfg, artifact_type='scenario_splits')
    with path.open('rb') as stream:
        # This file contains all splits; only validation days are retained/used.
        val_scenarios = pickle.load(stream)['val']
    norm = StateNormalizer.fit(demos['train']['state'])
    if args.width:
        cfg['model']['d_model'] = args.width
    cfg['full_diffusion']['objective_scales'] = [float(np.median(abs(demos['train'][key]))) for key in ('f1', 'f2')]
    cfg['train']['epochs'] = args.epochs
    loaders = {key: DataLoader(DemoDataset(demos[key], norm), batch_size=cfg['train']['batch_size'], shuffle=key == 'train', num_workers=0) for key in ('train', 'val')}
    model = MODELS[args.kind](cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    suffix = f'{args.kind}_seed{args.seed}' + (f'_width{args.width}' if args.width else '')
    args.output.mkdir(parents=True, exist_ok=True)
    best_path, last_path = args.output / f'{suffix}.pt', args.output / f'{suffix}_last.pt'
    protocol = {'kind': args.kind, 'training_seed': args.seed, 'epochs': args.epochs,
                'micro_batch_size': args.micro_batch_size, 'validation_every': args.validation_every,
                'feasibility_protocol': PROTOCOL, 'width': args.width,
                'selection': 'highest_fixed_strict_validation_hv_then_lowest_igd',
                'validation_scenarios': 64, 'validation_samples': 2, 'validation_steps': 20,
                'validation_reference': '11-weight training-validation demonstrations',
                'optimizer': 'AdamW warmup500 cosine', 'normalizer': 'train_only', 'data_seed': 42}
    start, step, history = 0, 0, []
    best = {'hv': -float('inf'), 'igd': float('inf'), 'epoch': None}
    if last_path.exists() and not args.resume:
        raise ValueError(f'training state exists: use --resume ({last_path})')
    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location=args.device, weights_only=False)
        if saved['protocol'] != protocol:
            raise ValueError('resume protocol does not match')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        _restore_rng_state(saved['rng'])
        start, step, history, best = saved['epoch'] + 1, saved['step'], saved['history'], saved['best']
    print(f'TRAIN {suffix} params={sum(p.numel() for p in model.parameters())} start={start}', flush=True)
    if start >= args.epochs:
        print(f'COMPLETE {best_path}', flush=True)
        return
    for epoch in range(start, args.epochs):
        started = time.perf_counter()
        model.train()
        losses = []
        for batch in loaders['train']:
            state, lam, action, raw = batch[:4]
            size = len(state)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = _scheduled_lr(cfg, step, args.epochs * len(loaders['train']))
            for i in range(0, size, args.micro_batch_size):
                j = min(i + args.micro_batch_size, size)
                tensors = [x[i:j].to(args.device) for x in (state, lam, action, raw)]
                if args.kind == 'paretohdnet':
                    loss = model.loss(tensors[0], tensors[3], tensors[1], tensors[2])[0]
                else:
                    loss = model.loss(*tensors[:3])[0]
                (loss * ((j - i) / size)).backward()
                losses.append((float(loss.detach()), j - i))
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            optimizer.step()
            step += 1
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()] if args.device == 'cuda' else []):
            torch.manual_seed(12345)
            validation = []
            for batch in loaders['val']:
                for i in range(0, len(batch[0]), args.micro_batch_size):
                    state, lam, action, raw = [x[i:i + args.micro_batch_size].to(args.device) for x in batch[:4]]
                    loss = model.validation_loss(state, raw, lam, action)[0] if args.kind == 'paretohdnet' else model.loss(state, lam, action)[0]
                    validation.append((float(loss), len(state)))
        row = {'epoch': epoch + 1, 'loss': sum(v*n for v,n in losses)/sum(n for _,n in losses),
               'val_loss': sum(v*n for v,n in validation)/sum(n for _,n in validation), 'seconds': time.perf_counter()-started}
        if (epoch + 1) % args.validation_every == 0 or epoch + 1 == args.epochs:
            metrics = pareto_validation(model, cfg, norm, val_scenarios, demos['val'], args.device)
            row['pareto'] = metrics
            if metrics['hv'] > best['hv'] + 1e-12 or (abs(metrics['hv'] - best['hv']) <= 1e-12 and metrics['igd'] < best['igd']):
                best = {**metrics, 'epoch': epoch + 1}
                metadata = checkpoint_metadata(cfg, weight_key='model', best_epoch=epoch + 1, best_val_loss=row['val_loss'],
                    training={**protocol, 'model_type': args.kind, 'best_validation': best,
                              'parameters': sum(p.numel() for p in model.parameters()),
                              'device': torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU', 'torch': torch.__version__})
                _atomic_torch_save({'model': model.state_dict(), 'normalizer': norm.state_dict(), 'cfg': cfg, 'metadata': metadata}, best_path)
                write_sidecar(best_path, metadata)
        history.append(row)
        print(f'{suffix} {row}', flush=True)
        atomic_json(args.output / f'{suffix}_history.json', {'protocol': protocol, 'best': best, 'history': history})
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            _atomic_torch_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': _rng_state(),
                               'epoch': epoch, 'step': step, 'history': history, 'best': best, 'protocol': protocol}, last_path)
    print(f'COMPLETE {best_path} best={best}', flush=True)


if __name__ == '__main__':
    main()
