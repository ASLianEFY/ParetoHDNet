from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from src.artifacts import artifact_metadata, write_sidecar
from src.utils import CACHE_DIR, ensure_output_dirs, get_logger, load_config, set_seed
from src.data.profiles import generate_dataset
from src.ies.system import IESSystem
from src.ies.uc_milp import solve_uc_milp

def tag(smoke):
    return '_smoke' if smoke else ''

def enable_uc_mode(cfg):
    cfg['uc'] = {'min_load_frac': 0.3, 'start_cost': 20.0, 'min_up': 3, 'min_down': 2, 'action_dim': 9, 'binary_loss_weight': 0.1, 'commitment_logit_scale': 5.0}
    return cfg

def _save_pickle(path, value, cfg, artifact_type, **metadata):
    with path.open('wb') as stream:
        pickle.dump(value, stream)
    write_sidecar(path, artifact_metadata(cfg, artifact_type, **metadata))

def _save_demos(path, demos, cfg, split):
    np.savez(path, **demos)
    write_sidecar(path, artifact_metadata(cfg, 'demonstrations', split=split, samples=int(demos['a0'].shape[0]), arrays={key: list(value.shape) for key, value in demos.items()}, feasibility_tolerance=float(cfg['milp']['feasibility_tol'])))

def _shard_dir(suffix: str, kind: str, split: str) -> Path:
    return CACHE_DIR / f'_partial{suffix}' / kind / split

def _load_progress(path: Path) -> set[int]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    return {int(i) for i in data.get('done', [])}

def _save_progress(path: Path, done: set[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'done': sorted(done)}))

def _stack_demo_records(records):
    records = sorted(records, key=lambda item: item['index'])
    return {'state': np.stack([item['state'] for item in records]).astype(np.float32), 'lam': np.asarray([item['lam'] for item in records], np.float32), 'a0': np.stack([item['a0'] for item in records]).astype(np.float32), 'f1': np.asarray([item['f1'] for item in records], np.float32), 'f2': np.asarray([item['f2'] for item in records], np.float32)}

def build_demos_resumable(system, scenarios, cfg, *, suffix, split, resume, log):
    shard_root = _shard_dir(suffix, 'demos', split)
    progress_path = shard_root / 'progress.json'
    done = _load_progress(progress_path) if resume else set()
    if not resume and shard_root.exists():
        for path in shard_root.glob('scenario_*.npz'):
            path.unlink()
        if progress_path.exists():
            progress_path.unlink()
        done = set()
    shard_root.mkdir(parents=True, exist_ok=True)
    grid = list(cfg['milp']['lambda_grid'])
    records = []
    for index, scenario in enumerate(scenarios):
        shard_path = shard_root / f'scenario_{index:05d}.npz'
        if index in done and shard_path.exists():
            with np.load(shard_path) as archive:
                for j in range(len(archive['lam'])):
                    records.append({'index': index * len(grid) + j, 'state': archive['state'][j], 'lam': float(archive['lam'][j]), 'a0': archive['a0'][j], 'f1': float(archive['f1'][j]), 'f2': float(archive['f2'][j])})
            continue
        state = scenario.state_grid(system.bat['soc_init'], system.gastank['soc_init'])
        states, lams, actions, f1s, f2s = ([], [], [], [], [])
        for lam in grid:
            result = solve_uc_milp(system, scenario, float(lam), cfg)
            states.append(state)
            lams.append(lam)
            actions.append(result['a0'])
            f1s.append(result['f1'])
            f2s.append(result['f2'])
        payload = {'state': np.stack(states).astype(np.float32), 'lam': np.asarray(lams, np.float32), 'a0': np.stack(actions).astype(np.float32), 'f1': np.asarray(f1s, np.float32), 'f2': np.asarray(f2s, np.float32)}
        np.savez(shard_path, **payload)
        done.add(index)
        _save_progress(progress_path, done)
        for j in range(len(grid)):
            records.append({'index': index * len(grid) + j, 'state': payload['state'][j], 'lam': float(payload['lam'][j]), 'a0': payload['a0'][j], 'f1': float(payload['f1'][j]), 'f2': float(payload['f2'][j])})
        if log and (index + 1) % 20 == 0:
            log.info('  UC-MILP demos: %d/%d scenarios', index + 1, len(scenarios))
    return _stack_demo_records(records)

def build_oracle_resumable(system, scenarios, cfg, *, suffix, resume, log):
    shard_root = _shard_dir(suffix, 'oracle', 'test')
    progress_path = shard_root / 'progress.json'
    done = _load_progress(progress_path) if resume else set()
    if not resume and shard_root.exists():
        for path in shard_root.glob('scenario_*.npy'):
            path.unlink()
        if progress_path.exists():
            progress_path.unlink()
        done = set()
    shard_root.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, int(cfg['milp']['oracle_grid_n']))
    fronts = [None] * len(scenarios)
    for index, scenario in enumerate(scenarios):
        shard_path = shard_root / f'scenario_{index:05d}.npy'
        if index in done and shard_path.exists():
            fronts[index] = np.load(shard_path)
            continue
        points = []
        for lam in grid:
            result = solve_uc_milp(system, scenario, float(lam), cfg)
            points.append((result['f1'], result['f2']))
        front = np.asarray(points, dtype=np.float32)
        np.save(shard_path, front)
        done.add(index)
        _save_progress(progress_path, done)
        fronts[index] = front
        if log and (index + 1) % 10 == 0:
            log.info('  UC-MILP oracle: %d/%d scenarios', index + 1, len(scenarios))
    return fronts

def main():
    parser = argparse.ArgumentParser(
        description='Generate scenarios and the legacy weighted-sum UC-MILP oracle. '
        'Use src.generate_epsilon_oracle for the publication dense reference.'
    )
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--skip-oracle', action='store_true')
    parser.add_argument('--uc-milp', action='store_true', default=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = load_config(smoke=args.smoke)
    enable_uc_mode(cfg)
    set_seed(cfg['seed'])
    ensure_output_dirs(CACHE_DIR)
    log = get_logger()
    if not args.skip_oracle:
        log.warning('This command generates the legacy weighted-sum oracle; use src.generate_epsilon_oracle for the dense epsilon reference.')
    suffix = tag(args.smoke) + '_uc_milp'
    scenario_path = CACHE_DIR / f'scenarios{suffix}.pkl'
    if args.resume and scenario_path.exists():
        log.info('Resuming with existing scenarios from %s', scenario_path.name)
        with scenario_path.open('rb') as stream:
            data = pickle.load(stream)
        counts = {split: len(values) for split, values in data.items()}
    else:
        log.info('Generating scenarios ...')
        data = generate_dataset(cfg)
        counts = {split: len(values) for split, values in data.items()}
        log.info('  train/val/test = %s/%s/%s', counts['train'], counts['val'], counts['test'])
        _save_pickle(scenario_path, data, cfg, 'scenario_splits', split_counts=counts)
    system = IESSystem(cfg)
    for split in ('train', 'val'):
        demo_path = CACHE_DIR / f'demos_{split}{suffix}.npz'
        if args.resume and demo_path.exists():
            log.info('Skipping completed %s demos (%s)', split, demo_path.name)
            continue
        log.info('Solving UC-MILP demonstrations (%s)%s ...', split, ' [resume]' if args.resume else '')
        demos = build_demos_resumable(system, data[split], cfg, suffix=suffix, split=split, resume=args.resume, log=log)
        _save_demos(demo_path, demos, cfg, split)
        log.info('  %s demos: %s samples, f1 range [%.1f,%.1f], f2 range [%.1f,%.1f]', split, demos['a0'].shape[0], demos['f1'].min(), demos['f1'].max(), demos['f2'].min(), demos['f2'].max())
    oracle_path = CACHE_DIR / f'oracle_test{suffix}.pkl'
    if not args.skip_oracle:
        if args.resume and oracle_path.exists():
            log.info('Skipping completed oracle (%s)', oracle_path.name)
        else:
            log.info('Solving UC-MILP oracle Pareto fronts (test)%s ...', ' [resume]' if args.resume else '')
            oracle = build_oracle_resumable(system, data['test'], cfg, suffix=suffix, resume=args.resume, log=log)
            _save_pickle(oracle_path, oracle, cfg, 'oracle_fronts', scenarios=len(oracle), grid_size=int(cfg['milp']['oracle_grid_n']), feasibility_tolerance=float(cfg['milp']['feasibility_tol']))
    elif oracle_path.exists():
        log.warning('--skip-oracle leaves existing oracle artifact untouched: %s', oracle_path)
    log.info('Data generation complete.')
if __name__ == '__main__':
    main()
