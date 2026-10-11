from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.artifacts import (
    artifact_metadata,
    read_sidecar,
    require_compatible_sidecar,
    write_sidecar,
)
from src.ies.system import IESSystem
from src.ies.uc_milp import _pareto_deduplicate, uc_epsilon_oracle_front
from src.scripts.gen_data import enable_uc_mode
from src.utils import CACHE_DIR, ensure_output_dirs, get_logger, load_config


def _solve_scenario(index, scenario, cfg, grid_size):
    started = time.perf_counter()
    result = uc_epsilon_oracle_front(
        IESSystem(cfg), scenario, cfg, grid_size=grid_size
    )
    return {
        'index': index,
        'runtime_seconds': time.perf_counter() - started,
        **result,
    }


def _save_shard(path: Path, result: dict) -> None:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp.npz')
    np.savez_compressed(
        temporary,
        front=result['front'],
        caps=result['caps'],
        decoded_points=result['decoded_points'],
        solver_points=result['solver_points'],
        endpoint_points=result['endpoint_points'],
        runtime_seconds=np.asarray(result['runtime_seconds']),
        max_epsilon_violation=np.asarray(result['max_epsilon_violation']),
        max_decoder_objective_delta=np.asarray(
            result['max_decoder_objective_delta']
        ),
        infeasible_caps=np.asarray(result['infeasible_caps']),
    )
    temporary.replace(path)


def _load_shard(path: Path) -> dict:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def _write_progress(path: Path, done: set[int], total: int) -> None:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        json.dumps({'done': sorted(done), 'total': total}, indent=2) + '\n'
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description='Generate dense epsilon-constraint UC-MILP reference fronts.'
    )
    parser.add_argument('--grid-size', type=int, default=101)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.grid_size < 2:
        parser.error('--grid-size must be at least 2')
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')

    cfg = enable_uc_mode(load_config())
    ensure_output_dirs(CACHE_DIR)
    logger = get_logger()
    scenario_path = CACHE_DIR / 'scenarios_uc_milp.pkl'
    require_compatible_sidecar(
        scenario_path, cfg, artifact_type='scenario_splits'
    )
    with scenario_path.open('rb') as stream:
        scenarios = pickle.load(stream)['test']
    if args.limit is not None:
        scenarios = scenarios[:args.limit]

    run_name = f'epsilon_k{args.grid_size}'
    if args.limit is not None:
        run_name += f'_n{len(scenarios)}'
    shard_root = CACHE_DIR / f'_partial_uc_milp_{run_name}' / 'oracle' / 'test'
    output_path = CACHE_DIR / f'oracle_test_uc_milp_{run_name}.pkl'
    previous_metadata = read_sidecar(output_path) if args.resume else None
    shard_root.mkdir(parents=True, exist_ok=True)
    progress_path = shard_root / 'progress.json'
    existing = sorted(shard_root.glob('scenario_*.npz'))
    if existing and not args.resume:
        parser.error(
            f'{len(existing)} shards already exist under {shard_root}; use --resume'
        )
    done = {
        int(path.stem.split('_')[-1])
        for path in existing
        if path.is_file()
    }
    pending = [
        (index, scenario)
        for index, scenario in enumerate(scenarios)
        if index not in done
    ]
    logger.info(
        'Epsilon oracle: scenarios=%d, grid=%d, workers=%d, completed=%d',
        len(scenarios), args.grid_size, args.workers, len(done),
    )
    wall_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_solve_scenario, index, scenario, cfg, args.grid_size): index
            for index, scenario in pending
        }
        for future in as_completed(futures):
            result = future.result()
            index = int(result['index'])
            _save_shard(shard_root / f'scenario_{index:05d}.npz', result)
            done.add(index)
            _write_progress(progress_path, done, len(scenarios))
            if len(done) % 10 == 0 or len(done) == len(scenarios):
                logger.info('  completed %d/%d scenarios', len(done), len(scenarios))

    shards = [_load_shard(shard_root / f'scenario_{i:05d}.npz') for i in range(len(scenarios))]
    fronts = [
        _pareto_deduplicate(
            np.vstack([item['decoded_points'], item['endpoint_points']])
        )
        for item in shards
    ]
    with output_path.open('wb') as stream:
        pickle.dump(fronts, stream)
    runtimes = np.asarray([float(item['runtime_seconds']) for item in shards])
    point_counts = np.asarray([len(item['front']) for item in shards])
    infeasible_caps = np.asarray(
        [
            int(item['infeasible_caps'])
            if 'infeasible_caps' in item
            else args.grid_size - len(item['caps'])
            for item in shards
        ]
    )
    current_wall_seconds = time.perf_counter() - wall_started
    preserved_wall_seconds = (
        previous_metadata.get('runtime_seconds', {}).get('wall')
        if previous_metadata is not None and not pending
        else None
    )
    metadata = artifact_metadata(
        cfg,
        'oracle_fronts',
        method='epsilon_constraint',
        scenarios=len(fronts),
        epsilon_grid_size=args.grid_size,
        endpoint_strategy='weighted_sum_extremes',
        objective='minimize_economic_cost',
        constrained_objective='emissions',
        feasibility_tolerance=float(cfg['milp']['feasibility_tol']),
        total_milp_solves=len(fronts) * (args.grid_size + 2),
        successful_epsilon_solves=int(len(fronts) * args.grid_size - infeasible_caps.sum()),
        infeasible_epsilon_solves=int(infeasible_caps.sum()),
        nondominated_points={
            'min': int(point_counts.min()),
            'mean': float(point_counts.mean()),
            'max': int(point_counts.max()),
        },
        runtime_seconds={
            'wall': (
                float(preserved_wall_seconds)
                if preserved_wall_seconds is not None
                else current_wall_seconds
            ),
            'scenario_sum': float(runtimes.sum()),
            'scenario_mean': float(runtimes.mean()),
            'last_reaggregation': current_wall_seconds,
        },
        max_epsilon_violation=float(
            max(float(item['max_epsilon_violation']) for item in shards)
        ),
        max_decoder_objective_delta=float(
            max(float(item['max_decoder_objective_delta']) for item in shards)
        ),
    )
    if previous_metadata is not None and not pending:
        metadata['provenance'] = previous_metadata.get(
            'provenance', metadata['provenance']
        )
    write_sidecar(output_path, metadata)
    logger.info(
        'Wrote %s; front points min/mean/max=%d/%.1f/%d',
        output_path.name,
        point_counts.min(), point_counts.mean(), point_counts.max(),
    )


if __name__ == '__main__':
    main()
