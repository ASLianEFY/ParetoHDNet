"""Optional continuous LP evaluation of ParetoHDNet candidate sets."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from src.evaluate import aggregate, aggregate_coverage, method_metrics, normalize, scenario_bounds
from src.ies.batch_decode import decode_batch
from src.ies.independent_audit import PROTOCOL, audit_dispatch, objective_values
from src.ies.joint_recourse_storage import PROTOCOL as LP_PROTOCOL, project_joint_balance
from src.inference import atomic_json, file_digest, load_policy
from src.utils import RESULTS_DIR, set_seed
OUTPUT = RESULTS_DIR / 'paretohdnet_lp'

def score(values, on, scenario, cfg, ideal, span, reference):
    violations = audit_dispatch(values, on, scenario, cfg)
    feasible = sum(violations.values()) <= cfg['milp']['feasibility_tol']
    points = objective_values(values, on, scenario, cfg)[feasible]
    metrics = method_metrics(normalize(points, ideal, span), reference)
    metrics['feas'] = float(feasible.mean())
    return metrics, feasible, points, {key: {'count': int((value > cfg['milp']['feasibility_tol']).sum()),
                                            'sum': float(value.sum()), 'max': float(value.max())}
                                      for key, value in violations.items()}


def evaluate_one(model_row, family, data):
    base_protocol = {'model': model_row['label'], 'kind': model_row['kind'], 'variant': model_row['variant'],
        'seed': model_row['seed'], 'family': family, 'K_lambda': 100, 'M_samples': 1, 'K_infer': 10,
        'checkpoint': str(model_row['checkpoint']), 'checkpoint_sha256': file_digest(model_row['checkpoint']),
        'scenario_path': str(data['source']), 'scenario_sha256': file_digest(data['source']),
        'oracle_path': str(data['oracle_source']), 'oracle_sha256': file_digest(data['oracle_source']),
        'scenario_ids': list(range(len(data['scenarios']))), 'original_ids': data['original_ids'],
        'infeasible_ids': data['infeasible_ids'], 'feasibility_protocol': PROTOCOL,
        'apply_recourse': True, 'milp_repair': False, 'evaluation_seed': '42+scenario_index',
        'source_sha256': {name: file_digest(Path(__file__).parent / name) for name in
            ('evaluate_lp.py', 'ies/joint_recourse_storage.py', 'ies/batch_decode.py', 'ies/independent_audit.py',
             'ies/system.py', 'models/paretohdnet.py')}}
    protocols, targets = {}, {}
    directory = OUTPUT / 'evaluations' / family
    for variant in ('base', 'joint_lp'):
        protocols[variant] = {**base_protocol, 'postprocess': variant,
                              'continuous_lp_projection': variant == 'joint_lp',
                              'projection_protocol': LP_PROTOCOL if variant == 'joint_lp' else 'historical_common_recourse'}
        targets[variant] = directory / f"{model_row['label']}_{variant}.json"
    signatures = {key: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                  for key, value in protocols.items()}
    for variant, path in targets.items():
        if path.exists() and json.loads(path.read_text())['signature'] != signatures[variant]:
            raise ValueError(f'Transfer evaluation changed: {path}')
    if all(path.exists() for path in targets.values()):
        print('SKIP', family, model_row['label'], flush=True)
        return
    shards = directory / 'partial' / model_row['label']
    shards.mkdir(parents=True, exist_ok=True)
    manifest = shards / 'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != signatures:
        raise ValueError('Partial transfer evaluation signature changed')
    atomic_json(manifest, signatures)
    model, norm, cfg, metadata = load_policy(model_row['kind'], model_row['checkpoint'])
    weights = np.linspace(0, 1, 100)
    lam = torch.tensor(np.stack([weights, 1 - weights], -1), dtype=torch.float32, device='cuda')
    records = {key: [] for key in targets}
    for index, (scenario, oracle) in enumerate(zip(data['scenarios'], data['oracle'])):
        path = shards / f'{index:05d}.json'
        if path.exists():
            saved = json.loads(path.read_text())
        else:
            ideal, span, front = scenario_bounds(oracle)
            reference = normalize(front, ideal, span)
            set_seed(42 + index)
            state = torch.tensor(norm(scenario.state_grid(cfg['system']['battery']['soc_init'],
                                  cfg['system']['gastank']['soc_init'])), device='cuda')[None].repeat(100, 1, 1)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                actions = model.sample(state, lam, M=1, K_infer=10).cpu().numpy().reshape(100, 24, 9)
            torch.cuda.synchronize()
            sampling = time.perf_counter() - start
            before, on = decode_batch(actions, scenario, cfg, apply_recourse=False)
            pre, _, _, _ = score(before, on, scenario, cfg, ideal, span, reference)
            values, on = decode_batch(actions, scenario, cfg, apply_recourse=True)
            base, mask, points, violations = score(values, on, scenario, cfg, ideal, span, reference)
            saved = {'base': {'scenario': index, 'candidate_count': 100, 'feasible_count': int(mask.sum()),
                'before': pre, 'after': base, 'feasible_points': points.tolist(), 'violations': violations,
                'sampling_seconds_observed': sampling}}
            projected, stats = project_joint_balance(values, on, scenario, cfg)
            post, joint_mask, joint_points, joint_violations = score(projected, on, scenario, cfg, ideal, span, reference)
            if (joint_mask & mask).sum() != mask.sum():
                raise ValueError('Joint recourse destroyed a previously feasible candidate')
            saved['joint_lp'] = {'scenario': index, 'candidate_count': 100, 'feasible_count': int(joint_mask.sum()),
                'before': pre, 'after': post, 'feasible_points': joint_points.tolist(), 'violations': joint_violations,
                'sampling_seconds_observed': sampling, 'lp': stats}
            atomic_json(path, saved)
        for variant in targets:
            records[variant].append(saved[variant])
        if (index + 1) % 50 == 0:
            print(f"{family} {model_row['label']} {index + 1}/{len(data['scenarios'])} "
                  f"base={saved['base']['after']['feas']:.3f} lp={saved['joint_lp']['after']['feas']:.3f}", flush=True)
    for variant, path in targets.items():
        if path.exists():
            continue
        aggregate_result = {}
        for section in ('before', 'after'):
            rows = [{key: float(value) for key, value in row[section].items()} for row in records[variant]]
            aggregate_result[section] = {**aggregate(rows), **aggregate_coverage(rows)}
        result = {'protocol': protocols[variant], 'signature': signatures[variant],
                  'checkpoint_metadata': metadata, 'aggregate': aggregate_result,
                  'per_scenario': records[variant], 'counts': {'candidates': len(records[variant]) * 100,
                  'scenarios': len(records[variant]), 'feasible': sum(r['feasible_count'] for r in records[variant])}}
        if variant == 'joint_lp':
            result['lp'] = {key: sum(row['lp'][key] for row in records[variant]) for key in
                ('attempted', 'repaired', 'fixed_decision_infeasible', 'solver_or_audit_failures', 'seconds')}
        atomic_json(path, result)
    print('DONE', family, model_row['label'], aggregate_result['after'], flush=True)
    del model
    torch.cuda.empty_cache()
