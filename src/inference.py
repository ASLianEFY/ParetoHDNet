"""ParetoHDNet loading and strict synthetic candidate-set evaluation."""
import copy
import hashlib
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from src.artifacts import checkpoint_weight_key
from src.data.dataset import StateNormalizer
from src.evaluate import aggregate, aggregate_coverage, json_safe, method_metrics, normalize, scenario_bounds
from src.ies.batch_decode import decode_batch
from src.ies.independent_audit import PROTOCOL, audit_dispatch, objective_values
from src.utils import RESULTS_DIR, set_seed
DEFAULT_OUTPUT = RESULTS_DIR / 'paretohdnet'

def load_policy(kind, checkpoint=None, strategy=None, device='cuda'):
    if kind != 'paretohdnet':
        raise ValueError('This package contains ParetoHDNet only.')
    from src.models.paretohdnet import ParetoHDNet
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = copy.deepcopy(saved['cfg'])
    policy = ParetoHDNet(cfg).to(device)
    policy.load_state_dict(saved[checkpoint_weight_key(saved)])
    policy.eval()
    return policy, StateNormalizer.load(saved['normalizer']), cfg, saved.get('metadata', {})


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def _row(points, mask, ideal, span, reference):
    row = method_metrics(normalize(points[mask], ideal, span), reference)
    row['feas'] = float(np.mean(mask))
    return row


@torch.no_grad()
def evaluate_job(job, scenarios, oracle, output, *, device='cuda', limit=None, scenario_ids=None,
                 oracle_path=None, scenario_path=None, source_protocol=None):
    output = Path(output)
    checkpoint = Path(job['checkpoint']) if job.get('checkpoint') else None
    ids = list(range(len(scenarios))) if scenario_ids is None else list(scenario_ids)
    if limit is not None:
        ids = ids[:limit]
    if not ids or len(scenarios) != len(oracle):
        raise ValueError('non-empty scenarios and aligned oracle are required')
    protocol = {**job, 'feasibility_protocol': PROTOCOL, 'audit': 'independent_equations_v1',
                'decoder': 'numpy_candidate_vectorized_v1', 'evaluation_seed': 42,
                'rng': 'torch.manual_seed(42 + scenario_id) before each scenario',
                'scenario_ids': ids, 'candidates_per_scenario': job['K_lambda'] * job['M_samples'],
                'milp_repair': False, 'apply_recourse': True,
                'checkpoint_sha256': file_digest(checkpoint) if checkpoint else None,
                'oracle_path': str(oracle_path) if oracle_path else None,
                'oracle_sha256': file_digest(oracle_path) if oracle_path else None,
                'scenario_sha256': file_digest(scenario_path) if scenario_path else None,
                'source_protocol': source_protocol,
                'source_sha256': {name: file_digest(Path(__file__).parent / name) for name in
                                  ('inference.py', 'ies/batch_decode.py', 'ies/independent_audit.py', 'ies/system.py')}}
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    target = output / f"{job['id']}.json"
    if target.exists():
        previous = json.loads(target.read_text())
        if previous.get('signature') != signature:
            raise ValueError(f'existing result has a different protocol: {target}')
        print(f"SKIP completed {job['id']}", flush=True)
        return previous
    shard_dir = output / 'partial' / job['id']
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest = shard_dir / 'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()).get('signature') != signature:
        raise ValueError(f'partial result has a different protocol: {manifest}')
    atomic_json(manifest, {'signature': signature, 'protocol': protocol})
    active_device = 'cpu' if job['kind'] == 'heuristic' else device
    old_cudnn = torch.backends.cudnn.enabled
    if job['kind'] == 'gru':
        torch.backends.cudnn.enabled = False
    set_seed(42)
    model, norm, cfg, metadata = load_policy(job['kind'], checkpoint, job.get('strategy'), active_device)
    lambdas = np.linspace(0, 1, job['K_lambda'])
    lam = torch.tensor(np.stack([lambdas, 1 - lambdas], axis=-1), dtype=torch.float32, device=active_device)
    training = metadata.get('training', {})
    protocol['training_seed'] = training.get('training_seed', cfg['seed'] if checkpoint else None)
    # Hardware is deliberately recorded separately: quality resume is allowed
    # only on the same environment because latency observations are pooled.
    environment = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                   'device': active_device, 'device_name': torch.cuda.get_device_name() if str(active_device).startswith('cuda') else platform.processor(),
                   'torch_threads': torch.get_num_threads(), 'cpu_workers': 1, 'cudnn': torch.backends.cudnn.enabled}
    env_path = shard_dir / 'environment.json'
    if env_path.exists() and json.loads(env_path.read_text()) != environment:
        raise ValueError('cannot pool runtime shards from different environments')
    atomic_json(env_path, environment)
    records = []
    warm = scenarios[ids[0]]
    state = torch.tensor(norm(warm.state_grid(cfg['system']['battery']['soc_init'], cfg['system']['gastank']['soc_init'])), device=active_device).unsqueeze(0).repeat(len(lam), 1, 1)
    for _ in range(3):
        model.sample(state, lam, M=job['M_samples'], K_infer=job['K_infer'])
    _sync(active_device)
    for number, index in enumerate(ids):
        shard = shard_dir / f'{index:05d}.json'
        if shard.exists():
            records.append(json.loads(shard.read_text()))
            continue
        sc = scenarios[index]
        ideal, span, pf = scenario_bounds(oracle[index])
        reference = normalize(pf, ideal, span)
        set_seed(42 + index)
        started = time.perf_counter()
        state = torch.tensor(norm(sc.state_grid(cfg['system']['battery']['soc_init'], cfg['system']['gastank']['soc_init'])), device=active_device).unsqueeze(0).repeat(len(lam), 1, 1)
        _sync(active_device)
        sampling_start = time.perf_counter()
        tensor = model.sample(state, lam, M=job['M_samples'], K_infer=job['K_infer'])
        _sync(active_device)
        sampling_seconds = time.perf_counter() - sampling_start
        samples = tensor.cpu().numpy()
        actions = samples.reshape(-1, cfg['horizon']['T'], 9)
        decode_start = time.perf_counter()
        after, on = decode_batch(actions, sc, cfg, apply_recourse=True)
        decode_seconds = time.perf_counter() - decode_start
        strict_start = time.perf_counter()
        violations = audit_dispatch(after, on, sc, cfg)
        points = objective_values(after, on, sc, cfg)
        total = sum(violations.values())
        tolerance = float(cfg['milp']['feasibility_tol'])
        feasible = total <= tolerance
        dispatch_seconds = time.perf_counter() - started
        audit_seconds = time.perf_counter() - strict_start
        # Before/legacy are diagnostic work excluded from production latency.
        before, pre_on = decode_batch(actions, sc, cfg, apply_recourse=False)
        pre_viol = audit_dispatch(before, pre_on, sc, cfg)
        pre_points = objective_values(before, pre_on, sc, cfg)
        pre_feasible = sum(pre_viol.values()) <= tolerance
        legacy_feasible = total - violations['heat_surplus'] <= tolerance
        metric_start = time.perf_counter()
        metrics = _row(points, feasible, ideal, span, reference)
        pre_metrics = _row(pre_points, pre_feasible, ideal, span, reference)
        legacy_metrics = _row(points, legacy_feasible, ideal, span, reference)
        reference_metrics = _row(np.asarray(oracle[index]), np.ones(len(oracle[index]), dtype=bool), ideal, span, reference)
        metric_seconds = time.perf_counter() - metric_start
        k, m = job['K_lambda'], job['M_samples']
        on_km = on.reshape(k, m, cfg['horizon']['T'], 3)
        modes = [len(np.unique(group.reshape(m, -1), axis=0)) for group in on_km]
        row = {'scenario': index, 'after': metrics, 'before': pre_metrics, 'legacy_heat_unchecked': legacy_metrics,
               'reference': reference_metrics, 'candidate_count': len(actions), 'feasible_count': int(feasible.sum()),
               'feasible_preference_fraction': float(feasible.reshape(k, m).any(axis=1).mean()),
               'all_preferences_have_feasible_candidate': bool(feasible.reshape(k, m).any(axis=1).all()),
               'feasible_to_infeasible': int((pre_feasible & ~feasible).sum()),
               'heat_surplus_candidates': int((violations['heat_surplus'] > tolerance).sum()),
               'unique_commitments': len(np.unique(on.reshape(len(on), -1), axis=0)),
               'within_preference_unique_commitments_mean': float(np.mean(modes)),
               'violations': {name: {'sum': float(v.sum()), 'max': float(v.max()), 'count': int((v > tolerance).sum())} for name, v in violations.items()},
               'latency': {'sampling_seconds': sampling_seconds, 'decode_seconds': decode_seconds,
                           'audit_seconds': audit_seconds, 'dispatch_seconds': dispatch_seconds,
                           'metrics_seconds': metric_seconds},
               'feasible_points': points[feasible].tolist()}
        atomic_json(shard, row)
        records.append(row)
        if (number + 1) % 50 == 0 or number == 0 or number + 1 == len(ids):
            print(f"{job['id']}: {number + 1}/{len(ids)} strict_feas={metrics['feas']:.3f}", flush=True)
    # JSON represents infinity as a string; convert metric rows back for aggregation.
    aggregates = {}
    for key in ('after', 'before', 'legacy_heat_unchecked', 'reference'):
        rows = [{name: float(value) for name, value in record[key].items()} for record in records]
        aggregates[key] = {**aggregate(rows), **aggregate_coverage(rows)}
    latency = {key: {'mean': float(np.mean(vals)), 'median': float(np.median(vals)),
                     'p95': float(np.quantile(vals, .95)), 'sum': float(np.sum(vals))}
               for key in records[0]['latency']
               for vals in [[record['latency'][key] for record in records]]}
    result = {'schema_version': 2, 'signature': signature, 'protocol': protocol, 'environment': environment,
              'checkpoint_metadata': metadata, 'aggregate': aggregates, 'latency': latency,
              'per_scenario': records, 'counts': {'scenarios': len(records),
              'candidates': sum(r['candidate_count'] for r in records), 'feasible': sum(r['feasible_count'] for r in records),
              'heat_surplus_candidates': sum(r['heat_surplus_candidates'] for r in records)}}
    atomic_json(target, result)
    print(f"DONE {job['id']} {aggregates['after']}", flush=True)
    del model
    torch.backends.cudnn.enabled = old_cudnn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
