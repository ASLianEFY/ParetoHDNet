"""Pareto normalization, quality metrics and JSON-safe aggregation."""
import math
import numpy as np
from src import metrics as M

HV_REF = np.array([1.05, 1.05])

def _objective_array(points, name='points', *, finite=True):
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f'{name} must have shape (n, 2), got {pts.shape}')
    if finite and (not np.all(np.isfinite(pts))):
        raise ValueError(f'{name} must contain only finite values')
    return pts


def scenario_bounds(oracle_pts):
    oracle_pts = _objective_array(oracle_pts, 'oracle_pts')
    if len(oracle_pts) == 0:
        raise ValueError('oracle_pts must not be empty')
    pf = M.pareto_filter(oracle_pts)
    ideal = pf.min(axis=0)
    nadir = pf.max(axis=0)
    span = np.where(nadir - ideal < 1e-09, 1.0, nadir - ideal)
    return (ideal, span, pf)


def normalize(points, ideal, span):
    points = _objective_array(points)
    ideal = np.asarray(ideal, dtype=float)
    span = np.asarray(span, dtype=float)
    if ideal.shape != (2,) or span.shape != (2,):
        raise ValueError('ideal and span must each have shape (2,)')
    if not np.all(np.isfinite(ideal)) or not np.all(np.isfinite(span)):
        raise ValueError('ideal and span must be finite')
    if np.any(span <= 0.0):
        raise ValueError('span must be strictly positive')
    return (points - ideal) / span


def method_metrics(points_norm, ref_front_norm):
    points_norm = _objective_array(points_norm, 'points_norm')
    ref_front_norm = _objective_array(ref_front_norm, 'ref_front_norm')
    if len(ref_front_norm) == 0:
        raise ValueError('ref_front_norm must not be empty')
    pf = M.pareto_filter(points_norm)
    return {'hv': M.hypervolume_2d(pf, HV_REF), 'igd': M.igd(pf, ref_front_norm), 'igdp': M.igd_plus(pf, ref_front_norm), 'spacing': M.spacing(pf), 'n': int(len(pf))}


def aggregate(rows):
    if not rows:
        raise ValueError('cannot aggregate an empty row collection')
    out = {}
    for key in ('hv', 'igd', 'igdp', 'spacing', 'feas', 'n'):
        try:
            vals = np.asarray([row[key] for row in rows], dtype=float)
        except KeyError as exc:
            raise ValueError(f'metric row is missing {exc.args[0]!r}') from exc
        if np.any(np.isnan(vals)):
            raise ValueError(f'metric {key!r} contains NaN')
        with np.errstate(invalid='ignore'):
            value = float(np.mean(vals))
        if math.isnan(value):
            raise ValueError(f'metric {key!r} has an undefined aggregate')
        out[key] = value
    return out


def aggregate_coverage(rows):
    valid = [row for row in rows if int(row['n']) > 0]
    total = len(rows)
    return {'scenarios': total, 'valid_front_scenarios': len(valid), 'empty_front_scenarios': total - len(valid), 'empty_front_rate': (total - len(valid)) / total, 'conditional_igd': float(np.mean([row['igd'] for row in valid])) if valid else float('inf'), 'conditional_igdp': float(np.mean([row['igdp'] for row in valid])) if valid else float('inf')}


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            string_key = str(key)
            if string_key in converted:
                raise ValueError(f'evaluation report has colliding dictionary key {string_key!r}')
            converted[string_key] = json_safe(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError('evaluation reports must not contain NaN')
        if math.isinf(value):
            return 'inf' if value > 0 else '-inf'
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'report value of type {type(value).__name__} is not serializable')
