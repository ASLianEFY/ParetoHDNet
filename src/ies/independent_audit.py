"""Recompute physical constraints from realised trajectories, without decoder residuals.

All continuous residuals are summed in the same power/SOC units as the original
protocol. Heat dumping is not permitted: heat supply must equal heat demand.
The independent implementation also checks boxes, storage dynamics and initial
conditions, which a projected trajectory normally satisfies by construction.
"""
from __future__ import annotations

import numpy as np

PROTOCOL = 'strict_carrier_balance_v2'
UNITS = ('chp', 'gt', 'gb')


def audit_dispatch(values: dict, commitment, scenario, cfg: dict) -> dict:
    """Return one violation per candidate and category; accepts (..., T) arrays."""
    p = {k: np.asarray(v, dtype=np.float64) for k, v in values.items()}
    on = np.asarray(commitment, dtype=np.float64)
    s = cfg['system']
    dt = float(cfg['horizon']['dt'])
    t = int(cfg['horizon']['T'])
    shape = p['chp'].shape
    if shape[-1] != t or on.shape != shape + (3,):
        raise ValueError('dispatch and commitment dimensions disagree with horizon')
    required = ('grid', *UNITS, 'eb', 'hp', 'ch', 'bat', 'gas', 'g_import', 'soc_bat', 'soc_gas')
    for name in required:
        expected = shape[:-1] + (t + 1,) if name.startswith('soc_') else shape
        if p[name].shape != expected:
            raise ValueError(f'invalid shape for {name}')
    if not all(np.isfinite(p[k]).all() for k in required) or not np.isfinite(on).all():
        raise ValueError('audit requires finite dispatch and commitment values')
    positive = lambda x: np.maximum(x, 0.0)
    total = lambda x: np.sum(x, axis=-1)
    heat = (s['chp']['eta_h'] * p['chp'] + s['gb']['eta_h'] * p['gb']
            + s['eb']['eta_h'] * p['eb'] + s['hp']['cop'] * p['hp'] - scenario.load_h)
    electric = (scenario.pv + scenario.wt + p['grid'] + s['chp']['eta_e'] * p['chp']
                + s['gt']['eta_e'] * p['gt'] - p['bat'] - scenario.load_e
                - p['eb'] - p['hp'] - p['ch'])
    cool = s['ch']['cop'] * p['ch'] - scenario.load_c
    gas = p['g_import'] - p['gas'] - p['chp'] - p['gt'] - p['gb'] - scenario.load_g
    out = {'heat_short': total(positive(-heat)), 'heat_surplus': total(positive(heat)),
           'elec_imbalance': total(abs(electric)), 'cool_short': total(positive(-cool)),
           'cool_surplus': total(positive(cool)), 'gas_balance': total(abs(gas))}
    zeros = np.zeros(shape[:-1], dtype=np.float64)
    for key in ('capacity', 'ramp', 'soc_box', 'soc_dynamics', 'soc_initial', 'soc_terminal',
                'uc_min_load', 'uc_off_output', 'uc_min_up', 'uc_min_down', 'uc_binary'):
        out[key] = zeros.copy()
    for name in (*UNITS, 'eb', 'hp', 'ch'):
        lo = 0.0 if name in UNITS else s[name]['p_min']
        out['capacity'] += total(positive(lo - p[name]) + positive(p[name] - s[name]['p_max']))
        out['ramp'] += total(positive(abs(np.diff(p[name], axis=-1)) - s[name]['ramp']))
    out['capacity'] += total(positive(p['grid'] - s['grid']['p_max_import'])
                             + positive(-s['grid']['p_max_export'] - p['grid'])
                             + positive(-p['g_import']))
    for name, soc_key, dev in (('bat', 'soc_bat', 'battery'), ('gas', 'soc_gas', 'gastank')):
        spec = s[dev]
        soc = p[soc_key]
        expected = ((1 - spec['self_discharge']) * soc[..., :-1]
                    + spec['eta_chg'] * positive(p[name]) * dt
                    - positive(-p[name]) * dt / spec['eta_dis'])
        out['capacity'] += total(positive(abs(p[name]) - spec['p_max']))
        out['soc_box'] += total(positive(spec['soc_min'] - soc) + positive(soc - spec['soc_max']))
        out['soc_dynamics'] += total(abs(soc[..., 1:] - expected))
        out['soc_initial'] += abs(soc[..., 0] - spec['soc_init'])
        out['soc_terminal'] += positive(spec['soc_init'] - soc[..., -1])
    up, down = int(cfg['uc']['min_up']), int(cfg['uc']['min_down'])
    for i, name in enumerate(UNITS):
        y = on[..., i]
        minimum = max(s[name]['p_min'], cfg['uc']['min_load_frac'] * s[name]['p_max'])
        out['uc_binary'] += total(np.minimum(abs(y), abs(y - 1)))
        out['uc_min_load'] += total(positive(minimum * y - p[name]))
        out['uc_off_output'] += total(positive(p[name] - s[name]['p_max'] * y))
        delta = np.diff(y, axis=-1, prepend=np.zeros_like(y[..., :1]))
        starts, stops = positive(delta), positive(-delta)
        for k in range(t):
            out['uc_min_up'] += positive(total(starts[..., max(0, k - up + 1):k + 1]) - y[..., k])
            out['uc_min_down'] += positive(total(stops[..., max(0, k - down + 1):k + 1]) - (1 - y[..., k]))
    return out


def objective_values(values, commitment, scenario, cfg):
    s, dt = cfg['system'], float(cfg['horizon']['dt'])
    grid, gas = values['grid'], values['g_import']
    starts = np.maximum(np.diff(commitment, axis=-2, prepend=np.zeros_like(commitment[..., :1, :])), 0)
    cost = np.sum((scenario.price_buy * np.maximum(grid, 0)
                   - scenario.price_sell * np.maximum(-grid, 0)
                   + scenario.price_gas * gas + s['battery']['degr_cost'] * abs(values['bat'])) * dt, axis=-1)
    cost += cfg['uc']['start_cost'] * starts.sum(axis=(-2, -1))
    carbon = np.sum((scenario.carbon * np.maximum(grid, 0) + cfg['data']['carbon_gas'] * gas) * dt, axis=-1)
    return np.stack([cost, carbon], axis=-1)
