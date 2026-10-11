"""Joint continuous recourse: fixed UC and storage directions, free powers/SOC."""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack

from src.ies.independent_audit import audit_dispatch

PROTOCOL = 'fixed_uc_signed_storage_joint_lp_v2'
POWERS = ('chp', 'gt', 'gb', 'eb', 'hp', 'bat', 'gas')


def project_joint_balance(values, commitment, scenario, cfg, *, time_limit=5.):
    started = time.perf_counter()
    out = {key: np.array(value, dtype=float, copy=True) for key, value in values.items()}
    on = np.asarray(commitment)
    horizon = cfg['horizon']['T']
    tol = cfg['milp']['feasibility_tol']
    failed = np.flatnonzero(sum(audit_dispatch(out, on, scenario, cfg).values()) > tol)
    stats = {'protocol': PROTOCOL, 'attempted': len(failed), 'repaired': 0,
             'fixed_decision_infeasible': 0, 'solver_or_audit_failures': 0,
             'solver_status_counts': {}, 'seconds': 0., 'commitment_changed': False,
             'storage_directions_changed': False, 'storage_adjusted_candidates': 0}
    if not len(failed):
        stats['seconds'] = time.perf_counter() - started
        return out, stats
    s = cfg['system']
    powers = len(POWERS) * horizon
    cols = {name: np.arange(i * horizon, (i + 1) * horizon) for i, name in enumerate(POWERS)}
    soc = {'battery': np.arange(powers, powers + horizon + 1),
           'gastank': np.arange(powers + horizon + 1, powers + 2 * (horizon + 1))}
    variables = powers + 2 * (horizon + 1)
    hours = np.arange(horizon)
    heat = np.zeros((horizon, variables))
    electric, gas = np.zeros_like(heat), np.zeros_like(heat)
    for name, coefficient in (('chp', s['chp']['eta_h']), ('gb', s['gb']['eta_h']),
                              ('eb', s['eb']['eta_h']), ('hp', s['hp']['cop'])):
        heat[hours, cols[name]] = coefficient
    for name, coefficient in (('chp', -s['chp']['eta_e']), ('gt', -s['gt']['eta_e']),
                              ('eb', 1.), ('hp', 1.), ('bat', 1.)):
        electric[hours, cols[name]] = coefficient
    for name in ('chp', 'gt', 'gb', 'gas'):
        gas[hours, cols[name]] = -1.
    ramp_rows, ramp_limits = [], []
    for name in POWERS[:5]:
        for hour in range(1, horizon):
            row = np.zeros(variables)
            row[cols[name][hour]], row[cols[name][hour - 1]] = 1., -1.
            ramp_rows.extend((row, -row))
            ramp_limits.extend((s[name]['ramp'], s[name]['ramp']))
    terminal = np.zeros((2, variables))
    for i, dev in enumerate(soc):
        terminal[i, soc[dev][-1]] = -1.
    physical = csr_matrix(np.vstack([electric, -electric, gas, *ramp_rows, *terminal]))
    physical = hstack([physical, csr_matrix((physical.shape[0], powers))], format='csr')
    deviations = hstack([eye(powers), csr_matrix((powers, variables - powers))], format='csr')
    aub = vstack([physical, hstack([deviations, -eye(powers)]),
                  hstack([-deviations, -eye(powers)])], format='csr')
    scales = [s[name]['p_max'] for name in POWERS[:5]] + [s['battery']['p_max'], s['gastank']['p_max']]
    cost = np.r_[np.zeros(variables), np.repeat(1. / np.maximum(scales, 1.), horizon)]
    for index in failed:
        target = np.concatenate([out[name][index] for name in POWERS])
        constant = scenario.load_e + out['ch'][index] - scenario.pv - scenario.wt
        bub = np.r_[s['grid']['p_max_import'] - constant, s['grid']['p_max_export'] + constant,
                    scenario.load_g, ramp_limits, -s['battery']['soc_init'], -s['gastank']['soc_init'], target, -target]
        equality_rows, equality_rhs = [heat], [scenario.load_h]
        for name, dev in (('bat', 'battery'), ('gas', 'gastank')):
            spec = s[dev]
            dynamics = np.zeros((horizon + 1, variables))
            dynamics[0, soc[dev][0]] = 1.
            dynamics[hours + 1, soc[dev][1:]] = 1.
            dynamics[hours + 1, soc[dev][:-1]] = -(1 - spec['self_discharge'])
            efficiency = np.where(out[name][index] >= 0., spec['eta_chg'], 1. / spec['eta_dis'])
            dynamics[hours + 1, cols[name]] = -efficiency * cfg['horizon']['dt']
            equality_rows.append(dynamics)
            equality_rhs.append(np.r_[spec['soc_init'], np.zeros(horizon)])
        aeq = csr_matrix(np.vstack(equality_rows))
        aeq = hstack([aeq, csr_matrix((aeq.shape[0], powers))], format='csr')
        bounds = []
        for unit, name in enumerate(POWERS[:5]):
            if name in ('chp', 'gt', 'gb'):
                minimum = max(s[name]['p_min'], cfg['uc']['min_load_frac'] * s[name]['p_max'])
                bounds.extend((minimum * y, s[name]['p_max'] * y) for y in on[index, :, unit])
            else:
                bounds.extend([(s[name]['p_min'], s[name]['p_max'])] * horizon)
        for name, dev in (('bat', 'battery'), ('gas', 'gastank')):
            bounds.extend((0., s[dev]['p_max']) if power >= 0. else (-s[dev]['p_max'], 0.)
                          for power in out[name][index])
        for dev in soc:
            bounds.extend([(s[dev]['soc_min'], s[dev]['soc_max'])] * (horizon + 1))
        bounds.extend([(0., None)] * powers)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unrecognized options detected.*')
            result = linprog(cost, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=np.concatenate(equality_rhs),
                             bounds=bounds, method='highs', options={'threads': 1, 'time_limit': time_limit,
                             'primal_feasibility_tolerance': 1e-8, 'dual_feasibility_tolerance': 1e-8})
        status = str(result.status)
        stats['solver_status_counts'][status] = stats['solver_status_counts'].get(status, 0) + 1
        if not result.success:
            stats['fixed_decision_infeasible' if result.status == 2 else 'solver_or_audit_failures'] += 1
            continue
        candidate = {key: value[index:index + 1].copy() for key, value in out.items()}
        for name in POWERS:
            candidate[name][0] = result.x[cols[name]]
        for name, dev in (('soc_bat', 'battery'), ('soc_gas', 'gastank')):
            candidate[name][0] = result.x[soc[dev]]
        candidate['grid'][0] = (constant + candidate['bat'][0] + candidate['eb'][0] + candidate['hp'][0]
                               - s['chp']['eta_e'] * candidate['chp'][0] - s['gt']['eta_e'] * candidate['gt'][0])
        candidate['g_import'][0] = (candidate['chp'][0] + candidate['gt'][0] + candidate['gb'][0]
                                    + scenario.load_g + candidate['gas'][0])
        if sum(audit_dispatch(candidate, on[index:index + 1], scenario, cfg).values())[0] > tol:
            stats['solver_or_audit_failures'] += 1
            continue
        stats['storage_adjusted_candidates'] += int(any(np.max(abs(candidate[name][0] - out[name][index])) > 1e-8
                                                        for name in ('bat', 'gas')))
        for name in out:
            out[name][index] = candidate[name][0]
        stats['repaired'] += 1
    stats['seconds'] = time.perf_counter() - started
    return out, stats
