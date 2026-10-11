"""Vectorise the scalar UC decoder over candidates, retaining the time scan.

This is a computational implementation of the existing physical decoder, not
an additional repair step. Validation compares every physical coordinate with
decode_uc_action. Constraints are audited by independent_audit, not here.
"""
from __future__ import annotations

import numpy as np


def decode_batch(actions, scenario, cfg, *, apply_recourse=True):
    a = np.asarray(actions, dtype=np.float64)
    tmax = int(cfg['horizon']['T'])
    if a.ndim != 3 or a.shape[1:] != (tmax, 9) or not np.isfinite(a).all():
        raise ValueError('actions must be finite (N, T, 9)')
    n = len(a)
    s, uc = cfg['system'], cfg['uc']
    margin, dt = float(cfg['model']['a0_margin']), float(cfg['horizon']['dt'])
    requested = a[..., 6:] > 0
    on = np.zeros_like(requested, dtype=np.int8)
    state = np.zeros((n, 3), dtype=bool)
    locked = np.zeros((n, 3), dtype=np.int64)
    for t in range(tmax):
        switching = (locked == 0) & (requested[:, t] != state)
        state = np.where(switching, requested[:, t], state)
        duration = np.where(state, int(uc['min_up']) - 1, int(uc['min_down']) - 1)
        locked = np.where(switching, duration, np.maximum(locked - 1, 0))
        on[:, t] = state
    if not cfg.get('full_diffusion', {}).get('commitment_projection', True):
        on = requested.astype(np.int8)
    u = np.clip((a[..., :4] / margin + 1) / 2, 0, 1)
    powers = {}
    for i, name in enumerate(('chp', 'gt', 'gb')):
        minimum = max(s[name]['p_min'], uc['min_load_frac'] * s[name]['p_max'])
        maximum, ramp = s[name]['p_max'], s[name]['ramp']
        request = (minimum + u[..., i] * (maximum - minimum)) * on[..., i]
        lo, hi = minimum * on[..., i].astype(float), maximum * on[..., i].astype(float)
        for t in range(tmax - 2, -1, -1):
            lo[:, t] = np.maximum(lo[:, t], lo[:, t + 1] - ramp)
            hi[:, t] = np.minimum(hi[:, t], hi[:, t + 1] + ramp)
        if np.any(lo > hi + 1e-9):
            raise ValueError('commitment schedule is incompatible with unit ramp limits')
        power = np.empty_like(request)
        power[:, 0] = np.clip(request[:, 0], lo[:, 0], hi[:, 0])
        for t in range(1, tmax):
            power[:, t] = np.clip(request[:, t], np.maximum(lo[:, t], power[:, t - 1] - ramp),
                                  np.minimum(hi[:, t], power[:, t - 1] + ramp))
        powers[name] = power
    powers['eb'] = s['eb']['p_min'] + u[..., 3] * (s['eb']['p_max'] - s['eb']['p_min'])
    powers['bat'] = np.clip(a[..., 4] / margin, -1, 1) * s['battery']['p_max']
    powers['gas'] = np.clip(a[..., 5] / margin, -1, 1) * s['gastank']['p_max']
    names = ('grid', 'chp', 'gt', 'gb', 'eb', 'hp', 'ch', 'bat', 'gas', 'g_import')
    out = {name: np.empty((n, tmax), dtype=float) for name in names}
    for key, dev in (('soc_bat', 'battery'), ('soc_gas', 'gastank')):
        out[key] = np.empty((n, tmax + 1), dtype=float)
        out[key][:, 0] = s[dev]['soc_init']

    def ramp_one(request, name, t):
        lower, upper = s[name]['p_min'], s[name]['p_max']
        value = np.clip(request, lower, upper)
        if t:
            prev, ramp = out[name][:, t - 1], s[name]['ramp']
            value = np.clip(value, np.maximum(lower, prev - ramp), np.minimum(upper, prev + ramp))
        scale = max(1, abs(lower), abs(upper))
        value = np.where(abs(value - lower) <= 1e-8 * scale, lower, value)
        return np.where(abs(value - upper) <= 1e-8 * scale, upper, value)

    def storage(request, soc, dev, remaining):
        d = s[dev]
        sd, pmax = d['self_discharge'], d['p_max']
        decay = (1 - sd) ** remaining
        charge_sum = remaining if sd == 0 else (1 - decay) / sd
        floor = min(max(d['soc_min'], (d['soc_init'] - d['eta_chg'] * pmax * dt * charge_sum) / decay), d['soc_max'])
        base = (1 - sd) * soc
        hi = (d['soc_max'] - base) / (d['eta_chg'] * dt)
        lo = np.where(base >= floor, -(base - floor) * d['eta_dis'] / dt,
                      (floor - base) / (d['eta_chg'] * dt))
        feasible_lo, feasible_hi = np.maximum(lo, -pmax), np.minimum(hi, pmax)
        power = np.clip(np.clip(request, -pmax, pmax), feasible_lo, feasible_hi)
        fallback = np.where(hi < -pmax, -pmax, np.where(lo > pmax, pmax, np.clip(hi, -pmax, pmax)))
        power = np.where(feasible_lo <= feasible_hi, power, fallback)
        next_soc = base + d['eta_chg'] * np.maximum(power, 0) * dt - np.maximum(-power, 0) * dt / d['eta_dis']
        return power, next_soc

    for t in range(tmax):
        for name in ('chp', 'gt', 'gb', 'eb'):
            out[name][:, t] = ramp_one(powers[name][:, t], name, t)
        out['bat'][:, t], out['soc_bat'][:, t + 1] = storage(powers['bat'][:, t], out['soc_bat'][:, t], 'battery', tmax - t - 1)
        chp, gt, gb, eb, bat = [out[name][:, t] for name in ('chp', 'gt', 'gb', 'eb', 'bat')]
        fixed_heat = s['chp']['eta_h'] * chp + s['gb']['eta_h'] * gb + s['eb']['eta_h'] * eb
        hp = ramp_one((scenario.load_h[t] - fixed_heat) / s['hp']['cop'], 'hp', t)
        ch = ramp_one(np.full(n, scenario.load_c[t] / s['ch']['cop']), 'ch', t)
        short = np.maximum(scenario.load_h[t] - fixed_heat - s['hp']['cop'] * hp, 0)
        grid_without_eb = (scenario.load_e[t] + hp + ch + bat - scenario.pv[t] - scenario.wt[t]
                           - s['chp']['eta_e'] * chp - s['gt']['eta_e'] * gt)
        if apply_recourse:
            upper = s['eb']['p_max'] if t == 0 else np.minimum(s['eb']['p_max'], out['eb'][:, t - 1] + s['eb']['ramp'])
            headroom = np.maximum(0, s['grid']['p_max_import'] - grid_without_eb - eb)
            add = np.minimum(np.minimum(short / s['eb']['eta_h'], np.maximum(0, upper - eb)), headroom)
            eb += add
            fixed_heat += s['eb']['eta_h'] * add
            short = np.maximum(scenario.load_h[t] - fixed_heat - s['hp']['cop'] * hp, 0)
            upper = s['gb']['p_max'] if t == 0 else np.minimum(s['gb']['p_max'], out['gb'][:, t - 1] + s['gb']['ramp'])
            add = np.minimum(short / s['gb']['eta_h'], np.maximum(0, upper - gb))
            gb += np.where(gb > 1e-9, add, 0)
        gas_request = powers['gas'][:, t]
        if apply_recourse:
            gas_request = np.maximum(gas_request, -(chp + gt + gb + scenario.load_g[t]))
        out['gas'][:, t], out['soc_gas'][:, t + 1] = storage(gas_request, out['soc_gas'][:, t], 'gastank', tmax - t - 1)
        out['grid'][:, t] = np.clip(grid_without_eb + eb, -s['grid']['p_max_export'], s['grid']['p_max_import'])
        out['g_import'][:, t] = np.maximum(chp + gt + gb + scenario.load_g[t] + out['gas'][:, t], 0)
        out['hp'][:, t], out['ch'][:, t] = hp, ch
    return out, on
