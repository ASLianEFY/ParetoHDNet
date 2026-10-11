from __future__ import annotations
import numpy as np
import pulp
UC_UNITS = ('chp', 'gt', 'gb')
MIN_LOAD_FRAC = 0.3
START_COST = 20.0
MIN_UP = 3
MIN_DOWN = 2

def build_problem(
    system,
    sc,
    lam,
    cfg,
    horizon,
    *,
    target_a0=None,
    relax_binaries=False,
    emissions_upper_bound=None,
):
    s = system.s
    dt = system.dt
    rng_t = range(horizon)
    prob = pulp.LpProblem(f'ies_uc_milp_{horizon}', pulp.LpMinimize)

    def var(name, low, high, cat='Continuous'):
        return [pulp.LpVariable(f'{name}_{t}', lowBound=low, upBound=high, cat=cat) for t in rng_t]
    chp = var('chp', 0.0, s['chp']['p_max'])
    gt = var('gt', 0.0, s['gt']['p_max'])
    gb = var('gb', 0.0, s['gb']['p_max'])
    eb = var('eb', s['eb']['p_min'], s['eb']['p_max'])
    hp = var('hp', s['hp']['p_min'], s['hp']['p_max'])
    ch = var('ch', s['ch']['p_min'], s['ch']['p_max'])
    bat_c = var('batc', 0, s['battery']['p_max'])
    bat_d = var('batd', 0, s['battery']['p_max'])
    gas_i = var('gasi', 0, s['gastank']['p_max'])
    gas_w = var('gasw', 0, s['gastank']['p_max'])
    g_imp = var('gimp', 0, None)
    p_imp = var('pimp', 0, s['grid']['p_max_import'])
    p_exp = var('pexp', 0, s['grid']['p_max_export'])
    soc_b = [pulp.LpVariable(f'socb_{t}', s['battery']['soc_min'], s['battery']['soc_max']) for t in range(horizon + 1)]
    soc_g = [pulp.LpVariable(f'socg_{t}', s['gastank']['soc_min'], s['gastank']['soc_max']) for t in range(horizon + 1)]
    units = {'chp': chp, 'gt': gt, 'gb': gb}
    uc_cfg = cfg.get('uc', {})
    min_load_frac = float(uc_cfg.get('min_load_frac', MIN_LOAD_FRAC))
    start_cost = float(uc_cfg.get('start_cost', START_COST))
    min_up = int(uc_cfg.get('min_up', MIN_UP))
    min_down = int(uc_cfg.get('min_down', MIN_DOWN))
    commitment_category = 'Continuous' if relax_binaries else 'Binary'
    on = {name: var(f'on_{name}', 0, 1, commitment_category) for name in UC_UNITS}
    startup = {name: var(f'start_{name}', 0, 1, commitment_category) for name in UC_UNITS}
    shutdown = {name: var(f'stop_{name}', 0, 1, commitment_category) for name in UC_UNITS}
    binary_count = 3 * len(UC_UNITS) * horizon
    for name in UC_UNITS:
        minimum = max(float(s[name]['p_min']), min_load_frac * s[name]['p_max'])
        for t in rng_t:
            prob += units[name][t] <= s[name]['p_max'] * on[name][t]
            prob += units[name][t] >= minimum * on[name][t]
            previous_on = 0 if t == 0 else on[name][t - 1]
            prob += on[name][t] - previous_on == startup[name][t] - shutdown[name][t]
            up_window = range(max(0, t - min_up + 1), t + 1)
            down_window = range(max(0, t - min_down + 1), t + 1)
            prob += pulp.lpSum((startup[name][tau] for tau in up_window)) <= on[name][t]
            prob += pulp.lpSum((shutdown[name][tau] for tau in down_window)) <= 1 - on[name][t]
    ee, eh = (s['chp']['eta_e'], s['chp']['eta_h'])
    for t in rng_t:
        prob += sc.pv[t] + sc.wt[t] + p_imp[t] + ee * chp[t] + s['gt']['eta_e'] * gt[t] + bat_d[t] == sc.load_e[t] + eb[t] + hp[t] + ch[t] + p_exp[t] + bat_c[t]
        prob += eh * chp[t] + s['gb']['eta_h'] * gb[t] + s['eb']['eta_h'] * eb[t] + s['hp']['cop'] * hp[t] == sc.load_h[t]
        prob += s['ch']['cop'] * ch[t] == sc.load_c[t]
        prob += g_imp[t] + gas_w[t] == chp[t] + gt[t] + gb[t] + sc.load_g[t] + gas_i[t]
    prob += soc_b[0] == s['battery']['soc_init']
    prob += soc_g[0] == s['gastank']['soc_init']
    for t in rng_t:
        prob += soc_b[t + 1] == (1 - s['battery']['self_discharge']) * soc_b[t] + s['battery']['eta_chg'] * bat_c[t] * dt - bat_d[t] * dt / s['battery']['eta_dis']
        prob += soc_g[t + 1] == (1 - s['gastank']['self_discharge']) * soc_g[t] + s['gastank']['eta_chg'] * gas_i[t] * dt - gas_w[t] * dt / s['gastank']['eta_dis']
    prob += soc_b[horizon] >= s['battery']['soc_init']
    prob += soc_g[horizon] >= s['gastank']['soc_init']
    for name, values in {'chp': chp, 'gt': gt, 'gb': gb, 'eb': eb, 'hp': hp, 'ch': ch}.items():
        ramp = s[name]['ramp']
        for t in range(1, horizon):
            prob += values[t] - values[t - 1] <= ramp
            prob += values[t - 1] - values[t] <= ramp
    degradation = s['battery']['degr_cost']
    carbon_gas = cfg['data']['carbon_gas']
    economic = pulp.lpSum((sc.price_buy[t] * p_imp[t] * dt - sc.price_sell[t] * p_exp[t] * dt + sc.price_gas[t] * g_imp[t] * dt + degradation * (bat_c[t] + bat_d[t]) * dt for t in rng_t))
    economic += pulp.lpSum((start_cost * startup[name][t] for name in UC_UNITS for t in rng_t))
    emissions = pulp.lpSum((sc.carbon[t] * p_imp[t] * dt + carbon_gas * g_imp[t] * dt for t in rng_t))
    if emissions_upper_bound is not None:
        emissions_upper_bound = float(emissions_upper_bound)
        if not np.isfinite(emissions_upper_bound) or emissions_upper_bound < 0.0:
            raise ValueError('emissions_upper_bound must be finite and non-negative')
        if target_a0 is not None:
            raise ValueError('emissions_upper_bound is not supported in repair mode')
        feasibility_tolerance = float(cfg['milp'].get('feasibility_tol', 0.001))
        numerical_tolerance = min(1e-5, feasibility_tolerance * 0.01)
        prob += emissions <= emissions_upper_bound + numerical_tolerance
        prob += economic
    elif target_a0 is None:
        prob += float(lam) * economic + (1 - float(lam)) * emissions
    else:
        target_a0 = np.asarray(target_a0, dtype=float)
        if target_a0.shape != (horizon, 6) or not np.all(np.isfinite(target_a0)):
            raise ValueError(f'target_a0 must be a finite ({horizon}, 6) array')
        raw = system.raw_from_a0(target_a0)
        targets = {'chp': system.scale_box(raw[:, 0], 'chp'), 'gt': system.scale_box(raw[:, 1], 'gt'), 'gb': system.scale_box(raw[:, 2], 'gb'), 'eb': system.scale_box(raw[:, 3], 'eb'), 'bat': system.scale_signed(raw[:, 4], 'battery'), 'gas': system.scale_signed(raw[:, 5], 'gastank')}
        realised = {'chp': chp, 'gt': gt, 'gb': gb, 'eb': eb, 'bat': [bat_c[t] - bat_d[t] for t in rng_t], 'gas': [gas_i[t] - gas_w[t] for t in rng_t]}
        scales = {'chp': s['chp']['p_max'], 'gt': s['gt']['p_max'], 'gb': s['gb']['p_max'], 'eb': s['eb']['p_max'], 'bat': s['battery']['p_max'], 'gas': s['gastank']['p_max']}
        deviations = []
        for name, values in realised.items():
            for t in rng_t:
                absolute = pulp.LpVariable(f'repair_abs_{name}_{t}', lowBound=0)
                prob += absolute >= values[t] - float(targets[name][t])
                prob += absolute >= float(targets[name][t]) - values[t]
                deviations.append(absolute / scales[name])
        prob += pulp.lpSum(deviations)
    prob._ies_variables = {'chp': chp, 'gt': gt, 'gb': gb, 'eb': eb, 'hp': hp, 'ch': ch, 'bat_c': bat_c, 'bat_d': bat_d, 'gas_i': gas_i, 'gas_w': gas_w, 'on': on}
    prob._ies_objectives = {'economic': economic, 'emissions': emissions}
    return (prob, binary_count)
