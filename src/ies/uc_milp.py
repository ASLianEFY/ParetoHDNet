from __future__ import annotations
import numpy as np
import pulp
from .milp import DispatchSolveError
from .system import IESSystem
from .uc_problem import UC_UNITS, build_problem

def uc_settings(cfg):
    values = cfg.get('uc', {})
    return {'min_load_frac': float(values.get('min_load_frac', 0.3)), 'start_cost': float(values.get('start_cost', 20.0)), 'min_up': int(values.get('min_up', 3)), 'min_down': int(values.get('min_down', 2))}

def uc_action_from_physical(system, cfg, chp, gt, gb, eb, bat, gas, commitment):
    margin = system.margin
    settings = uc_settings(cfg)
    powers = (np.asarray(chp), np.asarray(gt), np.asarray(gb))
    continuous = []
    for name, power in zip(UC_UNITS, powers):
        minimum = max(float(system.s[name]['p_min']), settings['min_load_frac'] * system.s[name]['p_max'])
        span = system.s[name]['p_max'] - minimum
        on = np.asarray(commitment[name], dtype=bool)
        encoded = np.full_like(power, -margin, dtype=float)
        encoded[on] = margin * (2.0 * (power[on] - minimum) / span - 1.0)
        continuous.append(encoded)
    eb_raw = margin * (2.0 * (np.asarray(eb) - system.s['eb']['p_min']) / (system.s['eb']['p_max'] - system.s['eb']['p_min']) - 1.0)
    continuous.extend([eb_raw, margin * np.asarray(bat) / system.s['battery']['p_max'], margin * np.asarray(gas) / system.s['gastank']['p_max']])
    switches = [np.where(np.asarray(commitment[name], bool), margin, -margin) for name in UC_UNITS]
    return np.stack([*continuous, *switches], axis=1).astype(np.float32)

def _project_switches(logits, cfg):
    requested = np.asarray(logits, dtype=float) > 0.0
    T, units = requested.shape
    projected = np.zeros((T, units), dtype=np.int8)
    settings = uc_settings(cfg)
    for unit in range(units):
        state, locked = (0, 0)
        for t in range(T):
            if locked:
                locked -= 1
            else:
                desired = int(requested[t, unit])
                if desired != state:
                    state = desired
                    duration = settings['min_up'] if state else settings['min_down']
                    locked = max(0, duration - 1)
            projected[t, unit] = state
    return projected


def _project_committed_power(requested, on, minimum, maximum, ramp):
    requested = np.asarray(requested, dtype=float)
    on = np.asarray(on, dtype=bool)
    lower = np.where(on, float(minimum), 0.0)
    upper = np.where(on, float(maximum), 0.0)
    viable_lower = lower.copy()
    viable_upper = upper.copy()
    for t in range(len(requested) - 2, -1, -1):
        viable_lower[t] = max(viable_lower[t], viable_lower[t + 1] - ramp)
        viable_upper[t] = min(viable_upper[t], viable_upper[t + 1] + ramp)
        if viable_lower[t] > viable_upper[t] + 1e-9:
            raise ValueError('commitment schedule is incompatible with unit ramp limits')
    projected = np.empty_like(requested)
    projected[0] = np.clip(requested[0], viable_lower[0], viable_upper[0])
    for t in range(1, len(requested)):
        feasible_lower = max(viable_lower[t], projected[t - 1] - ramp)
        feasible_upper = min(viable_upper[t], projected[t - 1] + ramp)
        if feasible_lower > feasible_upper + 1e-9:
            raise ValueError('commitment schedule has no ramp-feasible power trajectory')
        projected[t] = np.clip(requested[t], feasible_lower, feasible_upper)
    return projected

def decode_uc_action(system, action, sc, cfg, *, apply_recourse=True):
    action = np.asarray(action, dtype=float)
    T = int(cfg['horizon']['T'])
    if action.shape != (T, 9) or not np.all(np.isfinite(action)):
        raise ValueError(f'UC action must be a finite ({T}, 9) array')
    use_projection = bool(cfg.get('full_diffusion', {}).get('commitment_projection', True))
    on = _project_switches(action[:, 6:9], cfg) if use_projection else (action[:, 6:9] > 0.0).astype(np.int8)
    margin = system.margin
    u = np.clip((action[:, :4] / margin + 1.0) / 2.0, 0.0, 1.0)
    settings = uc_settings(cfg)
    powers = []
    for index, name in enumerate(UC_UNITS):
        minimum = max(float(system.s[name]['p_min']), settings['min_load_frac'] * system.s[name]['p_max'])
        power = minimum + u[:, index] * (system.s[name]['p_max'] - minimum)
        powers.append(
            _project_committed_power(
                power * on[:, index],
                on[:, index],
                minimum,
                system.s[name]['p_max'],
                system.s[name]['ramp'],
            )
        )
    eb = system.scale_box(u[:, 3], 'eb')
    bat = system.scale_signed(np.clip(action[:, 4] / margin, -1, 1), 'battery')
    gas = system.scale_signed(np.clip(action[:, 5] / margin, -1, 1), 'gastank')
    base_a0 = system.a0_from_physical(*powers, eb, bat, gas)
    decoded = system.decode_a0(base_a0, sc, apply_recourse=apply_recourse)
    return (decoded, on, base_a0)

def _commitment_from_decoded(system, dec, cfg, tol=1e-06):
    settings = uc_settings(cfg)
    commitment = {}
    for name in UC_UNITS:
        power = np.asarray(getattr(dec, name), dtype=float)
        commitment[name] = (power > tol).astype(np.int8)
    return (commitment, settings)

def uc_violation(system, dec, cfg, tol=1e-06, commitment=None):
    inferred, settings = _commitment_from_decoded(system, dec, cfg, tol)
    if commitment is None:
        commitment = inferred
    else:
        commitment = {name: np.asarray(commitment[:, i], dtype=np.int8) for i, name in enumerate(UC_UNITS)}
    out = {
        'uc_min_load': 0.0,
        'uc_off_output': 0.0,
        'uc_min_up': 0.0,
        'uc_min_down': 0.0,
    }
    for name, on in commitment.items():
        power = np.asarray(getattr(dec, name), dtype=float)
        minimum = max(float(system.s[name]['p_min']), settings['min_load_frac'] * system.s[name]['p_max'])
        out['uc_min_load'] += float(np.maximum(minimum * on - power, 0.0).sum())
        out['uc_off_output'] += float(np.where(power > tol, power, 0.0)[on == 0].sum())
        start = np.maximum(on - np.r_[0, on[:-1]], 0)
        stop = np.maximum(np.r_[0, on[:-1]] - on, 0)
        for t in range(len(on)):
            up_start = max(0, t - settings['min_up'] + 1)
            down_start = max(0, t - settings['min_down'] + 1)
            out['uc_min_up'] += float(max(start[up_start:t + 1].sum() - on[t], 0))
            out['uc_min_down'] += float(max(stop[down_start:t + 1].sum() - (1 - on[t]), 0))
    return out

def uc_total_violation(system, dec, cfg, commitment=None):
    return system.total_violation({**dec.viol, **uc_violation(system, dec, cfg, commitment=commitment)})

def uc_objectives(system, dec, cfg, commitment=None):
    inferred, settings = _commitment_from_decoded(system, dec, cfg)
    if commitment is None:
        commitment = inferred
    else:
        commitment = {name: np.asarray(commitment[:, i], dtype=np.int8) for i, name in enumerate(UC_UNITS)}
    starts = sum((np.maximum(on - np.r_[0, on[:-1]], 0).sum() for on in commitment.values()))
    return (float(dec.f1 + settings['start_cost'] * starts), float(dec.f2))

def _solve_uc_problem(
    system: IESSystem,
    sc,
    cfg: dict,
    *,
    lam: float,
    target_a0=None,
    emissions_upper_bound=None,
):
    T = int(cfg['horizon']['T'])
    problem, binaries = build_problem(
        system,
        sc,
        float(lam),
        cfg,
        T,
        target_a0=target_a0,
        emissions_upper_bound=emissions_upper_bound,
    )
    solver = pulp.PULP_CBC_CMD(msg=cfg['milp'].get('msg', False),
                              timeLimit=cfg['milp'].get('time_limit', 30),
                              threads=cfg['milp'].get('threads'))
    problem.solve(solver)
    status = pulp.LpStatus.get(problem.status, str(problem.status))
    # CBC's time-limited incumbent can have LpStatusOptimal without a proof.
    if status != 'Optimal' or problem.sol_status != pulp.LpSolutionOptimal:
        mode = (
            f'emissions_upper_bound={emissions_upper_bound:.9g}'
            if emissions_upper_bound is not None
            else f'lambda={lam:.6g}'
        )
        raise DispatchSolveError(f'UC-MILP status={status!r}, solution_status={problem.sol_status}, {mode}')
    variables = problem._ies_variables

    def extract(name):
        values = np.asarray([variable.value() for variable in variables[name]], dtype=float)
        if values.shape != (T,) or not np.all(np.isfinite(values)):
            raise DispatchSolveError(f'Invalid UC-MILP variable vector {name}')
        return values
    commitment = {name: np.rint(extract_on).astype(np.int8) for name, extract_on in ((name, np.asarray([v.value() for v in variables['on'][name]], float)) for name in UC_UNITS)}
    a0 = uc_action_from_physical(system, cfg, extract('chp'), extract('gt'), extract('gb'), extract('eb'), extract('bat_c') - extract('bat_d'), extract('gas_i') - extract('gas_w'), commitment)
    dec, projected_on, _ = decode_uc_action(system, a0, sc, cfg)
    violation = uc_total_violation(system, dec, cfg, projected_on)
    tolerance = float(cfg['milp'].get('feasibility_tol', 0.001))
    if not np.isfinite(violation) or violation > tolerance:
        raise DispatchSolveError(f'Shared decoder rejected UC-MILP trajectory: violation={violation:.6g}')
    f1, f2 = uc_objectives(system, dec, cfg, projected_on)
    solver_f1 = float(pulp.value(problem._ies_objectives['economic']))
    solver_f2 = float(pulp.value(problem._ies_objectives['emissions']))
    if not np.all(np.isfinite([solver_f1, solver_f2])):
        raise DispatchSolveError('UC-MILP returned non-finite objective values')
    return {
        'a0': a0,
        'f1': f1,
        'f2': f2,
        'solver_f1': solver_f1,
        'solver_f2': solver_f2,
        'status': status,
        'lam': float(lam),
        'violation': violation,
        'binary_variables': binaries,
    }


def solve_uc_milp(system: IESSystem, sc, lam: float, cfg: dict, *, target_a0=None):
    return _solve_uc_problem(
        system,
        sc,
        cfg,
        lam=float(lam),
        target_a0=target_a0,
    )


def solve_uc_epsilon_constraint(
    system: IESSystem,
    sc,
    emissions_upper_bound: float,
    cfg: dict,
):
    cap = float(emissions_upper_bound)
    if not np.isfinite(cap) or cap < 0.0:
        raise ValueError('emissions_upper_bound must be finite and non-negative')
    result = _solve_uc_problem(
        system,
        sc,
        cfg,
        lam=1.0,
        emissions_upper_bound=cap,
    )
    tolerance = float(cfg['milp'].get('feasibility_tol', 0.001))
    if result['solver_f2'] > cap + tolerance:
        raise DispatchSolveError(
            'UC-MILP violated the epsilon constraint: '
            f"solver_f2={result['solver_f2']:.9g}, cap={cap:.9g}"
        )
    result['emissions_upper_bound'] = cap
    result['epsilon_slack'] = cap - result['solver_f2']
    return result


def _pareto_deduplicate(points, *, rtol=1e-8, atol=1e-6):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError('points must be a finite (n, 2) array')
    if len(points) == 0:
        return points.copy()
    order = np.lexsort((points[:, 1], points[:, 0]))
    unique = []
    for point in points[order]:
        if not unique or not np.allclose(point, unique[-1], rtol=rtol, atol=atol):
            unique.append(point)
    unique = np.asarray(unique, dtype=float)
    keep = np.ones(len(unique), dtype=bool)
    for i, point in enumerate(unique):
        dominated = np.all(unique <= point, axis=1) & np.any(unique < point, axis=1)
        dominated[i] = False
        if np.any(dominated):
            keep[i] = False
    front = unique[keep]
    return front[np.argsort(front[:, 0])]


def uc_epsilon_oracle_front(system: IESSystem, sc, cfg: dict, grid_size: int = 101):
    grid_size = int(grid_size)
    if grid_size < 2:
        raise ValueError('grid_size must be at least 2')
    min_emissions = solve_uc_milp(system, sc, 0.0, cfg)
    min_cost = solve_uc_milp(system, sc, 1.0, cfg)
    lower = min_emissions['solver_f2']
    upper = min_cost['solver_f2']
    tolerance = float(cfg['milp'].get('feasibility_tol', 0.001))
    if upper < lower - tolerance:
        raise DispatchSolveError(
            f'Invalid epsilon endpoints: min-emissions={lower:.9g}, min-cost={upper:.9g}'
        )
    caps = np.linspace(lower, max(lower, upper), grid_size)
    results = []
    feasible_caps = []
    infeasible_caps = 0
    for cap in caps:
        try:
            result = solve_uc_epsilon_constraint(system, sc, cap, cfg)
        except DispatchSolveError as exc:
            if "status='Infeasible'" not in str(exc):
                raise
            infeasible_caps += 1
            continue
        results.append(result)
        feasible_caps.append(cap)
    if not results:
        raise DispatchSolveError('Epsilon-constraint grid produced no feasible points')
    caps = np.asarray(feasible_caps, dtype=float)
    decoded_points = np.asarray([(item['f1'], item['f2']) for item in results], dtype=float)
    solver_points = np.asarray(
        [(item['solver_f1'], item['solver_f2']) for item in results], dtype=float
    )
    endpoint_points = np.asarray(
        [
            (min_emissions['f1'], min_emissions['f2']),
            (min_cost['f1'], min_cost['f2']),
        ],
        dtype=float,
    )
    front = _pareto_deduplicate(np.vstack([decoded_points, endpoint_points]))
    return {
        'front': front.astype(np.float32),
        'caps': caps,
        'decoded_points': decoded_points,
        'solver_points': solver_points,
        'endpoint_points': endpoint_points,
        'max_epsilon_violation': float(
            np.maximum(solver_points[:, 1] - caps, 0.0).max(initial=0.0)
        ),
        'max_decoder_objective_delta': float(
            np.abs(decoded_points - solver_points).max(initial=0.0)
        ),
        'infeasible_caps': int(infeasible_caps),
    }


def solve_uc_lp_relaxation(system: IESSystem, sc, lam: float, cfg: dict):
    T = int(cfg['horizon']['T'])
    problem, binaries = build_problem(
        system, sc, float(lam), cfg, T, relax_binaries=True
    )
    solver = pulp.PULP_CBC_CMD(
        msg=cfg['milp'].get('msg', False),
        timeLimit=cfg['milp'].get('time_limit', 30),
    )
    problem.solve(solver)
    status = pulp.LpStatus.get(problem.status, str(problem.status))
    if status != 'Optimal':
        raise DispatchSolveError(
            f'UC-LP relaxation status={status!r}, lambda={lam:.6g}'
        )
    variables = problem._ies_variables

    def extract(name):
        values = np.asarray([variable.value() for variable in variables[name]], dtype=float)
        if values.shape != (T,) or not np.all(np.isfinite(values)):
            raise DispatchSolveError(f'Invalid UC-LP variable vector {name}')
        return values

    fractional = {
        name: np.asarray([value.value() for value in variables['on'][name]], dtype=float)
        for name in UC_UNITS
    }
    rounded = {name: (values >= 0.5).astype(np.int8) for name, values in fractional.items()}
    a0 = uc_action_from_physical(
        system,
        cfg,
        extract('chp'),
        extract('gt'),
        extract('gb'),
        extract('eb'),
        extract('bat_c') - extract('bat_d'),
        extract('gas_i') - extract('gas_w'),
        rounded,
    )
    decoded, commitment, _ = decode_uc_action(system, a0, sc, cfg)
    f1, f2 = uc_objectives(system, decoded, cfg, commitment)
    lower_f1 = float(pulp.value(problem._ies_objectives['economic']))
    lower_f2 = float(pulp.value(problem._ies_objectives['emissions']))
    return {
        'a0': a0,
        'f1': f1,
        'f2': f2,
        'lower_f1': lower_f1,
        'lower_f2': lower_f2,
        'violation': uc_total_violation(system, decoded, cfg, commitment),
        'fractional_commitment': np.stack([fractional[name] for name in UC_UNITS], axis=1),
        'relaxed_binary_variables': binaries,
        'status': status,
        'lam': float(lam),
    }

def repair_a0_uc_milp(system, sc, a0, cfg):
    _, _, base_a0 = decode_uc_action(system, a0, sc, cfg)
    return solve_uc_milp(system, sc, 0.5, cfg, target_a0=base_a0)

def build_uc_demos(system, scenarios, cfg, logger=None):
    states, lams, actions, f1s, f2s = ([], [], [], [], [])
    for i, scenario in enumerate(scenarios):
        state = scenario.state_grid(system.bat['soc_init'], system.gastank['soc_init'])
        for lam in cfg['milp']['lambda_grid']:
            result = solve_uc_milp(system, scenario, float(lam), cfg)
            states.append(state)
            lams.append(lam)
            actions.append(result['a0'])
            f1s.append(result['f1'])
            f2s.append(result['f2'])
        if logger and (i + 1) % 20 == 0:
            logger.info('  UC-MILP demos: %d/%d scenarios', i + 1, len(scenarios))
    return {'state': np.stack(states).astype(np.float32), 'lam': np.asarray(lams, np.float32), 'a0': np.stack(actions).astype(np.float32), 'f1': np.asarray(f1s, np.float32), 'f2': np.asarray(f2s, np.float32)}

def uc_oracle_front(system, scenarios, cfg, logger=None):
    grid = np.linspace(0.0, 1.0, int(cfg['milp']['oracle_grid_n']))
    fronts = []
    for i, scenario in enumerate(scenarios):
        fronts.append(np.asarray([(result['f1'], result['f2']) for result in (solve_uc_milp(system, scenario, float(lam), cfg) for lam in grid)], dtype=np.float32))
        if logger and (i + 1) % 10 == 0:
            logger.info('  UC-MILP oracle: %d/%d scenarios', i + 1, len(scenarios))
    return fronts
