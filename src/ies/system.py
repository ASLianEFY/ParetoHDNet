from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
RAW_COLS = ['chp', 'gt', 'gb', 'eb', 'bat', 'gas']
N_RAW = len(RAW_COLS)
ACTION_COLS = ['grid', 'chp', 'gt', 'gb', 'eb', 'hp', 'ch', 'bat', 'gas']
RAMP_DEVICES = ('chp', 'gt', 'gb', 'eb', 'hp', 'ch')

def _pp(x):
    return np.maximum(x, 0.0)

def _np_(x):
    return np.maximum(-x, 0.0)

@dataclass
class DecodeState:
    soc_bat: float
    soc_gas: float
    prev_powers: Optional[Dict[str, float]] = None

@dataclass
class DecodedStep:
    grid: float
    chp: float
    gt: float
    gb: float
    eb: float
    hp: float
    ch: float
    bat: float
    gas: float
    g_import: float
    soc_bat: float
    soc_gas: float
    cost: float
    carbon: float
    residuals: Dict[str, float]
    state: DecodeState
    start_soc_bat: float
    start_soc_gas: float

    @property
    def powers(self) -> Dict[str, float]:
        return {name: float(getattr(self, name)) for name in RAMP_DEVICES}

@dataclass
class Decoded:
    grid: np.ndarray
    chp: np.ndarray
    gt: np.ndarray
    gb: np.ndarray
    eb: np.ndarray
    hp: np.ndarray
    ch: np.ndarray
    bat: np.ndarray
    gas: np.ndarray
    g_import: np.ndarray
    soc_bat: np.ndarray
    soc_gas: np.ndarray
    f1: float
    f2: float
    viol: dict
    residuals: Optional[Dict[str, np.ndarray]] = None

class IESSystem:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dt = float(cfg['horizon']['dt'])
        self.T = int(cfg['horizon']['T'])
        self.s = cfg['system']
        self.bat = self.s['battery']
        self.gastank = self.s['gastank']
        self.margin = float(cfg['model'].get('a0_margin', 1.0))

    def scale_box(self, u, dev):
        d = self.s[dev]
        return d['p_min'] + u * (d['p_max'] - d['p_min'])

    def scale_signed(self, u, dev):
        return u * self.s[dev]['p_max']

    def initial_decode_state(self) -> DecodeState:
        return DecodeState(float(self.bat['soc_init']), float(self.gastank['soc_init']), None)

    def _ramp_one(self, requested: float, dev: str, previous: Optional[float]) -> float:
        lower = float(self.s[dev]['p_min'])
        upper = float(self.s[dev]['p_max'])
        value = float(np.clip(requested, lower, upper))
        if previous is not None:
            previous = float(previous)
            if not np.isfinite(previous) or not lower <= previous <= upper:
                raise ValueError(f'previous {dev} power must be finite and in [{lower}, {upper}]')
            ramp = float(self.s[dev]['ramp'])
            value = float(np.clip(value, max(lower, previous - ramp), min(upper, previous + ramp)))
        scale = max(1.0, abs(lower), abs(upper))
        if abs(value - lower) <= 1e-8 * scale:
            value = lower
        elif abs(value - upper) <= 1e-8 * scale:
            value = upper
        return value

    def _storage_one(self, requested: float, soc: float, dev: str, remaining: int):
        d = self.s[dev]
        dt = self.dt
        sd = d['self_discharge']
        pmax = d['p_max']
        smin, smax, sinit = (d['soc_min'], d['soc_max'], d['soc_init'])
        echg, edis = (d['eta_chg'], d['eta_dis'])
        p_req = float(np.clip(requested, -pmax, pmax))
        decay = (1.0 - sd) ** remaining
        charge_sum = remaining if sd == 0 else (1.0 - decay) / sd
        floor_next = max(smin, (sinit - echg * pmax * dt * charge_sum) / decay)
        floor_next = min(floor_next, smax)
        base = (1.0 - sd) * float(soc)
        p_hi = (smax - base) / (echg * dt)
        if base >= floor_next:
            p_lo = -(base - floor_next) * edis / dt
        else:
            p_lo = (floor_next - base) / (echg * dt)
        feasible_lo = max(p_lo, -pmax)
        feasible_hi = min(p_hi, pmax)
        if feasible_lo <= feasible_hi:
            power = float(np.clip(p_req, feasible_lo, feasible_hi))
        else:
            power = float(np.clip(p_req, -pmax, pmax))
            if p_hi < -pmax:
                power = -pmax
            elif p_lo > pmax:
                power = pmax
            else:
                power = float(np.clip(p_hi, -pmax, pmax))
        next_soc = base + echg * _pp(power) * dt - 1.0 / edis * _np_(power) * dt
        return (float(power), float(next_soc))

    def decode_step(self, raw_t, sc, t: int, state: DecodeState, steps_remaining: Optional[int]=None, *, apply_recourse: bool=True) -> DecodedStep:
        raw_t = np.asarray(raw_t, dtype=np.float64)
        if raw_t.shape != (N_RAW,):
            raise ValueError(f'raw_t must have shape ({N_RAW},), got {raw_t.shape}')
        if not np.all(np.isfinite(raw_t)):
            raise ValueError('raw_t must contain only finite values')
        if not 0 <= int(t) < len(sc.pv):
            raise IndexError(f'scenario step {t} is out of range')
        scenario_fields = ('pv', 'wt', 'load_e', 'load_h', 'load_c', 'load_g', 'price_buy', 'price_sell', 'price_gas', 'carbon')
        if any((not np.isfinite(float(np.asarray(getattr(sc, field))[t])) for field in scenario_fields)):
            raise ValueError(f'scenario step {t} must contain only finite values')
        if not np.isfinite(state.soc_bat) or not np.isfinite(state.soc_gas):
            raise ValueError('storage state must contain only finite SOC values')
        if steps_remaining is None:
            steps_remaining = len(sc.pv) - int(t) - 1
        steps_remaining = max(int(steps_remaining), 0)
        previous = state.prev_powers or {}
        requested = {name: float(self.scale_box(raw_t[i], name)) for i, name in enumerate(('chp', 'gt', 'gb', 'eb'))}
        realised = {name: self._ramp_one(requested[name], name, previous.get(name)) for name in ('chp', 'gt', 'gb', 'eb')}
        bat, soc_bat = self._storage_one(self.scale_signed(raw_t[4], 'battery'), state.soc_bat, 'battery', steps_remaining)
        s = self.s
        chp_e = s['chp']['eta_e'] * realised['chp']
        chp_h = s['chp']['eta_h'] * realised['chp']
        gt_e = s['gt']['eta_e'] * realised['gt']
        fixed_heat = chp_h + s['gb']['eta_h'] * realised['gb'] + s['eb']['eta_h'] * realised['eb']
        hp_requested = (float(sc.load_h[t]) - fixed_heat) / s['hp']['cop']
        hp = self._ramp_one(hp_requested, 'hp', previous.get('hp'))
        ch_requested = float(sc.load_c[t]) / s['ch']['cop']
        ch = self._ramp_one(ch_requested, 'ch', previous.get('ch'))
        cool_residual = s['ch']['cop'] * ch - float(sc.load_c[t])
        heat_short = _pp(float(sc.load_h[t]) - fixed_heat - s['hp']['cop'] * hp)
        if apply_recourse and heat_short > 0.0:
            eb_upper = float(s['eb']['p_max'])
            if 'eb' in previous:
                eb_upper = min(eb_upper, float(previous['eb']) + s['eb']['ramp'])
            grid_without_eb = float(sc.load_e[t]) + hp + ch + bat - float(sc.pv[t]) - float(sc.wt[t]) - chp_e - gt_e
            grid_eb_headroom = max(0.0, s['grid']['p_max_import'] - grid_without_eb - realised['eb'])
            eb_add = min(heat_short / s['eb']['eta_h'], max(0.0, eb_upper - realised['eb']), grid_eb_headroom)
            realised['eb'] += eb_add
            fixed_heat += s['eb']['eta_h'] * eb_add
            heat_short = _pp(float(sc.load_h[t]) - fixed_heat - s['hp']['cop'] * hp)
        gb_is_available = 'uc' not in self.cfg or realised['gb'] > 1e-09
        if apply_recourse and heat_short > 0.0 and gb_is_available:
            gb_upper = float(s['gb']['p_max'])
            if 'gb' in previous:
                gb_upper = min(gb_upper, float(previous['gb']) + s['gb']['ramp'])
            gb_add = min(heat_short / s['gb']['eta_h'], max(0.0, gb_upper - realised['gb']))
            realised['gb'] += gb_add
            fixed_heat += s['gb']['eta_h'] * gb_add
        heat_residual = fixed_heat + s['hp']['cop'] * hp - float(sc.load_h[t])
        gas_demand = realised['chp'] + realised['gt'] + realised['gb'] + float(sc.load_g[t])
        gas_requested = self.scale_signed(raw_t[5], 'gastank')
        if apply_recourse:
            gas_requested = max(gas_requested, -gas_demand)
        gas, soc_gas = self._storage_one(gas_requested, state.soc_gas, 'gastank', steps_remaining)
        grid_raw = float(sc.load_e[t]) + realised['eb'] + hp + ch + bat - float(sc.pv[t]) - float(sc.wt[t]) - chp_e - gt_e
        grid = float(np.clip(grid_raw, -s['grid']['p_max_export'], s['grid']['p_max_import']))
        elec_imbalance = abs(grid_raw - grid)
        gas_raw = realised['chp'] + realised['gt'] + realised['gb'] + float(sc.load_g[t]) + gas
        g_import = float(_pp(gas_raw))
        dt = self.dt
        cost = float(sc.price_buy[t]) * _pp(grid) * dt - float(sc.price_sell[t]) * _np_(grid) * dt + float(sc.price_gas[t]) * g_import * dt + self.bat['degr_cost'] * abs(bat) * dt
        carbon = float(sc.carbon[t]) * _pp(grid) * dt + self.cfg['data']['carbon_gas'] * g_import * dt
        realised.update(hp=hp, ch=ch)
        next_state = DecodeState(soc_bat, soc_gas, realised.copy())
        residuals = {'heat_short': float(_np_(heat_residual)), 'heat_surplus': float(_pp(heat_residual)), 'cool_short': float(_np_(cool_residual)), 'cool_surplus': float(_pp(cool_residual)), 'elec_imbalance': float(elec_imbalance), 'gas_short': float(_np_(gas_raw))}
        return DecodedStep(grid=grid, chp=realised['chp'], gt=realised['gt'], gb=realised['gb'], eb=realised['eb'], hp=hp, ch=ch, bat=bat, gas=gas, g_import=g_import, soc_bat=soc_bat, soc_gas=soc_gas, cost=float(cost), carbon=float(carbon), residuals=residuals, state=next_state, start_soc_bat=float(state.soc_bat), start_soc_gas=float(state.soc_gas))

    def decoded_from_steps(self, steps) -> Decoded:
        steps = list(steps)
        if not steps:
            raise ValueError('at least one realised step is required')
        names = ('grid', 'chp', 'gt', 'gb', 'eb', 'hp', 'ch', 'bat', 'gas', 'g_import')
        values = {name: np.asarray([getattr(step, name) for step in steps], dtype=np.float64) for name in names}
        soc_bat = np.asarray([steps[0].start_soc_bat] + [step.soc_bat for step in steps], dtype=np.float64)
        soc_gas = np.asarray([steps[0].start_soc_gas] + [step.soc_gas for step in steps], dtype=np.float64)
        residual_series = {name: np.asarray([step.residuals[name] for step in steps], dtype=np.float64) for name in ('heat_short', 'heat_surplus', 'cool_short', 'cool_surplus', 'elec_imbalance', 'gas_short')}
        viol = self._violations({name: values[name] for name in RAMP_DEVICES}, soc_bat, soc_gas, residual_series)
        return Decoded(values['grid'], values['chp'], values['gt'], values['gb'], values['eb'], values['hp'], values['ch'], values['bat'], values['gas'], values['g_import'], soc_bat, soc_gas, float(sum((step.cost for step in steps))), float(sum((step.carbon for step in steps))), viol, residual_series)

    def decode(self, raw: np.ndarray, sc, *, apply_recourse: bool=True) -> Decoded:
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != N_RAW:
            raise ValueError(f'raw must have shape (T, {N_RAW}), got {raw.shape}')
        if raw.shape[0] != len(sc.pv):
            raise ValueError('raw trajectory and scenario must have equal horizons')
        scenario_fields = ('pv', 'wt', 'load_e', 'load_h', 'load_c', 'load_g', 'price_buy', 'price_sell', 'price_gas', 'carbon')
        if not np.all(np.isfinite(raw)):
            raise ValueError('raw trajectory must contain only finite values')
        for field in scenario_fields:
            values_field = np.asarray(getattr(sc, field))
            if values_field.shape != (len(raw),) or not np.all(np.isfinite(values_field)):
                raise ValueError(f'scenario.{field} must be a finite length-{len(raw)} array')
        state = self.initial_decode_state()
        steps = []
        for t in range(len(raw)):
            step = self.decode_step(raw[t], sc, t, state, len(raw) - t - 1, apply_recourse=apply_recourse)
            steps.append(step)
            state = step.state
        return self.decoded_from_steps(steps)
    # Match the equality heat balance in uc_problem.build_problem. Historical
    # reports omitted heat_surplus and must not be mixed with this protocol.
    FEASIBILITY_PROTOCOL = 'strict_carrier_balance_v2'
    _FEASIBILITY_RESIDUALS = ('heat_short', 'heat_surplus', 'cool_short', 'cool_surplus', 'elec_imbalance', 'gas_short')

    def _violations(self, devs, soc_bat, soc_gas, residuals):
        ramp = 0.0
        for name, power in devs.items():
            ramp += float(np.sum(_pp(np.abs(np.diff(power)) - self.s[name]['ramp'])))
        soc_box = 0.0
        for soc, dev in ((soc_bat, 'battery'), (soc_gas, 'gastank')):
            d = self.s[dev]
            soc_box += float(np.sum(_pp(soc - d['soc_max']) + _pp(d['soc_min'] - soc)))
        terminal = _pp(self.bat['soc_init'] - soc_bat[-1]) + _pp(self.gastank['soc_init'] - soc_gas[-1])
        out = {'ramp': ramp, 'soc_box': soc_box, 'soc_terminal': float(terminal)}
        for name in self._FEASIBILITY_RESIDUALS:
            if name in residuals:
                out[name] = float(np.sum(residuals[name]))
        return out

    @staticmethod
    def total_violation(viol: dict) -> float:
        try:
            values = np.asarray(list(viol.values()), dtype=np.float64)
        except (TypeError, ValueError):
            return float('inf')
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            return float('inf')
        total = float(np.sum(values))
        return total if np.isfinite(total) and total >= 0.0 else float('inf')

    def is_feasible(self, viol: dict, tol: float=0.001) -> bool:
        total = self.total_violation(viol)
        return bool(np.isfinite(total) and total <= tol)

    def a0_from_physical(self, chp, gt, gb, eb, bat, gas) -> np.ndarray:
        g = self.margin

        def box_u(power, dev):
            d = self.s[dev]
            return g * (2.0 * (power - d['p_min']) / (d['p_max'] - d['p_min']) - 1.0)

        def signed_u(power, dev):
            return g * (power / self.s[dev]['p_max'])
        a0 = np.stack([box_u(chp, 'chp'), box_u(gt, 'gt'), box_u(gb, 'gb'), box_u(eb, 'eb'), signed_u(bat, 'battery'), signed_u(gas, 'gastank')], axis=1)
        return a0.astype(np.float64)

    def raw_from_a0(self, a0: np.ndarray) -> np.ndarray:
        a0 = np.asarray(a0)
        if a0.shape[-1] != N_RAW:
            raise ValueError(f'a0 last dimension must be {N_RAW}, got {a0.shape}')
        raw = np.empty_like(a0, dtype=np.float64)
        raw[..., 0:4] = np.clip((a0[..., 0:4] / self.margin + 1.0) / 2.0, 0.0, 1.0)
        raw[..., 4:6] = np.clip(a0[..., 4:6] / self.margin, -1.0, 1.0)
        return raw

    def decode_a0(self, a0: np.ndarray, sc, *, apply_recourse: bool=True) -> Decoded:
        return self.decode(self.raw_from_a0(a0), sc, apply_recourse=apply_recourse)

    def decode_t(self, raw_t, sc, t, soc_bat, soc_gas, prev_powers=None, steps_remaining=None, return_step=False):
        raw = self.raw_from_a0(np.asarray(raw_t, dtype=np.float64))
        state = DecodeState(float(soc_bat), float(soc_gas), prev_powers)
        step = self.decode_step(raw, sc, t, state, steps_remaining)
        if return_step:
            return step
        devices = step.powers
        devices.update(bat=step.bat, gas=step.gas)
        return (step.cost, step.carbon, step.soc_bat, step.soc_gas, devices)
