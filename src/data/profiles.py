from __future__ import annotations
from dataclasses import dataclass
import numpy as np
CHANNELS = ['pv', 'wt', 'load_e', 'load_h', 'load_c', 'load_g', 'soc_bat', 'soc_gas', 'price_buy', 'price_sell', 'price_gas', 'carbon']

@dataclass
class Scenario:
    pv: np.ndarray
    wt: np.ndarray
    load_e: np.ndarray
    load_h: np.ndarray
    load_c: np.ndarray
    load_g: np.ndarray
    price_buy: np.ndarray
    price_sell: np.ndarray
    price_gas: np.ndarray
    carbon: np.ndarray
    season: float

    def state_grid(self, soc_bat0: float, soc_gas0: float) -> np.ndarray:
        T = len(self.pv)
        grid = np.stack([self.pv, self.wt, self.load_e, self.load_h, self.load_c, self.load_g, np.full(T, soc_bat0), np.full(T, soc_gas0), self.price_buy, self.price_sell, self.price_gas, self.carbon], axis=1)
        return grid.astype(np.float32)

def _diurnal(t, peak_hour, width):
    d = np.minimum(np.abs(t - peak_hour), 24 - np.abs(t - peak_hour))
    return np.exp(-0.5 * (d / width) ** 2)

def generate_scenario(rng: np.random.Generator, cfg: dict) -> Scenario:
    dc = cfg['data']
    T = cfg['horizon']['T']
    t = np.arange(T)
    season = rng.uniform(0.0, 1.0)
    pv_shape = _diurnal(t, peak_hour=13.0, width=3.0)
    pv = dc['pv_capacity'] * (0.5 + 0.5 * season) * pv_shape
    pv *= np.clip(1.0 + dc['pv_noise'] * rng.standard_normal(T), 0.0, None)
    pv = np.clip(pv, 0.0, dc['pv_capacity'])
    w = np.zeros(T)
    w[0] = rng.uniform(0.3, 0.7)
    for k in range(1, T):
        w[k] = dc['wind_ar'] * w[k - 1] + (1 - dc['wind_ar']) * rng.uniform(0.2, 0.8)
        w[k] += dc['wind_noise'] * rng.standard_normal()
    w = np.clip(w, 0.0, 1.0)
    wt = dc['wt_capacity'] * (0.7 + 0.3 * (1 - season)) * w

    def noisy(base, shape):
        prof = base * shape
        prof = prof * (1.0 + dc['load_noise'] * rng.standard_normal(T))
        return np.clip(prof, 0.0, None)
    e_shape = 0.6 + 0.4 * (_diurnal(t, 9, 3) + _diurnal(t, 19, 3))
    load_e = noisy(dc['base_load_e'], e_shape)
    h_shape = (0.5 + 0.5 * (1 - season)) * (0.6 + 0.5 * (_diurnal(t, 7, 2) + _diurnal(t, 20, 3)))
    load_h = noisy(dc['base_load_h'], h_shape)
    c_shape = (0.2 + 0.8 * season) * (0.4 + 0.7 * _diurnal(t, 14, 4))
    load_c = noisy(dc['base_load_c'], c_shape)
    g_shape = 0.7 + 0.3 * _diurnal(t, 18, 4)
    load_g = noisy(dc['base_load_g'], g_shape)
    peak_mask = (t >= 8) & (t <= 11) | (t >= 18) & (t <= 21)
    price_buy = np.where(peak_mask, dc['price_buy_peak'], dc['price_buy_off']).astype(float)
    price_sell = dc['price_sell_ratio'] * price_buy
    if 'price_gas_peak' in dc and 'price_gas_off' in dc:
        gas_peak = (t >= 6) & (t <= 9) | (t >= 17) & (t <= 22)
        price_gas = np.where(gas_peak, dc['price_gas_peak'], dc['price_gas_off']).astype(float)
    else:
        price_gas = np.full(T, dc['price_gas'], dtype=float)
    carbon = dc['carbon_grid_base'] - dc['carbon_grid_swing'] * _diurnal(t, 13, 4)
    carbon += 0.03 * rng.standard_normal(T)
    carbon = np.clip(carbon, 0.05, None)
    return Scenario(pv, wt, load_e, load_h, load_c, load_g, price_buy, price_sell, price_gas, carbon, season)

def generate_dataset(cfg: dict):
    rng = np.random.default_rng(cfg['seed'])
    n = int(cfg['data']['n_scenarios'])
    scenarios = [generate_scenario(rng, cfg) for _ in range(n)]
    split = cfg['data']['split']
    absolute = all((float(x).is_integer() for x in split)) and int(sum(split)) == n
    if absolute:
        n_tr, n_va, n_te = (int(x) for x in split)
        if n_tr + n_va + n_te != n:
            raise ValueError('absolute data.split must sum to data.n_scenarios')
    else:
        f_tr, f_va, _ = split
        n_tr, n_va = (int(n * f_tr), int(n * f_va))
        n_te = n - n_tr - n_va
    return {'train': scenarios[:n_tr], 'val': scenarios[n_tr:n_tr + n_va], 'test': scenarios[n_tr + n_va:n_tr + n_va + n_te]}
