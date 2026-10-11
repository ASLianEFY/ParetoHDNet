from __future__ import annotations
import copy
import logging
import math
import random
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import yaml
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
CONFIG_PATH = SRC_DIR / 'configs' / 'default.yaml'
RESULTS_DIR = ROOT_DIR / 'results'
CACHE_DIR = ROOT_DIR / 'data_cache'


def default_oracle_path(data_suffix: str = '_uc_milp', *, legacy: bool = False) -> Path:
    """Return the explicit default oracle for the requested evaluation protocol.

    Full UC evaluations use the dense 101-level epsilon reference.  The legacy
    weighted-sum artifact remains available only when explicitly requested.
    """
    if legacy:
        return CACHE_DIR / f'oracle_test{data_suffix}.pkl'
    return CACHE_DIR / f'oracle_test{data_suffix}_epsilon_k101.pkl'

class ConfigError(ValueError):
    pass

def ensure_output_dirs(*directories: str | Path) -> tuple[Path, ...]:
    paths = tuple((Path(p) for p in directories)) if directories else (CACHE_DIR, RESULTS_DIR)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths

def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out

def _check_override_keys(override: Mapping[str, Any], base: Mapping[str, Any], path: str) -> None:
    for key, value in override.items():
        here = f'{path}.{key}' if path else str(key)
        if key not in base:
            raise ConfigError(f"Unknown configuration key '{here}'. Check the spelling or add the key to the base configuration before overriding it.")
        if isinstance(value, Mapping):
            if not isinstance(base[key], Mapping):
                raise ConfigError(f"Configuration key '{here}' cannot be a mapping.")
            _check_override_keys(value, base[key], here)

def _get(cfg: Mapping[str, Any], path: str) -> Any:
    value: Any = cfg
    for part in path.split('.'):
        if not isinstance(value, Mapping) or part not in value:
            raise ConfigError(f"Missing required configuration key '{path}'.")
        value = value[part]
    return value

def _number(cfg: Mapping[str, Any], path: str, *, integer: bool=False) -> float | int:
    value = _get(cfg, path)
    expected = isinstance(value, int) and (not isinstance(value, bool)) if integer else isinstance(value, (int, float)) and (not isinstance(value, bool))
    if not expected or not math.isfinite(float(value)):
        kind = 'integer' if integer else 'finite number'
        raise ConfigError(f"Configuration key '{path}' must be a {kind}; got {value!r}.")
    return value

def _positive(cfg: Mapping[str, Any], path: str, *, integer: bool=False, allow_zero: bool=False):
    value = _number(cfg, path, integer=integer)
    invalid = value < 0 if allow_zero else value <= 0
    if invalid:
        op = 'non-negative' if allow_zero else 'greater than zero'
        raise ConfigError(f"Configuration key '{path}' must be {op}; got {value!r}.")
    return value

def _probability(cfg: Mapping[str, Any], path: str, *, inclusive_one: bool=True) -> float:
    value = float(_number(cfg, path))
    upper_ok = value <= 1.0 if inclusive_one else value < 1.0
    if value < 0.0 or not upper_ok:
        interval = '[0, 1]' if inclusive_one else '[0, 1)'
        raise ConfigError(f"Configuration key '{path}' must be in {interval}; got {value!r}.")
    return value

def validate_config(cfg: Mapping[str, Any]) -> None:
    if not isinstance(cfg, Mapping):
        raise ConfigError(f'Configuration root must be a mapping; got {type(cfg).__name__}.')
    seed = _number(cfg, 'seed', integer=True)
    if not 0 <= seed < 2 ** 32:
        raise ConfigError(f"Configuration key 'seed' must be in [0, 2**32); got {seed!r}.")
    T = _positive(cfg, 'horizon.T', integer=True)
    L = _positive(cfg, 'horizon.L', integer=True)
    _positive(cfg, 'horizon.dt')
    if L != T:
        raise ConfigError(f"Configuration keys 'horizon.L' and 'horizon.T' must match: scenario state grids currently condition on the complete dispatch horizon (got L={L}, T={T}).")
    n_scenarios = _positive(cfg, 'data.n_scenarios', integer=True)
    split = _get(cfg, 'data.split')
    if not isinstance(split, (list, tuple)) or len(split) != 3:
        raise ConfigError("Configuration key 'data.split' must contain [train, val, test].")
    if any((isinstance(x, bool) or not isinstance(x, (int, float)) or (not math.isfinite(float(x))) or (x <= 0) for x in split)):
        raise ConfigError("Every value in 'data.split' must be a finite number greater than zero.")
    absolute = all((float(x).is_integer() for x in split)) and int(sum(split)) == int(n_scenarios)
    if absolute:
        counts = tuple((int(x) for x in split))
        if any((c < 1 for c in counts)):
            raise ConfigError(f"Absolute 'data.split' counts must each be at least 1 (got {list(counts)}).")
    else:
        if not math.isclose(float(sum(split)), 1.0, rel_tol=0.0, abs_tol=1e-09):
            raise ConfigError(f"Configuration key 'data.split' must either be absolute counts summing to 'data.n_scenarios' ({n_scenarios}) or fractions summing to 1.0; got {list(split)!r}.")
        counts = (int(n_scenarios * split[0]), int(n_scenarios * split[1]))
        if counts[0] < 1 or counts[1] < 1 or n_scenarios - sum(counts) < 1:
            raise ConfigError("Configuration keys 'data.n_scenarios' and 'data.split' must produce at least one train, validation, and test scenario.")
    for key in ('pv_capacity', 'wt_capacity', 'base_load_e', 'base_load_h', 'base_load_c', 'base_load_g', 'price_buy_peak', 'price_buy_off', 'price_gas_off', 'price_gas_peak', 'carbon_grid_base', 'carbon_gas'):
        _positive(cfg, f'data.{key}')
    for key in ('load_noise', 'wind_noise', 'pv_noise', 'carbon_grid_swing'):
        _positive(cfg, f'data.{key}', allow_zero=True)
    _probability(cfg, 'data.wind_ar')
    _probability(cfg, 'data.price_sell_ratio')
    for key in ('p_max_import', 'p_max_export', 'ramp'):
        _positive(cfg, f'system.grid.{key}')
    for device in ('chp', 'gt', 'gb', 'eb', 'hp', 'ch'):
        p_min = _positive(cfg, f'system.{device}.p_min', allow_zero=True)
        p_max = _positive(cfg, f'system.{device}.p_max')
        _positive(cfg, f'system.{device}.ramp')
        if p_min >= p_max:
            raise ConfigError(f"Configuration keys 'system.{device}.p_min/p_max' must satisfy 0 <= p_min < p_max; got {p_min!r}/{p_max!r}.")
    for device, efficiency in (('chp', 'eta_e'), ('chp', 'eta_h'), ('gt', 'eta_e'), ('gb', 'eta_h'), ('eb', 'eta_h')):
        eta = _positive(cfg, f'system.{device}.{efficiency}')
        if eta > 1:
            raise ConfigError(f"Configuration key 'system.{device}.{efficiency}' cannot exceed 1.")
    _positive(cfg, 'system.hp.cop')
    _positive(cfg, 'system.ch.cop')
    for device in ('battery', 'gastank'):
        soc_min = _positive(cfg, f'system.{device}.soc_min', allow_zero=True)
        soc_max = _positive(cfg, f'system.{device}.soc_max')
        soc_init = _positive(cfg, f'system.{device}.soc_init', allow_zero=True)
        _positive(cfg, f'system.{device}.p_max')
        eta_chg = _positive(cfg, f'system.{device}.eta_chg')
        eta_dis = _positive(cfg, f'system.{device}.eta_dis')
        if eta_chg > 1 or eta_dis > 1:
            raise ConfigError(f"Storage efficiencies for 'system.{device}' cannot exceed 1.")
        _probability(cfg, f'system.{device}.self_discharge', inclusive_one=False)
        if not soc_min <= soc_init <= soc_max:
            raise ConfigError(f"Configuration keys 'system.{device}.soc_min/soc_init/soc_max' must satisfy soc_min <= soc_init <= soc_max; got {soc_min}/{soc_init}/{soc_max}.")
    _positive(cfg, 'system.battery.degr_cost', allow_zero=True)
    if 'uc' in cfg:
        uc = _get(cfg, 'uc')
        if not isinstance(uc, Mapping):
            raise ConfigError("Configuration key 'uc' must be a mapping.")
        _positive(cfg, 'uc.min_load_frac')
        _positive(cfg, 'uc.start_cost', allow_zero=True)
        _positive(cfg, 'uc.min_up', integer=True)
        _positive(cfg, 'uc.min_down', integer=True)
        action_dim = _positive(cfg, 'uc.action_dim', integer=True)
        if action_dim != 9:
            raise ConfigError("Configuration key 'uc.action_dim' must be 9.")
    d_model = _positive(cfg, 'model.d_model', integer=True)
    n_heads = _positive(cfg, 'model.n_heads', integer=True)
    _positive(cfg, 'model.n_enc_blocks', integer=True)
    _positive(cfg, 'model.n_pref_tokens', integer=True)
    if d_model % 2 or d_model % n_heads:
        raise ConfigError(f"Configuration key 'model.d_model' must be even and divisible by 'model.n_heads' (got {d_model} and {n_heads}).")
    n_channels = _positive(cfg, 'model.n_channels', integer=True)
    if n_channels != 12:
        raise ConfigError(f"Configuration key 'model.n_channels' must be 12 to match the fixed scenario channel grid; got {n_channels}.")
    base = _positive(cfg, 'model.unet_base', integer=True)
    stages = _positive(cfg, 'model.unet_stages', integer=True)
    if base % 8:
        raise ConfigError("Configuration key 'model.unet_base' must be divisible by 8 (GroupNorm).")
    if T < 2 ** (stages - 1):
        raise ConfigError(f"Configuration key 'model.unet_stages={stages}' is too large for horizon.T={T}.")
    k_train = _positive(cfg, 'model.K_train', integer=True)
    k_infer = _positive(cfg, 'model.K_infer', integer=True)
    if k_infer > k_train:
        raise ConfigError("Configuration key 'model.K_infer' cannot exceed 'model.K_train'.")
    if _get(cfg, 'model.beta_schedule') != 'cosine':
        raise ConfigError("Configuration key 'model.beta_schedule' must currently be 'cosine'.")
    _positive(cfg, 'model.cfg_w', allow_zero=True)
    _probability(cfg, 'model.p_uncond')
    _positive(cfg, 'model.phys_beta', allow_zero=True)
    margin = float(_positive(cfg, 'model.a0_margin'))
    if margin > 1:
        raise ConfigError("Configuration key 'model.a0_margin' must be in (0, 1].")
    full = _get(cfg, 'full_diffusion')
    if not isinstance(full, Mapping):
        raise ConfigError("Configuration key 'full_diffusion' must be a mapping.")
    if not isinstance(full.get('uc_hybrid', False), bool):
        raise ConfigError("Configuration key 'full_diffusion.uc_hybrid' must be boolean.")
    for key in ('commitment_head', 'commitment_projection'):
        if key in full and not isinstance(full[key], bool):
            raise ConfigError(f"Configuration key 'full_diffusion.{key}' must be boolean.")
    if 'anchor_context' in full and (not isinstance(full['anchor_context'], bool)):
        raise ConfigError("Configuration key 'full_diffusion.anchor_context' must be boolean.")
    if 'architecture_version' in full:
        _positive(cfg, 'full_diffusion.architecture_version', integer=True)
    if 'p_uncond' in full:
        _probability(cfg, 'full_diffusion.p_uncond')
    for key in ('x0_weight', 'commitment_weight', 'commitment_sequence_weight', 'commitment_event_weight', 'anchor_weight', 'objective_weight', 'preference_objective_weight', 'pre_recourse_weight', 'projection_weight', 'min_snr_gamma'):
        if key in full:
            _positive(cfg, f'full_diffusion.{key}', allow_zero=True)
    if 'commitment_teacher_forcing' in full:
        _probability(cfg, 'full_diffusion.commitment_teacher_forcing')
    if 'preference_jitter' in full:
        jitter = float(_positive(cfg, 'full_diffusion.preference_jitter', allow_zero=True))
        if jitter > 0.5:
            raise ConfigError("Configuration key 'full_diffusion.preference_jitter' must be in [0, 0.5].")
    for key in ('anchor_pretrain_epochs', 'joint_finetune_epoch'):
        if key in full:
            _positive(cfg, f'full_diffusion.{key}', integer=True, allow_zero=True)
    if 'objective_scales' in full:
        scales = full['objective_scales']
        if not isinstance(scales, (list, tuple)) or len(scales) != 2:
            raise ConfigError("Configuration key 'full_diffusion.objective_scales' must have length 2.")
        if any((isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(float(value))) or (value <= 0) for value in scales)):
            raise ConfigError('Every full_diffusion.objective_scales value must be finite and positive.')
    _positive(cfg, 'train.batch_size', integer=True)
    _positive(cfg, 'train.epochs', integer=True)
    _positive(cfg, 'train.lr')
    _positive(cfg, 'train.weight_decay', allow_zero=True)
    ema_decay = _probability(cfg, 'train.ema_decay', inclusive_one=False)
    if ema_decay == 0:
        raise ConfigError("Configuration key 'train.ema_decay' must be in (0, 1).")
    _positive(cfg, 'train.ema_update_after_step', integer=True, allow_zero=True)
    _positive(cfg, 'train.ema_update_every', integer=True)
    _positive(cfg, 'train.warmup_steps', integer=True, allow_zero=True)
    schedule = _get(cfg, 'train.lr_schedule')
    if schedule not in ('constant', 'cosine'):
        raise ConfigError("Configuration key 'train.lr_schedule' must be 'constant' or 'cosine'.")
    min_lr = _positive(cfg, 'train.min_lr', allow_zero=True)
    if min_lr > _get(cfg, 'train.lr'):
        raise ConfigError("Configuration key 'train.min_lr' must not exceed 'train.lr'.")
    for key in ('pareto_validation_every', 'pareto_validation_scenarios', 'pareto_validation_samples', 'pareto_validation_steps'):
        _positive(cfg, f'train.{key}', integer=True)
    _positive(cfg, 'train.grad_clip')
    _positive(cfg, 'train.num_workers', integer=True, allow_zero=True)
    validation_seed = _number(cfg, 'train.validation_seed', integer=True)
    if not 0 <= validation_seed < 2 ** 32:
        raise ConfigError(f"Configuration key 'train.validation_seed' must be in [0, 2**32); got {validation_seed!r}.")
    lambda_grid = _get(cfg, 'milp.lambda_grid')
    if not isinstance(lambda_grid, (list, tuple)) or not lambda_grid:
        raise ConfigError("Configuration key 'milp.lambda_grid' must be a non-empty list.")
    if any((isinstance(x, bool) or not isinstance(x, (int, float)) or (not 0 <= x <= 1) for x in lambda_grid)):
        raise ConfigError("Every value in 'milp.lambda_grid' must be in [0, 1].")
    _positive(cfg, 'milp.oracle_grid_n', integer=True)
    _positive(cfg, 'milp.time_limit')
    _positive(cfg, 'milp.feasibility_tol', allow_zero=True)
    if not isinstance(_get(cfg, 'milp.msg'), bool):
        raise ConfigError("Configuration key 'milp.msg' must be boolean.")
    for path in ('eval.K_lambda', 'eval.M_samples'):
        _positive(cfg, path, integer=True)

def load_config(path: str | Path=CONFIG_PATH, smoke: bool=False) -> dict:
    path = Path(path)
    try:
        with path.open('r', encoding='utf-8') as stream:
            loaded = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not load configuration '{path}': {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"Configuration '{path}' must contain a top-level mapping.")
    smoke_block = loaded.pop('smoke', {})
    if not isinstance(smoke_block, Mapping):
        raise ConfigError("Configuration key 'smoke' must be a mapping of overrides.")
    _check_override_keys(smoke_block, loaded, 'smoke')
    validate_config(loaded)
    smoke_cfg = _deep_merge(loaded, smoke_block)
    validate_config(smoke_cfg)
    cfg = smoke_cfg if smoke else copy.deepcopy(loaded)
    cfg['_smoke'] = bool(smoke)
    return cfg

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def get_device():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_logger(name: str='ies') -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(message)s', '%H:%M:%S'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
