from __future__ import annotations
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import numpy as np
SIDECAR_SUFFIX = '.metadata.json'
ARTIFACT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 2
DATA_FINGERPRINT_KEYS = ('seed', 'horizon', 'data', 'system', 'milp', 'uc')
DATA_ARTIFACT_TYPES = frozenset({'scenario_splits', 'demonstrations', 'oracle_fronts'})

def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0])) if not str(k).startswith('_')}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError('Configuration fingerprints require finite floating-point values.')
        return 0.0 if value == 0 else value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f'Unsupported fingerprint value {value!r} ({type(value).__name__}).')

def config_fingerprint(cfg: Mapping[str, Any]) -> str:
    payload = json.dumps(_canonical(cfg), sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def data_config_fingerprint(cfg: Mapping[str, Any]) -> str:
    payload = {key: cfg[key] for key in DATA_FINGERPRINT_KEYS if key in cfg}
    return config_fingerprint(payload)

def _git_provenance(root: Path) -> dict[str, Any]:

    def run(*args: str) -> str | None:
        try:
            return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True, timeout=2).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    commit = run('rev-parse', 'HEAD')
    if commit is None:
        return {'available': False}
    status = run('status', '--porcelain')
    if status is None:
        return {'available': True, 'commit': commit, 'dirty': None}
    return {'available': True, 'commit': commit, 'dirty': bool(status)}

def build_provenance(cfg: Mapping[str, Any], *, command: str | None=None, root: str | Path | None=None) -> dict[str, Any]:
    project_root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    return {'schema_version': ARTIFACT_SCHEMA_VERSION, 'created_at': datetime.now(timezone.utc).isoformat(), 'config_fingerprint': config_fingerprint(cfg), 'command': command if command is not None else ' '.join(sys.argv), 'python': platform.python_version(), 'platform': platform.platform(), 'git': _git_provenance(project_root)}

def sidecar_path(artifact_path: str | Path) -> Path:
    path = Path(artifact_path)
    return path.with_name(path.name + SIDECAR_SUFFIX)

def write_sidecar(artifact_path: str | Path, metadata: Mapping[str, Any]) -> Path:
    path = sidecar_path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(_canonical(metadata), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    temp.replace(path)
    return path

def read_sidecar(artifact_path: str | Path, *, required: bool=False) -> dict[str, Any] | None:
    path = sidecar_path(artifact_path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f'Artifact sidecar not found: {path}')
        return None
    with path.open('r', encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'Artifact sidecar must contain a JSON object: {path}')
    return value

def artifact_metadata(cfg: Mapping[str, Any], artifact_type: str, **fields: Any) -> dict[str, Any]:
    metadata = {'schema_version': ARTIFACT_SCHEMA_VERSION, 'artifact_type': artifact_type, 'config_fingerprint': config_fingerprint(cfg), 'data_config_fingerprint': data_config_fingerprint(cfg), 'provenance': build_provenance(cfg), **fields}
    return metadata

def require_compatible_sidecar(artifact_path: str | Path, cfg: Mapping[str, Any], *, artifact_type: str | None=None, fields: Mapping[str, Any] | None=None) -> dict[str, Any]:
    path = Path(artifact_path)
    try:
        metadata = read_sidecar(path, required=True)
    except FileNotFoundError as exc:
        mode = ' --smoke' if bool(cfg.get('_smoke')) else ''
        raise ValueError(f'Artifact metadata is missing for {path}. Regenerate it with `python -m src.scripts.gen_data{mode}`.') from exc
    assert metadata is not None
    effective_type = artifact_type if artifact_type is not None else metadata.get('artifact_type')
    if effective_type in DATA_ARTIFACT_TYPES:
        expected = data_config_fingerprint(cfg)
        actual = metadata.get('data_config_fingerprint')
        if actual is None:
            if metadata.get('config_fingerprint') != config_fingerprint(cfg):
                raise ValueError(f'Artifact {path} is missing data_config_fingerprint and its legacy config_fingerprint no longer matches after a non-data config change. Re-stamp data sidecars or regenerate the data.')
        elif actual != expected:
            raise ValueError(f'Artifact {path} was generated from an incompatible data configuration. Regenerate the data and dependent checkpoints before continuing.')
    else:
        expected_fingerprint = config_fingerprint(cfg)
        if metadata.get('config_fingerprint') != expected_fingerprint:
            raise ValueError(f'Artifact {path} was generated from an incompatible configuration. Regenerate the data and dependent checkpoints before continuing.')
    if artifact_type is not None and metadata.get('artifact_type') != artifact_type:
        raise ValueError(f"Artifact {path} has type {metadata.get('artifact_type')!r}; expected {artifact_type!r}.")
    for key, expected in (fields or {}).items():
        if metadata.get(key) != expected:
            raise ValueError(f'Artifact {path} metadata field {key!r} is {metadata.get(key)!r}; expected {expected!r}.')
    return metadata


def require_oracle_sidecar(
    artifact_path: str | Path,
    cfg: Mapping[str, Any],
    *,
    expected_method: str | None = None,
    expected_grid_size: int | None = None,
    expected_scenarios: int | None = None,
) -> dict[str, Any]:
    """Validate an oracle sidecar and its protocol-specific metadata."""
    metadata = require_compatible_sidecar(
        artifact_path, cfg, artifact_type='oracle_fronts'
    )
    if expected_method is not None and metadata.get('method') != expected_method:
        raise ValueError(
            f'Oracle {artifact_path} uses method {metadata.get("method")!r}; '
            f'expected {expected_method!r}.'
        )
    if expected_grid_size is not None:
        actual = metadata.get('epsilon_grid_size', metadata.get('grid_size'))
        if actual is None or int(actual) != int(expected_grid_size):
            raise ValueError(
                f'Oracle {artifact_path} has grid size {actual!r}; '
                f'expected {expected_grid_size}.'
            )
    if expected_scenarios is not None and int(metadata.get('scenarios', -1)) != int(expected_scenarios):
        raise ValueError(
            f'Oracle {artifact_path} contains {metadata.get("scenarios")!r} '
            f'scenarios; expected {expected_scenarios}.'
        )
    return metadata

def checkpoint_metadata(cfg: Mapping[str, Any], *, weight_key: str='ema', best_epoch: int | None=None, best_val_loss: float | None=None, **fields: Any) -> dict[str, Any]:
    if weight_key not in {'ema', 'model'}:
        raise ValueError("checkpoint weight_key must be 'ema' or 'model'.")
    metadata: dict[str, Any] = {'schema_version': CHECKPOINT_SCHEMA_VERSION, 'config_fingerprint': config_fingerprint(cfg), 'preferred_weights': weight_key, 'provenance': build_provenance(cfg)}
    if best_epoch is not None:
        metadata['best_epoch'] = int(best_epoch)
    if best_val_loss is not None:
        metadata['best_val_loss'] = float(best_val_loss)
    metadata.update(fields)
    return metadata

def checkpoint_weight_key(checkpoint: Mapping[str, Any]) -> str:
    metadata = checkpoint.get('metadata')
    if isinstance(metadata, Mapping):
        explicit = metadata.get('preferred_weights', metadata.get('weight_key'))
        if explicit is not None:
            if explicit not in {'ema', 'model'}:
                raise ValueError(f"Checkpoint metadata 'preferred_weights' must be 'ema' or 'model'; got {explicit!r}.")
            if explicit not in checkpoint:
                raise KeyError(f"Checkpoint metadata selects '{explicit}' weights, but that payload is missing.")
            return str(explicit)
    if 'ema' in checkpoint:
        return 'ema'
    if 'model' in checkpoint:
        return 'model'
    raise KeyError("Checkpoint contains neither 'ema' nor legacy 'model' weights.")

def checkpoint_weights(checkpoint: Mapping[str, Any]) -> Any:
    return checkpoint[checkpoint_weight_key(checkpoint)]
