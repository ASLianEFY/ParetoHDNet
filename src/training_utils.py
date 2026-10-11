"""Learning-rate schedule, RNG snapshots and atomic checkpoint writes."""
import math
import random
import numpy as np
import torch

def _scheduled_lr(cfg, step, total_steps):
    base_lr = float(cfg['train']['lr'])
    warmup = int(cfg['train']['warmup_steps'])
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup
    if cfg['train'].get('lr_schedule', 'constant') != 'cosine':
        return base_lr
    min_lr = float(cfg['train'].get('min_lr', 0.0))
    denominator = max(total_steps - warmup - 1, 1)
    progress = min(max((step - warmup) / denominator, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _rng_state():
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def _atomic_torch_save(value, path):
    temporary = path.with_name(f'.{path.name}.tmp')
    torch.save(value, temporary)
    temporary.replace(path)
