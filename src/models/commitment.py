from __future__ import annotations

import torch


@torch.no_grad()
def project_commitment_logits(
    logits: torch.Tensor,
    margin: float,
    min_up: int,
    min_down: int,
) -> torch.Tensor:
    """Project commitment logits with a batched GPU time scan."""
    if logits.ndim < 3:
        raise ValueError('commitment logits must have shape (..., T, units)')
    if min_up < 1 or min_down < 1:
        raise ValueError('minimum up/down durations must be positive')
    requested = logits > 0
    state = torch.zeros(
        requested.shape[:-2] + (requested.shape[-1],),
        dtype=torch.bool,
        device=requested.device,
    )
    locked = torch.zeros_like(state, dtype=torch.long)
    projected_steps = []
    for step in range(requested.shape[-2]):
        desired = requested[..., step, :]
        switching = (locked == 0) & (desired != state)
        state = torch.where(switching, desired, state)
        duration = torch.where(
            state,
            torch.full_like(locked, int(min_up) - 1),
            torch.full_like(locked, int(min_down) - 1),
        )
        locked = torch.where(switching, duration, torch.clamp(locked - 1, min=0))
        projected_steps.append(state)
    projected = torch.stack(projected_steps, dim=-2)
    positive = torch.full_like(logits, float(margin))
    return torch.where(projected, positive, -positive)
