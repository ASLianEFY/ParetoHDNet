from __future__ import annotations
import copy
import math
import torch
import torch.nn as nn
from .encoder import PreferenceCrossAttention, TemporalChannelEncoder
from .commitment import project_commitment_logits
from .unet1d import UNet1D

def cosine_alpha_bar(K, s=0.008):
    steps = torch.arange(K + 1, dtype=torch.float64)
    f = torch.cos((steps / K + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    return abar[1:].clamp_min(1e-05).float()

class DiffusionDecoder(nn.Module):

    def __init__(self, cfg: dict, action_dim: int | None=None):
        super().__init__()
        if action_dim is None:
            action_dim = int(cfg.get('uc', {}).get('action_dim', 9))
        if action_dim != 9:
            raise ValueError('lean hybrid UC diffusion requires action_dim=9')
        self.cfg = copy.deepcopy(cfg)
        self.K = int(cfg['model']['K_train'])
        self.K_infer = int(cfg['model']['K_infer'])
        self.cfg_w = float(cfg['model']['cfg_w'])
        self.T = int(cfg['horizon']['T'])
        full_cfg = cfg.get('full_diffusion', {})
        self.hybrid_uc = True
        self.architecture_version = int(full_cfg.get('architecture_version', 3))
        self.hybrid_v3 = True
        self.p_uncond = float(full_cfg.get('p_uncond', cfg['model']['p_uncond']))
        self.phys_beta = float(cfg['model']['phys_beta'])
        self.action_dim = 9
        self.use_commitment_head = bool(full_cfg.get('commitment_head', True))
        self.use_commitment_projection = bool(full_cfg.get('commitment_projection', True))
        self.diffusion_dim = 6 if self.use_commitment_head else 9
        self.margin = float(cfg['model'].get('a0_margin', 1.0))
        self.x0_weight = float(full_cfg.get('x0_weight', 1.0))
        self.commitment_weight = float(full_cfg.get('commitment_weight', 1.0))
        self.commitment_teacher_forcing = float(full_cfg.get('commitment_teacher_forcing', 0.0))
        self.anchor_weight = float(full_cfg.get('anchor_weight', 1.0))
        self.anchor_context = bool(full_cfg.get('anchor_context', False))
        self.training_stage = 'joint'
        self.encoder = TemporalChannelEncoder(cfg)
        self.unet = UNet1D(cfg, self.diffusion_dim)
        d = cfg['model']['d_model']
        self.null_ctx = nn.Parameter(torch.zeros(1, 1, d))
        self.preference_fusion = PreferenceCrossAttention(d, int(cfg['model']['n_heads']), n_pref_tokens=int(cfg['model'].get('n_pref_tokens', 2)))
        self.commitment_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3)) if self.use_commitment_head else None
        self.commitment_context = nn.Linear(3, d)
        self.continuous_anchor_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 6))
        self.continuous_anchor_context = nn.Linear(6, d)
        self.register_buffer('abar', cosine_alpha_bar(self.K))
        objective_scales = full_cfg.get('objective_scales', [1.0, 1.0])
        self.register_buffer('objective_scales', torch.as_tensor(objective_scales, dtype=torch.float32))

    def _hybrid_features(self, state, lam):
        z = self.encoder(state)
        B = z.shape[0]
        base_context = z.reshape(B, -1, z.shape[-1])
        temporal = self.preference_fusion(z.mean(dim=2), lam)
        if self.commitment_head is None:
            commitment_logits = temporal.new_zeros(temporal.shape[0], temporal.shape[1], 3)
        else:
            commitment_logits = self.commitment_head(temporal)
        anchor = self.margin * torch.tanh(self.continuous_anchor_head(temporal))
        return (base_context, temporal, commitment_logits, anchor)

    def _hybrid_context(self, base_context, temporal, commitment, anchor):
        tokens = [base_context, temporal, self.commitment_context(commitment)]
        if self.anchor_context:
            tokens.append(self.continuous_anchor_context(anchor))
        return torch.cat(tokens, dim=1)

    def encode(self, state, lam=None):
        if lam is None:
            raise ValueError('hybrid UC diffusion requires preference conditioning')
        base_context, temporal, commitment_logits, anchor = self._hybrid_features(state, lam)
        return (self._hybrid_context(base_context, temporal, torch.sigmoid(commitment_logits), anchor), commitment_logits)

    def configure_training_stage(self, epoch):
        self.training_stage = 'joint'
        return self.training_stage

    def _gather(self, k):
        return self.abar[k - 1]

    @staticmethod
    def _straight_through_commitment(logits):
        soft = torch.sigmoid(logits)
        hard = (soft >= 0.5).to(dtype=soft.dtype)
        return hard + soft - soft.detach()

    @torch.no_grad()
    def _project_commitment(self, logits):
        return project_commitment_logits(
            logits, self.margin, int(self.cfg['uc']['min_up']), int(self.cfg['uc']['min_down'])
        )

    def loss(self, state, raw_state, lam, a0, *, objectives=None, apply_cfg_dropout=True, generator=None, timesteps=None, noise=None, teacher_mask=None):
        B = a0.shape[0]
        dev = a0.device
        target_on = (a0[..., 6:9] > 0).to(dtype=a0.dtype)
        base_context, temporal, commitment_logits, anchor = self._hybrid_features(state, lam)
        commitment_condition = torch.sigmoid(commitment_logits)
        if self.use_commitment_head and apply_cfg_dropout and self.commitment_teacher_forcing:
            if teacher_mask is None:
                teacher = torch.rand(B, device=dev, generator=generator) < self.commitment_teacher_forcing
            else:
                teacher = teacher_mask.to(device=dev, dtype=torch.bool)
                if teacher.shape != (B,):
                    raise ValueError(f'teacher_mask must have shape ({B},)')
            commitment_condition = torch.where(teacher.view(B, 1, 1), target_on, commitment_condition)
        ctx = self._hybrid_context(base_context, temporal, commitment_condition, anchor)
        if self.commitment_head is None:
            loss_binary = a0.new_zeros(())
        else:
            loss_binary = nn.functional.binary_cross_entropy_with_logits(commitment_logits, target_on)
        loss_anchor = nn.functional.smooth_l1_loss(anchor, a0[..., :6])
        if timesteps is None:
            timesteps = torch.randint(1, self.K + 1, (B,), device=dev, generator=generator)
        else:
            timesteps = timesteps.to(device=dev)
            if timesteps.is_floating_point() and (not torch.all(timesteps == timesteps.round())):
                raise ValueError('timesteps must contain integer values')
            timesteps = timesteps.to(dtype=torch.long)
            if timesteps.shape != (B,) or torch.any((timesteps < 1) | (timesteps > self.K)):
                raise ValueError(f'timesteps must have shape ({B},) with values in [1, {self.K}]')
        abar = self._gather(timesteps).view(B, 1, 1)
        diffusion_target = a0[..., :self.diffusion_dim]
        if noise is None:
            noise = torch.randn(diffusion_target.shape, dtype=a0.dtype, device=dev, generator=generator)
        else:
            noise = noise.to(device=dev, dtype=a0.dtype)
            if noise.shape == a0.shape:
                noise = noise[..., :self.diffusion_dim]
            if noise.shape != diffusion_target.shape:
                raise ValueError(f'noise must have shape {tuple(diffusion_target.shape)}; got {tuple(noise.shape)}')
        a_k = abar.sqrt() * diffusion_target + (1 - abar).sqrt() * noise
        eps_pred = self.unet(a_k, timesteps, ctx, lam)
        loss_eps = (eps_pred - noise).square().mean()
        x0_raw = (a_k - (1 - abar).sqrt() * eps_pred) / abar.sqrt()
        x0_hat = x0_raw.clamp(-self.margin, self.margin)
        x0_per_sample = nn.functional.smooth_l1_loss(x0_hat, diffusion_target, reduction='none').mean(dim=(1, 2))
        x0_weights = abar.flatten()
        loss_x0 = (x0_weights * x0_per_sample).sum() / x0_weights.sum().clamp_min(1e-08)
        loss = loss_eps + self.x0_weight * loss_x0 + self.anchor_weight * loss_anchor + self.commitment_weight * loss_binary
        zero = a0.new_zeros(())
        return (loss, loss_eps.detach(), zero)

    def validation_loss(self, state, raw_state, lam, a0, *, objectives=None, generator=None, timesteps=None, noise=None):
        return self.loss(state, raw_state, lam, a0, apply_cfg_dropout=False, generator=generator, timesteps=timesteps, noise=noise, objectives=objectives)

    @torch.no_grad()
    def sample(self, state, lam, w=None, K_infer=None, M=1):
        K_infer = self.K_infer if K_infer is None else K_infer
        if isinstance(K_infer, bool) or not isinstance(K_infer, int) or (not 1 <= K_infer <= self.K):
            raise ValueError(f'K_infer must be an integer in [1, {self.K}]')
        if isinstance(M, bool) or not isinstance(M, int) or M < 1:
            raise ValueError('M must be a positive integer')
        dev = state.device
        B = state.shape[0]
        base_context, temporal, commitment_logits, anchor = self._hybrid_features(state, lam)
        if self.use_commitment_head:
            requests = commitment_logits.unsqueeze(1).expand(-1, M, -1, -1).clone()
            flat_requests = requests.reshape(B * M, self.T, 3)
            if self.use_commitment_projection:
                commitment = self._project_commitment(flat_requests).view(B, M, self.T, 3)
            else:
                positive = torch.full_like(flat_requests, self.margin)
                commitment = torch.where(flat_requests > 0, positive, -positive).view(B, M, self.T, 3)
            commitment_condition = (commitment > 0).to(dtype=state.dtype).reshape(B * M, self.T, 3)
        else:
            commitment = None
            commitment_condition = state.new_full((B * M, self.T, 3), 0.5)
        ctx = self._hybrid_context(base_context.repeat_interleave(M, dim=0), temporal.repeat_interleave(M, dim=0), commitment_condition, anchor.repeat_interleave(M, dim=0))
        lam = lam.repeat_interleave(M, dim=0)
        BM = B * M
        steps = torch.linspace(self.K, 1, K_infer, device=dev).round().long()
        steps = torch.unique_consecutive(steps)
        x = torch.randn(BM, self.T, self.diffusion_dim, device=dev, dtype=self.null_ctx.dtype)
        for idx in range(len(steps)):
            k = steps[idx].expand(BM)
            abar_k = self._gather(k).view(BM, 1, 1)
            eps = self.unet(x, k, ctx, lam)
            x0 = (x - (1 - abar_k).sqrt() * eps) / abar_k.sqrt()
            x0 = x0.clamp(-self.margin, self.margin)
            if idx < len(steps) - 1:
                k_next = steps[idx + 1].expand(BM)
                abar_n = self._gather(k_next).view(BM, 1, 1)
                x = abar_n.sqrt() * x0 + (1 - abar_n).sqrt() * eps
            else:
                x = x0
        generated = x.view(B, M, self.T, self.diffusion_dim)
        if self.use_commitment_head:
            return torch.cat([generated, commitment], dim=-1)
        switches = generated[..., 6:9].reshape(B * M, self.T, 3)
        if self.use_commitment_projection:
            switches = self._project_commitment(switches)
        else:
            positive = torch.full_like(switches, self.margin)
            switches = torch.where(switches > 0, positive, -positive)
        return torch.cat([
            generated[..., :6], switches.view(B, M, self.T, 3)
        ], dim=-1)
