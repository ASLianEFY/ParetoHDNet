from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def step_embedding(k, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=k.device) / half)
    args = k.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)

class ResBlock1D(nn.Module):

    def __init__(self, in_ch, out_ch, c_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.ada = nn.Linear(c_dim, 2 * out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, c):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.ada(c).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = self.conv2(F.silu(h))
        return h + self.skip(x)

class CrossAttn1D(nn.Module):

    def __init__(self, channels, ctx_dim, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, heads, kdim=ctx_dim, vdim=ctx_dim, batch_first=True)

    def forward(self, x, ctx):
        B, C, T = x.shape
        q = self.norm(x).transpose(1, 2)
        out, _ = self.attn(q, ctx, ctx, need_weights=False)
        return x + out.transpose(1, 2)

class SelfAttn1D(nn.Module):

    def __init__(self, channels, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x):
        B, C, T = x.shape
        h = self.norm(x).transpose(1, 2)
        out, _ = self.attn(h, h, h, need_weights=False)
        return x + out.transpose(1, 2)

class UNet1D(nn.Module):

    def __init__(self, cfg: dict, action_dim: int=6):
        super().__init__()
        m = cfg['model']
        base = m['unet_base']
        n_stages = m['unet_stages']
        d_model = m['d_model']
        c_dim = base * 4
        chans = [base * 2 ** i for i in range(n_stages)]
        self.step_dim = base
        self.step_mlp = nn.Sequential(nn.Linear(base, c_dim), nn.SiLU(), nn.Linear(c_dim, c_dim))
        self.lam_mlp = nn.Sequential(nn.Linear(2, c_dim), nn.SiLU(), nn.Linear(c_dim, c_dim))
        self.null_lam = nn.Parameter(torch.zeros(2))
        self.in_proj = nn.Conv1d(action_dim, chans[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.down_cross = nn.ModuleList()
        self.down_self = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i in range(n_stages):
            ic = chans[max(i - 1, 0)] if i > 0 else chans[0]
            oc = chans[i]
            self.down_blocks.append(nn.ModuleList([ResBlock1D(ic if j == 0 else oc, oc, c_dim) for j in range(2)]))
            use_attn = i >= n_stages - 2
            self.down_cross.append(CrossAttn1D(oc, d_model) if use_attn else None)
            self.down_self.append(SelfAttn1D(oc) if use_attn else None)
            self.downsample.append(nn.Conv1d(oc, oc, 3, stride=2, padding=1) if i < n_stages - 1 else None)
        self.mid1 = ResBlock1D(chans[-1], chans[-1], c_dim)
        self.mid_cross = CrossAttn1D(chans[-1], d_model)
        self.mid2 = ResBlock1D(chans[-1], chans[-1], c_dim)
        self.up_blocks = nn.ModuleList()
        self.up_cross = nn.ModuleList()
        for j, i in enumerate(reversed(range(n_stages))):
            oc = chans[i]
            prev_ch = chans[-1] if j == 0 else chans[i + 1]
            cat_ch = prev_ch + chans[i]
            self.up_blocks.append(nn.ModuleList([ResBlock1D(cat_ch, oc, c_dim), ResBlock1D(oc, oc, c_dim)]))
            use_attn = i >= n_stages - 2
            self.up_cross.append(CrossAttn1D(oc, d_model) if use_attn else None)
        self.out_norm = nn.GroupNorm(8, chans[0])
        self.out_proj = nn.Conv1d(chans[0], action_dim, 3, padding=1)
        self.n_stages = n_stages

    def cond_vector(self, k, lam):
        c = self.step_mlp(step_embedding(k, self.step_dim)) + self.lam_mlp(lam)
        return c

    def forward(self, a_k, k, ctx, lam):
        c = self.cond_vector(k, lam)
        x = self.in_proj(a_k.transpose(1, 2))
        skips = []
        for i in range(self.n_stages):
            for blk in self.down_blocks[i]:
                x = blk(x, c)
            if self.down_self[i] is not None:
                x = self.down_self[i](x)
            if self.down_cross[i] is not None:
                x = self.down_cross[i](x, ctx)
            skips.append(x)
            if self.downsample[i] is not None:
                x = self.downsample[i](x)
        x = self.mid1(x, c)
        x = self.mid_cross(x, ctx)
        x = self.mid2(x, c)
        for j, i in enumerate(reversed(range(self.n_stages))):
            skip = skips[i]
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode='nearest')
            x = torch.cat([x, skip], dim=1)
            for blk in self.up_blocks[j]:
                x = blk(x, c)
            if self.up_cross[j] is not None:
                x = self.up_cross[j](x, ctx)
        x = self.out_proj(F.silu(self.out_norm(x)))
        return x.transpose(1, 2)
