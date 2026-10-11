from __future__ import annotations
import math
import torch
import torch.nn as nn

def sinusoidal_pos(length: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(length, dim)
    pos = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

class AxisAttention(nn.Module):

    def __init__(self, dim, heads):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        h = self.ln(x)
        out, _ = self.attn(h, h, h, need_weights=False)
        return x + out

class FFN(nn.Module):

    def __init__(self, dim, mult=4):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return x + self.net(self.ln(x))

class EncoderBlock(nn.Module):

    def __init__(self, dim, heads):
        super().__init__()
        self.temporal = AxisAttention(dim, heads)
        # ``spatial`` is retained as the state-dict key used by existing
        # checkpoints; this axis is the 12-signal channel axis, not geography.
        self.spatial = AxisAttention(dim, heads)
        self.ffn = FFN(dim)

    def forward(self, x):
        B, L, N, d = x.shape
        xt = x.permute(0, 2, 1, 3).reshape(B * N, L, d)
        xt = self.temporal(xt).reshape(B, N, L, d).permute(0, 2, 1, 3)
        xs = xt.reshape(B * L, N, d)
        xs = self.spatial(xs).reshape(B, L, N, d)
        return self.ffn(xs)

class PreferenceCrossAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, n_pref_tokens: int=2):
        super().__init__()
        if n_pref_tokens < 1:
            raise ValueError('n_pref_tokens must be >= 1')
        self.n_pref_tokens = int(n_pref_tokens)
        self.pref_mlp = nn.Sequential(nn.Linear(2, d_model), nn.SiLU(), nn.Linear(d_model, d_model * self.n_pref_tokens))
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = FFN(d_model)

    def forward(self, temporal, lam):
        batch = temporal.shape[0]
        kv = self.ln_kv(self.pref_mlp(lam).view(batch, self.n_pref_tokens, -1))
        out, _ = self.attn(self.ln_q(temporal), kv, kv, need_weights=False)
        return self.ffn(temporal + out)

class TemporalChannelEncoder(nn.Module):
    """Encode the full-day state with temporal and channel attention.

    The module used to be exposed as ``TemporalSpatialEncoder``.  ``spatial``
    in that historical name referred to the non-temporal signal/channel axis,
    not to a geographic spatial graph.  The public name now matches the
    temporal--channel formulation used in the manuscript.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg['model']
        self.L = cfg['horizon']['L']
        self.N = m['n_channels']
        self.d = m['d_model']
        self.proj_w = nn.Parameter(torch.randn(self.N, self.d) * 0.1)
        self.proj_b = nn.Parameter(torch.zeros(self.N, self.d))
        self.ch_emb = nn.Parameter(torch.randn(self.N, self.d) * 0.02)
        self.register_buffer('pos_emb', sinusoidal_pos(self.L, self.d))
        self.blocks = nn.ModuleList([EncoderBlock(self.d, m['n_heads']) for _ in range(m['n_enc_blocks'])])

    def forward(self, state):
        x = state.unsqueeze(-1) * self.proj_w + self.proj_b
        x = x + self.ch_emb.view(1, 1, self.N, self.d)
        x = x + self.pos_emb.view(1, self.L, 1, self.d)
        for blk in self.blocks:
            x = blk(x)
        return x


# Keep the historical import available for checkpoints and downstream scripts.
TemporalSpatialEncoder = TemporalChannelEncoder
