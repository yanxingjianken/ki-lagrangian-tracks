# Copy of roger/03_era5/scripts/ctx_model.py without the scale constants (kiforecast.py takes them from the checkpoint).
"""Conditional kinematic interpolant in three dimensions: acceleration and velocity score of a parcel given the observed
tracks of its patch.

State: horizontal polar-stereographic position (x, y) and log-pressure height z = -H ln(p / 1000 hPa). Context tokens
(one per observed parcel and time) are compressed by a Perceiver encoder; query states read the latents by
cross-attention and return the acceleration (3) and the scaled velocity score (3). Inputs and outputs are normalized
per axis (POS, VEL, ACC) relative to the anchor time and the patch centroid; time / 24 h.
"""
import math

import torch
from torch import nn

N_FEAT = 8


class Fourier(nn.Module):
    """Raw features plus sin/cos of the first `n_fourier` features at octave frequencies."""

    def __init__(self, n_in, n_fourier, octaves=6):
        super().__init__()
        self.n_fourier = n_fourier
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(octaves, dtype=torch.float32))
        self.n_out = n_in + 2 * n_fourier * octaves

    def forward(self, z):
        f = z[..., : self.n_fourier, None] * self.freqs
        return torch.cat([z, torch.sin(f).flatten(-2), torch.cos(f).flatten(-2)], dim=-1)


class Block(nn.Module):
    """Pre-norm attention block (cross-attention when a context is given) with an MLP."""

    def __init__(self, d, heads, cross=False):
        super().__init__()
        self.nq = nn.LayerNorm(d)
        self.nk = nn.LayerNorm(d) if cross else None
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, kv=None, pad=None):
        h = self.nq(q)
        k = h if kv is None else self.nk(kv)
        q = q + self.attn(h, k, k, key_padding_mask=pad, need_weights=False)[0]
        return q + self.mlp(q)


def embed(n_in, n_fourier, d):
    f = Fourier(n_in, n_fourier)
    return nn.Sequential(f, nn.Linear(f.n_out, d), nn.GELU(), nn.Linear(d, d))


class ContextKI(nn.Module):
    """Perceiver encoder over context tokens and a cross-attention decoder for query states.

    Token features: (tau, x, y, z, sin lat, u, v, w); query features: the same for the state to be advanced.
    """

    def __init__(self, d=384, heads=8, latents=96, enc_layers=6, dec_layers=3):
        super().__init__()
        self.tok, self.qry = embed(N_FEAT, 4, d), embed(N_FEAT, 4, d)
        self.latents = nn.Parameter(torch.randn(latents, d) * 0.02)
        self.read = Block(d, heads, cross=True)
        self.enc = nn.ModuleList([Block(d, heads) for _ in range(enc_layers)])
        self.dec = nn.ModuleList([Block(d, heads, cross=True) for _ in range(dec_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 6))
        nn.init.zeros_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)

    def encode(self, tokens, pad=None):
        """tokens (B, M, 8), pad (B, M) True where absent -> latents (B, L, d)."""
        z = self.latents.expand(tokens.shape[0], -1, -1)
        z = self.read(z, self.tok(tokens), pad)
        for blk in self.enc:
            z = blk(z)
        return z

    def decode(self, z, queries):
        """queries (B, Q, 8) -> acceleration (B, Q, 3) and scaled score (B, Q, 3)."""
        h = self.qry(queries)
        for blk in self.dec:
            h = blk(h, z)
        out = self.head(h)
        return out[..., :3], out[..., 3:]

    def forward(self, tokens, pad, queries):
        return self.decode(self.encode(tokens, pad), queries)
