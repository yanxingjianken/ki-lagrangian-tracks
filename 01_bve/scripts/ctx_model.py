"""Conditional kinematic interpolant: acceleration and velocity score of a parcel given the observed tracks of its patch.

Context tokens (one per observed parcel and time) are compressed by a Perceiver encoder into latent vectors; query
states (t, x, v) read the latents by cross-attention and return the acceleration and the scaled velocity score.
Units in and out are normalized (positions / 1000 km, velocities / 100 km h-1, accelerations / 10 km h-2,
time / 24 h, relative to the anchor time and the patch centroid).
"""
import math

import torch
from torch import nn

POS_KM, VEL_KMH, ACC_KMH2, TIME_H = 1000.0, 100.0, 10.0, 24.0


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

    Token features: (tau, x, y, sin lat, u, v); query features: the same for the state to be advanced.
    """

    def __init__(self, d=256, heads=8, latents=64, enc_layers=4, dec_layers=2):
        super().__init__()
        self.tok, self.qry = embed(6, 3, d), embed(6, 3, d)
        self.latents = nn.Parameter(torch.randn(latents, d) * 0.02)
        self.read = Block(d, heads, cross=True)
        self.enc = nn.ModuleList([Block(d, heads) for _ in range(enc_layers)])
        self.dec = nn.ModuleList([Block(d, heads, cross=True) for _ in range(dec_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4))
        nn.init.zeros_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)

    def encode(self, tokens, pad=None):
        """tokens (B, M, 6), pad (B, M) True where absent -> latents (B, L, d)."""
        z = self.latents.expand(tokens.shape[0], -1, -1)
        z = self.read(z, self.tok(tokens), pad)
        for blk in self.enc:
            z = blk(z)
        return z

    def decode(self, z, queries):
        """queries (B, Q, 6) -> acceleration (B, Q, 2) and scaled score (B, Q, 2)."""
        h = self.qry(queries)
        for blk in self.dec:
            h = blk(h, z)
        out = self.head(h)
        return out[..., :2], out[..., 2:]

    def forward(self, tokens, pad, queries):
        return self.decode(self.encode(tokens, pad), queries)
