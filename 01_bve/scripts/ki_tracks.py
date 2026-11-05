"""Kinematic-interpolant networks trained on true parcel tracks, and the rollouts that fill gaps and forecast.

Model (after Jiang et al. 2026, arXiv:2610.22519, notebook oceans_currents_interpolant.ipynb): x' = v, v' = a(t, x, v),
plus a velocity-score network s(t, x, v) for the stochastic sampler. The mean path is the parcel's own track (known
identities), so the targets are its true acceleration; the rotating Gaussian noise of the paper adds off-track states
whose target acceleration pulls back toward the track.
"""
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

OMEGA = np.pi * np.sqrt(3)


class Field(nn.Module):
    """MLP f(t, x, v) with polynomial and Fourier features of t (the paper's Field, sizes and frequencies configurable)."""

    def __init__(self, state_dim=4, width=256, depth=2, freqs=(1, 2, 4), out_dim=2):
        super().__init__()
        self.freqs = tuple(freqs)
        layers = [nn.Linear(4 + 2 * len(self.freqs) + state_dim, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        self.net = nn.Sequential(*layers, nn.Linear(width, out_dim))
        for layer in self.net[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t, x, v):
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(-1, 1).expand(x.shape[0], 1)
        feats = [torch.ones_like(t), t, t**2, t**3]
        for f in self.freqs:
            feats += [torch.sin(2 * np.pi * f * t), torch.cos(2 * np.pi * f * t)]
        return self.net(torch.cat(feats + [x, v], dim=-1))


class KINet(nn.ModuleList):
    """The acceleration network [0] and the velocity-score network [1]; one forward call returns both (as DDP needs)."""

    def __init__(self, **field_kwargs):
        super().__init__([Field(**field_kwargs), Field(**field_kwargs)])

    def forward(self, t, x, v):
        return self[0](t, x, v), self[1](t, x, v)


def load_patch(path, patch, t_end_h, noise=1.0, delta_h=6.0, t_start_h=0.0, center=None, scale=None):
    """Tracks of one patch (patch < 0: all parcels), normalized like the paper on the window [t_start_h, t_end_h].

    Model time is (t - t_start_h) / (t_end_h - t_start_h), so the window is [0, 1]; times before it (backcast) are
    negative and after it (forecast) exceed 1. The training parcels inside the window set the centre and scale
    unless they are given (to apply a trained model to other data).

    Returns a namespace with x, v, a (T, N, 2) inside the window and x_all, v_all, a_all, t_all over the whole
    track, all in model units, the split indices, and the conversion constants (centre, scale in km, clock in h).
    """
    with np.load(path) as f:
        sel = f["patch"] == patch if patch >= 0 else np.ones(len(f["patch"]), bool)
        time_h = f["time_h"]
        keep = (time_h >= t_start_h - 1e-9) & (time_h <= t_end_h + 1e-9)
        X, Vk, Ak = f["x_km"][:, sel], f["v_kmh"][:, sel], f["a_kmh2"][:, sel]
        split = f["split"][sel]
    train, val, test = (np.where(split == s)[0] for s in (0, 1, 2))
    if center is None:
        center = X[keep][:, train].reshape(-1, 2).mean(0)
        scale = float(np.sqrt(((X[keep][:, train] - center) ** 2).mean()))
    clock = float(t_end_h - t_start_h)
    h = delta_h / clock
    sx = noise * h**1.5 / (8 * np.sqrt(3))
    t_all = (time_h - t_start_h) / clock
    return SimpleNamespace(x_all=(X - center) / scale, v_all=Vk * clock / scale, a_all=Ak * clock**2 / scale,
                           t_all=t_all, keep=keep, x=(X[keep] - center) / scale, v=Vk[keep] * clock / scale,
                           a=Ak[keep] * clock**2 / scale, t=t_all[keep], train=train, val=val, test=test,
                           center=np.asarray(center), scale=float(scale), clock=clock, h=h, sx=sx, sv=sx * OMEGA / h)


def training_batch(d, rng, idx, batch, device):
    """States on and around true tracks with their target acceleration and scaled velocity score."""
    i = rng.choice(idx, batch)
    k = rng.integers(len(d.t), size=batch)
    x, v, a = d.x[k, i], d.v[k, i], d.a[k, i]
    z1, z2 = rng.standard_normal((2, batch, 2))
    th = rng.uniform(0.0, 2 * np.pi, (batch, 1))
    par = np.cos(th) * z1 + np.sin(th) * z2
    perp = -np.sin(th) * z1 + np.cos(th) * z2
    arrays = (d.t[k], x + d.sx * par, v + d.sv * perp, a - d.sx * (OMEGA / d.h) ** 2 * par, -perp)
    return tuple(torch.tensor(np.asarray(z), dtype=torch.float32, device=device) for z in arrays)


def batch_loss(model, batch, score_weight=1.0):
    t, x, v, a, q = batch
    a_hat, q_hat = model(t, x, v)
    return ((a_hat - a) ** 2).mean() + score_weight * ((q_hat - q) ** 2).mean()


@torch.no_grad()
def integrate(model, d, x0, v0, t0, t1, n_steps, kappa=0.0, seed=0):
    """Heun (deterministic) or Euler-Maruyama (kappa > 0) integration from t0 to t1 (t1 < t0 runs backward).

    t0 and t1 are scalars or per-parcel arrays (N,). Returns positions (n_steps + 1, N, 2) in model units.
    """
    a_net, s_net = model
    gen = torch.Generator(device=x0.device).manual_seed(seed)
    col = lambda t: torch.as_tensor(t, dtype=x0.dtype, device=x0.device).reshape(-1, 1).expand(x0.shape[0], 1)
    t0, t1 = col(t0), col(t1)
    dt = (t1 - t0) / n_steps
    x, v, path = x0.clone(), v0.clone(), [x0.clone()]
    for s in range(n_steps):
        t, tn = t0 + s * dt, t0 + (s + 1) * dt
        if kappa:
            b = a_net(t, x, v) + kappa * s_net(t, x, v) / d.sv
            noise = torch.sqrt(2 * kappa * dt.abs()) * torch.randn(v.shape, generator=gen, device=v.device)
            x, v = x + dt * v, v + dt * b + noise
        else:
            b = a_net(t, x, v)
            xp, vp = x + dt * v, v + dt * b
            x, v = x + 0.5 * dt * (v + vp), v + 0.5 * dt * (b + a_net(tn, xp, vp))
        path.append(x.clone())
    return torch.stack(path)


def infill(model, d, x0, v0, x1, v1, t0, t1, n_steps):
    """Fill a gap: forward from (x0, v0) at t0, backward from (x1, v1) at t1, blended with a smoothstep weight."""
    fwd = integrate(model, d, x0, v0, t0, t1, n_steps)
    bwd = integrate(model, d, x1, v1, t1, t0, n_steps).flip(0)
    u = torch.linspace(0, 1, n_steps + 1, device=x0.device)[:, None, None]
    w = 3 * u**2 - 2 * u**3
    return (1 - w) * fwd + w * bwd


def stereo_to_lonlat(x_km):
    """Inverse north-polar stereographic projection (tangent at the pole) -> lon, lat in degrees."""
    r = np.hypot(x_km[..., 0], x_km[..., 1])
    lat = 90.0 - np.rad2deg(2 * np.arctan(r / (2 * 6371.0)))
    lon = np.rad2deg(np.arctan2(x_km[..., 0], -x_km[..., 1])) % 360.0
    return lon, lat


def lonlat_to_stereo(lon, lat):
    """North-polar stereographic projection (tangent at the pole) in km."""
    r = 2 * 6371.0 * np.tan(np.deg2rad(90.0 - lat) / 2)
    lam = np.deg2rad(lon)
    return np.stack([r * np.sin(lam), -r * np.cos(lam)], axis=-1)


def great_circle_km(lon1, lat1, lon2, lat2):
    p1, p2, dl = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon2 - lon1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return 6371.0 * np.arccos(np.clip(c, -1.0, 1.0))


def to_km(d, x_model):
    return np.asarray(x_model) * d.scale + d.center
