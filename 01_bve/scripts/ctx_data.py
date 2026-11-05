"""Patches of hourly tracks on the GPU, minibatches for the conditional KI, and conditional forecast/backcast rollouts.

A sample is one patch, an anchor time t0 and a direction: forecast (context = all observed parcels over
[t0 - P, t0], targets after t0) or backcast (context over [t0, t0 + P], targets before t0).
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ctx_model import ACC_KMH2, POS_KM, TIME_H, VEL_KMH

OMEGA = np.pi * np.sqrt(3)
A_KM = 6371.0


def flow_names(config, split):
    return [f["name"] for f in json.loads(Path(config).read_text())["flows"] if f["split"] == split]


def load_patches(root, names, windows, patch_ids, n_per=1000, device="cuda", seed=0):
    """Stack patches of data/tracks_lagranto_<name>_w<k>.npz into (Np, T, n_per, ...) GPU tensors."""
    rng = np.random.default_rng(seed)
    X, V, A, S, meta = [], [], [], [], []
    for name in names:
        for w in windows:
            with np.load(Path(root) / f"data/tracks_lagranto_{name}_w{w}.npz") as f:
                x, v, a, lat, patch, t = f["x_km"], f["v_kmh"], f["a_kmh2"], f["lat"], f["patch"], f["time_h"]
            for p in patch_ids:
                idx = np.where(patch == p)[0]
                idx = idx if len(idx) == n_per else rng.choice(idx, n_per, replace=len(idx) < n_per)
                X.append(x[:, idx])
                V.append(v[:, idx])
                A.append(a[:, idx])
                S.append(np.sin(np.deg2rad(lat[:, idx])))
                meta.append((name, w, int(p)))
    to = lambda z: torch.tensor(np.stack(z), dtype=torch.float32, device=device)
    return SimpleNamespace(X=to(X), V=to(V), A=to(A), S=to(S), meta=meta, time_h=t)


def context(D, p, t0, direction, kidx, span_h=48, step_h=3):
    """Context tokens (B, Tc*K, 6) and the patch centroid at t0 (B, 2) for patches p, anchors t0, directions."""
    offs = torch.arange(0, span_h + 1, step_h, device=D.X.device)
    tc = t0[:, None] - direction[:, None] * offs[None]
    ix = (p[:, None, None], tc[:, :, None], kidx[:, None, :])
    centroid = D.X[p, t0].mean(1)
    tau = ((tc - t0[:, None]).float() / TIME_H)[:, :, None, None].expand(-1, -1, kidx.shape[1], 1)
    tok = torch.cat([tau, (D.X[ix] - centroid[:, None, None]) / POS_KM, D.S[ix][..., None], D.V[ix] / VEL_KMH], -1)
    return tok.flatten(1, 2), centroid


def sample_batch(D, B, K=128, Q=256, span_h=48, horizon_h=24, sx_km=5.0, delta_h=6.0, gen=None):
    """One minibatch: context tokens, query states around true tracks, target accelerations and scores."""
    dev, (Np, T, N) = D.X.device, D.X.shape[:3]
    r = lambda *s: torch.rand(*s, device=dev, generator=gen)
    p = (r(B) * Np).long()
    direction = (r(B) < 0.5).long() * 2 - 1
    lo = torch.where(direction > 0, span_h, horizon_h)
    hi = torch.where(direction > 0, T - 1 - horizon_h, T - 1 - span_h)
    t0 = lo + (r(B) * (hi - lo + 1)).long()
    kidx = r(B, N).argsort(1)[:, :K]
    tok, centroid = context(D, p, t0, direction, kidx, span_h)
    qi = (r(B, Q) * N).long()
    qt = t0[:, None] + direction[:, None] * (1 + (r(B, Q) * horizon_h).long())
    ix = (p[:, None], qt, qi)
    th = r(B, Q, 1) * 2 * np.pi
    z1, z2 = torch.randn(2, B, Q, 2, device=dev, generator=gen)
    par, perp = torch.cos(th) * z1 + torch.sin(th) * z2, -torch.sin(th) * z1 + torch.cos(th) * z2
    w = OMEGA / delta_h
    xq, vq = D.X[ix] + sx_km * par, D.V[ix] + sx_km * w * perp
    aq = D.A[ix] - sx_km * w**2 * par
    tau = ((qt - t0[:, None]).float() / TIME_H)[..., None]
    qry = torch.cat([tau, (xq - centroid[:, None]) / POS_KM, D.S[ix][..., None], vq / VEL_KMH], -1)
    return tok, qry, aq / ACC_KMH2, -perp


def sin_lat(x_km):
    r = torch.hypot(x_km[..., 0], x_km[..., 1])
    return torch.cos(2 * torch.atan(r / (2 * A_KM)))


@torch.no_grad()
def rollout(model, tok, centroid, x0, v0, direction, horizon_h=24, steps_per_h=6):
    """Integrate x' = v, v' = a(tau, x, v | context) from tau = 0 (Heun); direction (B,) +1 forward, -1 backward.

    x0, v0: (B, Q, 2) in km and km/h. Returns hourly positions (horizon_h + 1, B, Q, 2) in km.
    """
    z = model.encode(tok)
    dt = direction.float()[:, None, None] / steps_per_h

    def acc(tau, x, v):
        q = torch.cat([tau / TIME_H, (x - centroid[:, None]) / POS_KM, sin_lat(x)[..., None], v / VEL_KMH], -1)
        return model.decode(z, q)[0] * ACC_KMH2

    x, v, out = x0.clone(), v0.clone(), [x0.clone()]
    for s in range(horizon_h * steps_per_h):
        tau = (s * dt).expand(-1, x.shape[1], 1)
        a0 = acc(tau, x, v)
        xp, vp = x + dt * v, v + dt * a0
        x, v = x + 0.5 * dt * (v + vp), v + 0.5 * dt * (a0 + acc(tau + dt, xp, vp))
        if (s + 1) % steps_per_h == 0:
            out.append(x.clone())
    return torch.stack(out)
