# Adapted from roger/02_dinosaur/scripts/ctx_data.py (ERA5 releases, sharded loading, sin latitude from x).
"""ERA5 patches of hourly tracks on the GPU, minibatches for the conditional KI, and forecast/backcast rollouts.

A sample is one patch (release, start level, patch), an anchor time t0 and a direction: forecast (context = the
patch's observed parcels over [t0 - P, t0], targets after t0) or backcast (context over [t0, t0 + P], targets before).
Horizontal positions are taken relative to the patch centroid at t0; height z is absolute. Southern-hemisphere
patches are stored mirrored to the north (12_dataset.py), so sin(latitude) follows from the stereographic radius.
Only x, v, a (float32) live on the GPU: 157 h x 9 x 4 bytes = 5.65 kB per track.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ctx_model import ACC, POS, TIME_H, VEL

OMEGA = np.pi * np.sqrt(3)
A_KM = 6371.0
ACC_CLIP = 30.0  # normalized; removes the kinks where caltra -j lifts a parcel back above the ground


def release_names(root, split):
    cfg = json.loads((Path(root) / "configs/releases.json").read_text())
    return [r["name"] for r in cfg["releases"] if r["split"] == split]


def patch_ids(root, part):
    """Patch ids of a part of the split-by-patch rule: train, validation, test or unseen_latitudes."""
    return json.loads((Path(root) / "configs/releases.json").read_text())["split_by_patch"][part]


def _groups(level, patch, ids):
    return [(int(k), int(q)) for k in np.unique(level) for q in ids if np.any((level == k) & (patch == q))]


def load_patches(root, names, ids, n_per=480, device="cuda", seed=0, extra=False, max_patches=None):
    """Patches (release, level, patch id in ids) of data/tracks/tracks_<release>.npz as (Np, T, n_per, 3) tensors.

    The tensors are allocated once on the device and filled release by release (no host-side stack). A patch that
    lost parcels (start below the ground) is filled up by repeating some of its parcels; the count is returned in
    meta. extra=True also returns latitude and pressure (for evaluation). max_patches keeps a random subset.
    """
    rng = np.random.default_rng(seed)
    plan = []
    for name in names:
        with np.load(Path(root) / f"data/tracks/tracks_{name}.npz") as f:
            level, patch = f["level"], f["patch"]
            T = f["time_h"].size
        plan += [(name, k, q) for k, q in _groups(level, patch, ids)]
    if max_patches is not None and len(plan) > max_patches:
        plan = [plan[i] for i in sorted(rng.choice(len(plan), max_patches, replace=False))]
    Np = len(plan)
    new = lambda *s: torch.empty(*s, dtype=torch.float32, device=device)  # noqa: E731
    X, V, A = new(Np, T, n_per, 3), new(Np, T, n_per, 3), new(Np, T, n_per, 3)
    LAT, P = (new(Np, T, n_per), new(Np, T, n_per)) if extra else (None, None)
    meta, filled, i = [], 0, 0
    for name in dict.fromkeys(n for n, _, _ in plan):
        mine = [(k, q) for n, k, q in plan if n == name]
        with np.load(Path(root) / f"data/tracks/tracks_{name}.npz") as f:
            level, patch, hemi = f["level"], f["patch"], f["hemisphere"]
            x, v, a = f["x_km"], f["v_kmh"], f["a_kmh2"]
            lat, p = (f["lat"], f["p_hpa"]) if extra else (None, None)
            t = f["time_h"]
        for k, q in mine:
            idx = np.flatnonzero((level == k) & (patch == q))
            if len(idx) < n_per:
                filled += n_per - len(idx)
                idx = np.r_[idx, rng.choice(idx, n_per - len(idx))]
            elif len(idx) > n_per:
                idx = rng.choice(idx, n_per, replace=False)
            for dst, src in ((X, x), (V, v), (A, a)) + (((LAT, lat), (P, p)) if extra else ()):
                dst[i] = torch.from_numpy(np.ascontiguousarray(src[:, idx])).to(device)
            meta.append((name, k, q, int(hemi[idx[0]])))
            i += 1
    return SimpleNamespace(X=X, V=V, A=A, LAT=LAT, P=P, meta=meta, time_h=t, filled=filled)


def _scale(dev):
    return tuple(torch.tensor(s, dtype=torch.float32, device=dev) for s in (POS, VEL, ACC))


def sin_lat(x_km):
    r = torch.hypot(x_km[..., 0], x_km[..., 1])
    return torch.cos(2 * torch.atan(r / (2 * A_KM)))


def context(D, p, t0, direction, kidx, span_h=48, step_h=3):
    """Context tokens (B, Tc*K, 8) and the horizontal patch centroid at t0 (B, 2)."""
    pos, vel, _ = _scale(D.X.device)
    offs = torch.arange(0, span_h + 1, step_h, device=D.X.device)
    tc = t0[:, None] - direction[:, None] * offs[None]
    ix = (p[:, None, None], tc[:, :, None], kidx[:, None, :])
    centroid = D.X[p, t0, :, :2].mean(1)
    x = D.X[ix]
    s = sin_lat(x)
    x = torch.cat([x[..., :2] - centroid[:, None, None], x[..., 2:]], -1)
    tau = ((tc - t0[:, None]).float() / TIME_H)[:, :, None, None].expand(-1, -1, kidx.shape[1], 1)
    tok = torch.cat([tau, x / pos, s[..., None], D.V[ix] / vel], -1)
    return tok.flatten(1, 2), centroid


def sample_batch(D, B, K=192, Q=256, span_h=48, horizon_h=24, sx_km=(5.0, 5.0, 0.01), delta_h=6.0, gen=None):
    """One minibatch: context tokens, query states around true tracks, target accelerations and scores.

    Context parcels and query parcels are disjoint (K + Q <= parcels per patch).
    """
    dev, (Np, T, N) = D.X.device, D.X.shape[:3]
    assert K + Q <= N, (K, Q, N)
    pos, vel, acc = _scale(dev)
    sx = torch.tensor(sx_km, dtype=torch.float32, device=dev)
    r = lambda *s: torch.rand(*s, device=dev, generator=gen)  # noqa: E731
    p = (r(B) * Np).long()
    direction = (r(B) < 0.5).long() * 2 - 1
    lo = torch.where(direction > 0, span_h, horizon_h)
    hi = torch.where(direction > 0, T - 1 - horizon_h, T - 1 - span_h)
    t0 = lo + (r(B) * (hi - lo + 1)).long()
    perm = r(B, N).argsort(1)
    kidx, qi = perm[:, :K], perm[:, K:K + Q]
    tok, centroid = context(D, p, t0, direction, kidx, span_h)
    qt = t0[:, None] + direction[:, None] * (1 + (r(B, Q) * horizon_h).long())
    ix = (p[:, None], qt, qi)
    th = r(B, Q, 1) * 2 * np.pi
    z1, z2 = torch.randn(2, B, Q, 3, device=dev, generator=gen)
    par, perp = torch.cos(th) * z1 + torch.sin(th) * z2, -torch.sin(th) * z1 + torch.cos(th) * z2
    w = OMEGA / delta_h
    xq, vq = D.X[ix] + sx * par, D.V[ix] + sx * w * perp
    aq = D.A[ix] - sx * w**2 * par
    s = sin_lat(xq)
    xq = torch.cat([xq[..., :2] - centroid[:, None], xq[..., 2:]], -1)
    tau = ((qt - t0[:, None]).float() / TIME_H)[..., None]
    qry = torch.cat([tau, xq / pos, s[..., None], vq / vel], -1)
    return tok, qry, (aq / acc).clamp(-ACC_CLIP, ACC_CLIP), -perp


@torch.no_grad()
def rollout(model, tok, centroid, x0, v0, direction, horizon_h=24, steps_per_h=6):
    """Integrate x' = v, v' = a(tau, x, v | context) from tau = 0 (Heun); direction (B,) +1 forward, -1 backward.

    x0, v0: (B, Q, 3) in km and km/h (x, y, z). Returns hourly positions (horizon_h + 1, B, Q, 3) in km.
    """
    pos, vel, acc = _scale(x0.device)
    z = model.encode(tok)
    dt = direction.float()[:, None, None] / steps_per_h

    def accel(tau, x, v):
        rel = torch.cat([x[..., :2] - centroid[:, None], x[..., 2:]], -1)
        q = torch.cat([tau / TIME_H, rel / pos, sin_lat(x)[..., None], v / vel], -1)
        return model.decode(z, q)[0] * acc

    x, v, out = x0.clone(), v0.clone(), [x0.clone()]
    for s in range(horizon_h * steps_per_h):
        tau = (s * dt).expand(-1, x.shape[1], 1)
        a0 = accel(tau, x, v)
        xp, vp = x + dt * v, v + dt * a0
        x, v = x + 0.5 * dt * (v + vp), v + 0.5 * dt * (a0 + accel(tau + dt, xp, vp))
        if (s + 1) % steps_per_h == 0:
            out.append(x.clone())
    return torch.stack(out)
