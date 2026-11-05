# Adapted from roger/01_bve/scripts/ctx_data.py (three-dimensional state, start levels).
"""3-D patches of hourly tracks on the GPU, minibatches for the conditional KI, and conditional forecast/backcast rollouts.

A sample is one patch (flow, window, start level, patch), an anchor time t0 and a direction: forecast (context = the
patch's observed parcels over [t0 - P, t0], targets after t0) or backcast (context over [t0, t0 + P], targets before).
Horizontal positions are taken relative to the patch centroid at t0; height z is absolute.
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


def flow_names(config, split):
    return [f["name"] for f in read_flows(config) if f["split"] == split]


def read_flows(config):
    """Usable flows of the configuration (a list, or a dict with key "flows"); status must be "ok" if present."""
    flows = json.loads(Path(config).read_text())
    flows = flows["flows"] if isinstance(flows, dict) else flows
    return [f for f in flows if f.get("status", "ok") == "ok"]


def load_patches(root, names, windows, levels, patch_ids, n_per=1000, device="cuda", seed=0):
    """Stack patches of data/tracks_<flow>_w<k>_l<level>.npz into (Np, T, n_per, ...) GPU tensors."""
    rng = np.random.default_rng(seed)
    X, V, A, S, P, meta = [], [], [], [], [], []
    for name in names:
        for w in windows:
            for lev in levels:
                with np.load(Path(root) / f"data/tracks_{name}_w{w}_l{lev}.npz") as f:
                    x, v, a, lat, p, patch = f["x_km"], f["v_kmh"], f["a_kmh2"], f["lat"], f["p_hpa"], f["patch"]
                    t = f["time_h"]
                for q in patch_ids:
                    idx = np.where(patch == q)[0]
                    if len(idx) == 0:
                        continue
                    idx = idx if len(idx) == n_per else rng.choice(idx, n_per, replace=len(idx) < n_per)
                    X.append(x[:, idx])
                    V.append(v[:, idx])
                    A.append(a[:, idx])
                    S.append(np.sin(np.deg2rad(lat[:, idx])))
                    P.append(p[:, idx])
                    meta.append((name, w, lev, int(q)))
    to = lambda z: torch.tensor(np.stack(z), dtype=torch.float32, device=device)
    return SimpleNamespace(X=to(X), V=to(V), A=to(A), S=to(S), P=to(P), meta=meta, time_h=t)


def _scale(dev):
    return tuple(torch.tensor(s, dtype=torch.float32, device=dev) for s in (POS, VEL, ACC))


def context(D, p, t0, direction, kidx, span_h=48, step_h=3):
    """Context tokens (B, Tc*K, 8) and the horizontal patch centroid at t0 (B, 2)."""
    pos, vel, _ = _scale(D.X.device)
    offs = torch.arange(0, span_h + 1, step_h, device=D.X.device)
    tc = t0[:, None] - direction[:, None] * offs[None]
    ix = (p[:, None, None], tc[:, :, None], kidx[:, None, :])
    centroid = D.X[p, t0, :, :2].mean(1)
    x = D.X[ix]
    x = torch.cat([x[..., :2] - centroid[:, None, None], x[..., 2:]], -1)
    tau = ((tc - t0[:, None]).float() / TIME_H)[:, :, None, None].expand(-1, -1, kidx.shape[1], 1)
    tok = torch.cat([tau, x / pos, D.S[ix][..., None], D.V[ix] / vel], -1)
    return tok.flatten(1, 2), centroid


def sample_batch(D, B, K=192, Q=256, span_h=48, horizon_h=24, sx_km=(5.0, 5.0, 0.01), delta_h=6.0, gen=None):
    """One minibatch: context tokens, query states around true tracks, target accelerations and scores."""
    dev, (Np, T, N) = D.X.device, D.X.shape[:3]
    pos, vel, acc = _scale(dev)
    sx = torch.tensor(sx_km, dtype=torch.float32, device=dev)
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
    z1, z2 = torch.randn(2, B, Q, 3, device=dev, generator=gen)
    par, perp = torch.cos(th) * z1 + torch.sin(th) * z2, -torch.sin(th) * z1 + torch.cos(th) * z2
    w = OMEGA / delta_h
    xq, vq = D.X[ix] + sx * par, D.V[ix] + sx * w * perp
    aq = D.A[ix] - sx * w**2 * par
    xq = torch.cat([xq[..., :2] - centroid[:, None], xq[..., 2:]], -1)
    tau = ((qt - t0[:, None]).float() / TIME_H)[..., None]
    qry = torch.cat([tau, xq / pos, D.S[ix][..., None], vq / vel], -1)
    return tok, qry, (aq / acc).clamp(-ACC_CLIP, ACC_CLIP), -perp


def sin_lat(x_km):
    r = torch.hypot(x_km[..., 0], x_km[..., 1])
    return torch.cos(2 * torch.atan(r / (2 * A_KM)))


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
