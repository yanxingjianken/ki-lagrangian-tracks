"""Forecasts of parcel tracks with the conditional Kinematic Interpolant trained on ERA5 (roger/03_era5).

Load the checkpoint (its per-axis scales travel with it), build context tokens from the observed tracks of a patch,
and integrate target parcels forward either deterministically (x' = v, v' = a, Heun) or as a stochastic ensemble:

    x' = v,   dv = [a + (eps^2 / 2) s / sigma_v] dt + eps dW,   eps = eps_rel * sigma_v,

where a is the predicted acceleration, s the scaled velocity-score head (s = sigma_v * grad_v log p) and
sigma_v = s_x * pi * sqrt(3) / delta the velocity noise of the training interpolant (s_x = 5, 5, 0.01 km; delta = 6 h).
eps_rel (per sqrt(hour)) sets both the random kicks and the pull of the score term (relaxation rate eps_rel^2 / 2
per hour). Ensemble members are stacked as extra queries: the decoder's queries attend only to the context latents,
so one encoding per patch serves every member.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ctx_model import ContextKI

A_KM = 6371.0
SX_KM = (5.0, 5.0, 0.01)     # training noise in position (03_era5/scripts/ctx_data.py, sample_batch)
DELTA_H = 6.0                # training noise time scale
OMEGA = np.pi * np.sqrt(3)
TRACKS = Path(__file__).resolve().parents[2] / "03_era5/data/tracks"
MODEL = Path(__file__).resolve().parents[2] / "03_era5/outputs/models/context_ki_era5.pt"


class Forecaster:
    def __init__(self, path=MODEL, device="cuda"):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.args, self.dev = ck["args"], device
        self.model = ContextKI(d=self.args["d_model"]).to(device).eval()
        self.model.load_state_dict(ck["state"])
        sc = ck["scales"]
        t = lambda v: torch.tensor(v, dtype=torch.float32, device=device)  # noqa: E731
        self.pos, self.vel, self.acc, self.time_h = t(sc["pos_km"]), t(sc["vel_kmh"]), t(sc["acc_kmh2"]), float(sc["time_h"])
        self.sigma_v = t(SX_KM) * float(OMEGA / DELTA_H)

    @staticmethod
    def sin_lat(x):
        r = torch.hypot(x[..., 0], x[..., 1])
        return torch.cos(2 * torch.atan(r / (2 * A_KM)))

    def context(self, X, V, t0, kidx, span_h=48, step_h=3):
        """Tokens of the observed parcels kidx (B, K) over [t0 - span, t0]; X, V (B, T, N, 3) on the device."""
        B = X.shape[0]
        offs = torch.arange(0, span_h + 1, step_h, device=self.dev)
        tc = t0[:, None] - offs[None]
        b = torch.arange(B, device=self.dev)
        ix = (b[:, None, None], tc[:, :, None], kidx[:, None, :])
        centroid = X[b, t0, :, :2].mean(1)
        x = X[ix]
        s = self.sin_lat(x)
        x = torch.cat([x[..., :2] - centroid[:, None, None], x[..., 2:]], -1)
        tau = ((tc - t0[:, None]).float() / self.time_h)[:, :, None, None].expand(-1, -1, kidx.shape[1], 1)
        tok = torch.cat([tau, x / self.pos, s[..., None], V[ix] / self.vel], -1)
        return tok.flatten(1, 2), centroid

    def _heads(self, z, centroid, tau, x, v):
        rel = torch.cat([x[..., :2] - centroid[:, None], x[..., 2:]], -1)
        q = torch.cat([tau / self.time_h, rel / self.pos, self.sin_lat(x)[..., None], v / self.vel], -1)
        a, s = self.model.decode(z, q)
        return a * self.acc, s

    @torch.no_grad()
    def ode(self, tok, centroid, x0, v0, horizon_h=24, steps_per_h=6):
        """Deterministic forward tracks (Heun): hourly positions (H + 1, B, Q, 3)."""
        z = self.model.encode(tok)
        dt = 1.0 / steps_per_h
        x, v, out = x0.clone(), v0.clone(), [x0.clone()]
        for k in range(horizon_h * steps_per_h):
            tau = torch.full_like(x[..., :1], k * dt)
            a0, _ = self._heads(z, centroid, tau, x, v)
            xp, vp = x + dt * v, v + dt * a0
            a1, _ = self._heads(z, centroid, tau + dt, xp, vp)
            x, v = x + 0.5 * dt * (v + vp), v + 0.5 * dt * (a0 + a1)
            if (k + 1) % steps_per_h == 0:
                out.append(x.clone())
        return torch.stack(out)

    @torch.no_grad()
    def ensemble(self, tok, centroid, x0, v0, members=16, eps_rel=1.0, horizon_h=24, steps_per_h=6, gen=None):
        """Stochastic forward tracks (Euler-Maruyama): hourly positions (H + 1, B, M, Q, 3)."""
        B, Q, _ = x0.shape
        z = self.model.encode(tok)
        x = x0[:, None].expand(B, members, Q, 3).reshape(B, members * Q, 3).clone()
        v = v0[:, None].expand(B, members, Q, 3).reshape(B, members * Q, 3).clone()
        eps = torch.as_tensor(eps_rel, dtype=torch.float32, device=self.dev) * self.sigma_v
        dt = 1.0 / steps_per_h
        out = [x.clone()]
        for k in range(horizon_h * steps_per_h):
            tau = torch.full_like(x[..., :1], k * dt)
            a, s = self._heads(z, centroid, tau, x, v)
            drift = a + 0.5 * eps**2 * s / self.sigma_v
            noise = torch.randn(x.shape, device=self.dev, generator=gen)
            x, v = x + dt * v, v + dt * drift + eps * np.sqrt(dt) * noise
            if (k + 1) % steps_per_h == 0:
                out.append(x.clone())
        return torch.stack(out).reshape(horizon_h + 1, B, members, Q, 3)


def load_patches(names, ids, levels=None, n_per=480, device="cuda", seed=0, max_patches=None):
    """Patches (release, level, patch id) of the 03_era5 datasets: X, V (Np, T, n_per, 3) on the device, meta, lat, p."""
    rng = np.random.default_rng(seed)
    plan = []
    for name in names:
        with np.load(TRACKS / f"tracks_{name}.npz") as f:
            level, patch = f["level"], f["patch"]
        for k in (np.unique(level) if levels is None else levels):
            for q in ids:
                if np.any((level == k) & (patch == q)):
                    plan.append((name, int(k), int(q)))
    if max_patches is not None and len(plan) > max_patches:
        plan = [plan[i] for i in sorted(rng.choice(len(plan), max_patches, replace=False))]
    X, V, LAT, P, meta = [], [], [], [], []
    for name in dict.fromkeys(n for n, _, _ in plan):
        with np.load(TRACKS / f"tracks_{name}.npz") as f:
            level, patch, x, v, lat, p, hemi = (f[k] for k in ("level", "patch", "x_km", "v_kmh", "lat", "p_hpa", "hemisphere"))
            rtime = str(f["release_time"])
        for n, k, q in plan:
            if n != name:
                continue
            idx = np.flatnonzero((level == k) & (patch == q))
            idx = idx[:n_per] if len(idx) >= n_per else np.r_[idx, rng.choice(idx, n_per - len(idx))]
            X.append(x[:, idx]); V.append(v[:, idx]); LAT.append(lat[:, idx]); P.append(p[:, idx])
            meta.append(dict(release=name, release_time=rtime, level=k, patch=q, hemisphere=int(hemi[idx[0]])))
    to = lambda a: torch.from_numpy(np.stack(a)).to(device)  # noqa: E731
    return to(X), to(V), np.stack(LAT), np.stack(P), meta
