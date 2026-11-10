"""Forward and backward ensembles of air-parcel tracks with the conditional Kinematic Interpolant (CPU or GPU).

The model (Jiang, Marzouk, Bose & Pooladian, arXiv:2610.22519, here conditioned on the observed motion of nearby
parcels) predicts the acceleration a and a scaled velocity score s of a parcel from its state and the tracks of the
other parcels of its patch over 48 h. Tracks are integrated with time step dt (hours, negative backward):

    x <- x + dt v,   v <- v + dt a + |dt| (eps^2 / 2) s / sigma_v + eps sqrt(|dt|) N(0, 1),   eps = eps_rel sigma_v,

where sigma_v = s_x pi sqrt(3) / delta is the velocity noise of training (s_x = 5, 5, 0.01 km, delta = 6 h); the score
term and the random kicks give the ensemble its spread (eps_rel chosen so that the spread matches the error at 24 h).
Positions are north-polar stereographic x, y (km) and z = -7 km ln(p / 1000 hPa); time in hours.
"""
from __future__ import annotations

import numpy as np
import torch

from ctx_model import ContextKI

A_KM = 6371.0
SX_KM = (5.0, 5.0, 0.01)
DELTA_H = 6.0
OMEGA = np.pi * np.sqrt(3)


class Model:
    def __init__(self, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.dev = device
        self.net = ContextKI(d=ck["args"]["d_model"]).to(device).eval()
        self.net.load_state_dict(ck["state"])
        sc = ck["scales"]
        t = lambda v: torch.tensor(v, dtype=torch.float32, device=device)  # noqa: E731
        self.pos, self.vel, self.acc, self.time_h = t(sc["pos_km"]), t(sc["vel_kmh"]), t(sc["acc_kmh2"]), float(sc["time_h"])
        self.sigma_v = t(SX_KM) * float(OMEGA / DELTA_H)

    @staticmethod
    def sin_lat(x):
        r = torch.hypot(x[..., 0], x[..., 1])
        return torch.cos(2 * torch.atan(r / (2 * A_KM)))

    def context(self, Xc, Vc, direction, centroid_xy=None):
        """Xc, Vc (17, K, 3): the context parcels every 3 h from the anchor, going back (direction +1, forecast) or
        forward (direction -1, backcast) 48 h; row 0 is the anchor time. centroid_xy: horizontal centroid of the whole
        patch at the anchor (as in training; default: of the context parcels). Returns tokens (1, 17 K, 8), centroid."""
        Xc, Vc = (torch.as_tensor(np.asarray(a, np.float32), device=self.dev) for a in (Xc, Vc))
        n_t, k = Xc.shape[:2]
        tau = (-direction * 3.0 * torch.arange(n_t, device=self.dev) / self.time_h)[:, None, None].expand(n_t, k, 1)
        centroid = (Xc[0, :, :2].mean(0) if centroid_xy is None
                    else torch.as_tensor(np.asarray(centroid_xy, np.float32), device=self.dev))
        rel = torch.cat([Xc[..., :2] - centroid, Xc[..., 2:]], -1)
        tok = torch.cat([tau, rel / self.pos, self.sin_lat(Xc)[..., None], Vc / self.vel], -1)
        return tok.reshape(1, n_t * k, 8), centroid[None]

    def _heads(self, z, centroid, tau, x, v):
        rel = torch.cat([x[..., :2] - centroid[:, None], x[..., 2:]], -1)
        q = torch.cat([tau / self.time_h, rel / self.pos, self.sin_lat(x)[..., None], v / self.vel], -1)
        a, s = self.net.decode(z, q)
        return a * self.acc, s

    @torch.no_grad()
    def ensemble(self, tok, centroid, x0, v0, direction=1, members=16, eps_rel=1.5, hours=24, steps_per_h=6, seed=0):
        """x0, v0 (Q, 3) -> hourly positions (hours + 1, members, Q, 3); direction +1 forward, -1 backward."""
        g = torch.Generator(device=self.dev).manual_seed(seed)
        x0, v0 = (torch.as_tensor(np.asarray(a, np.float32), device=self.dev) for a in (x0, v0))
        q = x0.shape[0]
        z = self.net.encode(tok)
        x = x0[None].expand(members, q, 3).reshape(1, members * q, 3).clone()
        v = v0[None].expand(members, q, 3).reshape(1, members * q, 3).clone()
        eps = eps_rel * self.sigma_v
        dt = direction / steps_per_h
        out = [x.clone()]
        for k in range(hours * steps_per_h):
            tau = torch.full_like(x[..., :1], k * dt)
            a, s = self._heads(z, centroid, tau, x, v)
            noise = torch.randn(x.shape, device=self.dev, generator=g)
            x, v = x + dt * v, v + dt * a + abs(dt) * 0.5 * eps**2 * s / self.sigma_v + eps * np.sqrt(abs(dt)) * noise
            if (k + 1) % steps_per_h == 0:
                out.append(x.clone())
        return torch.stack(out).reshape(hours + 1, members, q, 3).cpu().numpy()


def stereo_to_lonlat(x):
    r = np.hypot(x[..., 0], x[..., 1])
    lat = 90.0 - np.rad2deg(2 * np.arctan(r / (2 * A_KM)))
    lon = np.rad2deg(np.arctan2(x[..., 0], -x[..., 1])) % 360.0
    return lon, lat


def lonlat_to_stereo(lon, lat):
    r = 2 * A_KM * np.tan(np.deg2rad(90.0 - np.asarray(lat)) / 2)
    lam = np.deg2rad(lon)
    return np.stack([r * np.sin(lam), -r * np.cos(lam)], axis=-1)
