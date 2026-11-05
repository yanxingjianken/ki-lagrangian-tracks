"""Light parcel tracker on the regular grid (bilinear, Heun) and Lagrangian scale statistics; used to choose the flow regime."""
import numpy as np
import torch
import torch.nn.functional as F

A_KM = 6371.0


def seed_equal_area(n, lat_min, lat_max, seed=0):
    """Random parcels uniform in area between two latitudes [deg]; returns (lat, lon) in degrees."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(np.sin(np.deg2rad(lat_min)), np.sin(np.deg2rad(lat_max)), n)
    return np.rad2deg(np.arcsin(s)), rng.uniform(0.0, 360.0, n)


def sample(uv, lat, lon):
    """Bilinear sample of uv (B, 2, nlat, nlon) on a 90N..90S, 0..360 grid at (B, N) parcel positions [deg]."""
    padded = torch.cat([uv, uv[..., :1]], dim=-1)
    x = (lon % 360.0) / 180.0 - 1.0
    y = (90.0 - lat) / 90.0 - 1.0
    grid = torch.stack([x, y], dim=-1)[:, None, :, :]
    return F.grid_sample(padded, grid, mode="bilinear", align_corners=True)[..., 0, :]


def cartesian(lat, lon):
    la, lo = torch.deg2rad(lat), torch.deg2rad(lon)
    return torch.stack([torch.cos(la) * torch.cos(lo), torch.cos(la) * torch.sin(lo), torch.sin(la)], dim=-1)


def lat_lon(r):
    r = r / r.norm(dim=-1, keepdim=True)
    return torch.rad2deg(torch.asin(r[..., 2].clamp(-1, 1))), torch.rad2deg(torch.atan2(r[..., 1], r[..., 0])) % 360.0


def rate(uv, r):
    lat, lon = lat_lon(r)
    u, v = sample(uv, lat, lon).unbind(1)
    la, lo = torch.deg2rad(lat), torch.deg2rad(lon)
    e_lon = torch.stack([-torch.sin(lo), torch.cos(lo), torch.zeros_like(lo)], -1)
    e_lat = torch.stack([-torch.sin(la) * torch.cos(lo), -torch.sin(la) * torch.sin(lo), torch.cos(la)], -1)
    return (u[..., None] * e_lon + v[..., None] * e_lat) / (A_KM * 1e3)


def heun(uv0, uv1, r, dt):
    """One Heun step of unit vectors r (B, N, 3) with the wind at the step start (uv0) and end (uv1)."""
    k1 = rate(uv0, r)
    rp = r + dt * k1
    rp = rp / rp.norm(dim=-1, keepdim=True)
    out = r + 0.5 * dt * (k1 + rate(uv1, rp))
    return out / out.norm(dim=-1, keepdim=True)


def lagrangian_scales(r, dt_h):
    """Median speed [m/s] and turning time |v| / |a_tangential| [h] from unit-vector tracks r (T, ..., 3)."""
    x = A_KM * r
    v = np.gradient(x, dt_h, axis=0)
    a = np.gradient(v, dt_h, axis=0)
    a_t = a - (a * r).sum(-1, keepdims=True) * r
    speed = np.linalg.norm(v, axis=-1)
    turn = speed / np.maximum(np.linalg.norm(a_t, axis=-1), 1e-12)
    return speed / 3.6, turn
