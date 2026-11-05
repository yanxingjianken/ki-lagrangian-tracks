"""Nondivergent barotropic vorticity equation on the sphere (torch-harmonics, float64, batched on GPU).

State: spectral absolute-vorticity-free relative vorticity zeta_s with shape (B, lmax, mmax).
Grid: Legendre-Gauss, row 0 = north pole side (lat descending), lon from 0 eastward.
"""
import numpy as np
import torch
import torch_harmonics as th

A_EARTH = 6.37122e6
OMEGA = 7.292e-5
torch.set_default_dtype(torch.float64)


class SphereBVE:
    def __init__(self, nlat=128, nlon=256, dt=600.0, tau_hd=6 * 3600.0, hd_order=4, drag=0.0,
                 device="cuda"):
        self.nlat, self.nlon, self.dt, self.drag = nlat, nlon, dt, drag
        self.lmax = nlon // 3 + 1
        kw = dict(lmax=self.lmax, mmax=self.lmax, grid="legendre-gauss")
        self.sht = th.RealSHT(nlat, nlon, **kw).to(device)
        self.isht = th.InverseRealSHT(nlat, nlon, **kw).to(device)
        self.vsht = th.RealVectorSHT(nlat, nlon, **kw).to(device)
        self.ivsht = th.InverseRealVectorSHT(nlat, nlon, **kw).to(device)
        cost, w = th.quadrature.legendre_gauss_weights(nlat, -1, 1)
        self.lat = -np.arcsin(cost)
        self.lon = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        self.quad_w = np.asarray(w)
        LAT, LON = np.meshgrid(self.lat, self.lon, indexing="ij")
        self.LAT, self.LON = LAT, LON
        t = lambda x: torch.as_tensor(x, device=device)
        self.area = t(np.repeat(self.quad_w[:, None], nlon, 1) * (2 * np.pi / nlon) * A_EARTH**2)
        self.f = t(2 * OMEGA * np.sin(LAT))
        self.coslat = t(np.cos(LAT))
        l = torch.arange(self.lmax, device=device, dtype=torch.float64)[:, None].expand(self.lmax, self.lmax)
        self.lap = -l * (l + 1) / A_EARTH**2
        self.invlap = torch.where(l > 0, -A_EARTH**2 / (l * (l + 1)).clamp(min=1), torch.zeros_like(l))
        lnorm = (l * (l + 1)) / ((self.lmax - 1) * self.lmax)
        self.hd = torch.exp(-dt / tau_hd * lnorm**hd_order)
        self.device = device

    def to_spec(self, grid):
        return self.sht(grid)

    def to_grid(self, spec):
        return self.isht(spec)

    def uv(self, zeta_s):
        vd = torch.stack([zeta_s, torch.zeros_like(zeta_s)], dim=-3)
        return self.ivsht(self.invlap * vd / A_EARTH)

    def vrtdiv(self, uv):
        return self.lap * A_EARTH * self.vsht(uv)

    def zeta_from_u(self, u):
        return self.vrtdiv(torch.stack([u, torch.zeros_like(u)], dim=-3))[..., 0, :, :]

    def tendency(self, zeta_s, forcing_s=None):
        eta = self.to_grid(zeta_s) + self.f
        out = -self.vrtdiv(self.uv(zeta_s) * eta.unsqueeze(-3))[..., 1, :, :]
        if self.drag:
            out = out - self.drag * zeta_s
        if forcing_s is not None:
            out = out + forcing_s
        return out

    def step(self, zeta_s, ref_s=None, forcing_s=None):
        dt, F = self.dt, (lambda z: self.tendency(z, forcing_s))
        k1 = F(zeta_s)
        k2 = F(zeta_s + 0.5 * dt * k1)
        k3 = F(zeta_s + 0.5 * dt * k2)
        k4 = F(zeta_s + dt * k3)
        new = zeta_s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out = self.hd * new if ref_s is None else ref_s + self.hd * (new - ref_s)
        self.last_dissipation = out - new
        return out


def gaussian_jet(lat, u0, lat0_deg, width_deg):
    return u0 * np.exp(-((lat - np.deg2rad(lat0_deg)) / np.deg2rad(width_deg)) ** 2)


def pv_step_jet(model, lat0_deg, width_deg, umax, iters=3):
    """Zonal wind of a Rayleigh–Kuo-stable jet whose absolute vorticity is f plus a tanh step of width width_deg.

    The step height is scaled so that max(u) = umax. Returns u on the model latitudes (nlat,).
    """
    LAT = torch.as_tensor(model.LAT, device=model.device)
    step = 0.5 * torch.tanh((LAT - np.deg2rad(lat0_deg)) / np.deg2rad(width_deg))
    amp = 1e-5
    for _ in range(iters):
        zs = model.to_spec(amp * step)
        u = model.uv(zs[None])[0, 0, :, 0]
        amp *= umax / u.max().item()
    zs = model.to_spec(amp * step)
    return model.uv(zs[None])[0, 0, :, 0].cpu().numpy()


def wave_perturbation(LAT, LON, m, lat0_deg=45.0, width_deg=10.0, lon_center_deg=None, lon_width_deg=None):
    """Vorticity pattern cos(lat) exp(-((lat-lat0)/w)^2) cos(m lon), optionally zonally windowed (packet)."""
    p = np.cos(LAT) * np.exp(-((LAT - np.deg2rad(lat0_deg)) / np.deg2rad(width_deg)) ** 2) * np.cos(m * LON)
    if lon_center_deg is not None:
        dl = np.angle(np.exp(1j * (LON - np.deg2rad(lon_center_deg))))
        p = p * np.exp(-(dl / np.deg2rad(lon_width_deg)) ** 2)
    return p
