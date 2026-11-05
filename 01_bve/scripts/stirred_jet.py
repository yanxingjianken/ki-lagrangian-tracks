"""Stochastically stirred midlatitude jet on the sphere: BVE + relaxation to a zonal jet + red-noise vorticity stirring.

d(zeta)/dt = -J(psi, zeta + f) - (zeta - zeta_jet)/tau_r + A * S(t) - hyperdiffusion, with S an AR(1) process of
unit rms, band-limited to total wavenumbers l1..l2 and confined to a Gaussian latitude band (northern hemisphere).
"""
import numpy as np
import torch
import torch_harmonics as th

from bve_core import A_EARTH, SphereBVE, gaussian_jet


class StirredJet:
    """Batched stirred-jet integrator; one batch member per stirring amplitude.

    Args:
        amps: stirring amplitudes A [s^-2], one per batch member.
        u0, lat0, width: relaxation jet u = u0 exp(-((lat - lat0)/width)^2) [m/s, deg, deg].
        tau_r_days: relaxation time toward the jet vorticity.
        tau_s_days: decorrelation time of the stirring.
        l_band: inclusive band of total wavenumbers that is stirred.
        stir_lat, stir_width: centre and e-folding half-width of the stirring band [deg].
        nlat, nlon, dt: model Gaussian grid and time step [s].
        seed: random seed of the stirring.
        device: torch device.
    """

    def __init__(self, amps, u0=40.0, lat0=45.0, width=10.0, tau_r_days=10.0, tau_s_days=1.0, l_band=(8, 14),
                 stir_lat=45.0, stir_width=12.0, nlat=128, nlon=256, dt=600.0, seed=0, device="cuda"):
        self.m = m = SphereBVE(nlat, nlon, dt=dt, device=device)
        self.device, self.B = device, len(amps)
        u = torch.as_tensor(gaussian_jet(m.LAT, u0, lat0, width), device=device)
        self.zjet = m.zeta_from_u(u)[None]
        self.amps = torch.as_tensor(np.asarray(amps, float), device=device)[:, None, None]
        self.tau_r = tau_r_days * 86400.0
        self.ar = float(np.exp(-dt / (tau_s_days * 86400.0)))
        l = torch.arange(m.lmax, device=device)[:, None].expand(m.lmax, m.lmax)
        mm = torch.arange(m.lmax, device=device)[None, :].expand(m.lmax, m.lmax)
        self.band = ((l >= l_band[0]) & (l <= l_band[1]) & (mm <= l)).to(torch.float64)
        mask = np.exp(-((m.LAT - np.deg2rad(stir_lat)) / np.deg2rad(stir_width)) ** 2)
        self.mask = torch.as_tensor(mask, device=device)
        self.inside = torch.as_tensor(mask > 0.5, device=device)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.stir = self._white()
        self.zeta = self.zjet.repeat(self.B, 1, 1) + 1e-7 * self._white()

    def _white(self):
        m = self.m
        shape = (self.B, m.lmax, m.lmax)
        kw = dict(generator=self.gen, device=self.device, dtype=torch.float64)
        coef = torch.complex(torch.randn(shape, **kw), torch.randn(shape, **kw)) * self.band
        grid = m.to_grid(coef) * self.mask
        rms = grid[:, self.inside].pow(2).mean(-1).sqrt()[:, None, None]
        return m.to_spec(grid / rms)

    def step(self):
        """Advance every member by one time step."""
        self.stir = self.ar * self.stir + np.sqrt(1.0 - self.ar**2) * self._white()
        forcing = -(self.zeta - self.zjet) / self.tau_r + self.amps * self.stir
        self.zeta = self.m.step(self.zeta, forcing_s=forcing)

    def run(self, n_steps):
        for _ in range(n_steps):
            self.step()


class RegularGrid:
    """Exact evaluation of the spectral wind on a regular lon-lat grid (lat 90N..90S, lon 0..360-dlon)."""

    def __init__(self, m, nlat=181, nlon=360):
        self.m = m
        self.ivsht = th.InverseRealVectorSHT(nlat, nlon, lmax=m.lmax, mmax=m.lmax, grid="equiangular").to(m.device)
        self.isht = th.InverseRealSHT(nlat, nlon, lmax=m.lmax, mmax=m.lmax, grid="equiangular").to(m.device)
        self.lat = np.linspace(90.0, -90.0, nlat)
        self.lon = np.arange(nlon) * 360.0 / nlon

    def uv(self, zeta_s):
        """(B, 2, nlat, nlon) eastward and northward wind [m/s]."""
        vd = torch.stack([zeta_s, torch.zeros_like(zeta_s)], dim=-3)
        return self.ivsht(self.m.invlap * vd / A_EARTH)

    def zeta(self, zeta_s):
        """(B, nlat, nlon) relative vorticity [1/s]."""
        return self.isht(zeta_s)
