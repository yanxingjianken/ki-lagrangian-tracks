import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bve_core import A_EARTH, OMEGA, SphereBVE, gaussian_jet  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def m():
    return SphereBVE(device=DEV)


def test_grid_orientation(m):
    assert m.lat[0] > 0 > m.lat[-1]
    assert np.all(np.diff(m.lat) < 0)


def test_solid_body_vorticity_and_inverse(m):
    u0 = 20.0
    u = torch.as_tensor(u0 * np.cos(m.LAT), device=DEV)
    zs = m.zeta_from_u(u)
    zeta = m.to_grid(zs).cpu().numpy()
    expect = 2 * u0 * np.sin(m.LAT) / A_EARTH
    assert np.abs(zeta - expect).max() < 1e-6 * np.abs(expect).max()
    uv = m.uv(zs).cpu().numpy()
    assert np.abs(uv[0] - u0 * np.cos(m.LAT)).max() < 1e-6 * u0
    assert np.abs(uv[1]).max() < 1e-6 * u0


def test_zonal_jet_is_steady(m):
    u = torch.as_tensor(gaussian_jet(m.LAT, 25.0, 45.0, 15.0), device=DEV)
    zs = m.zeta_from_u(u)
    tend = m.to_grid(m.tendency(zs[None]))[0].abs().max().item()
    assert tend < 1e-16


def test_rossby_haurwitz_translation(m):
    """Williamson et al. (1992) case-6 RH wave (R=4) translates at nu = (R(3+R)w - 2*Omega)/((1+R)(2+R))."""
    w = K = 7.848e-6
    R = 4
    mu = np.sin(m.LAT)

    def zeta(lon):
        return 2 * w * mu - K * mu * np.cos(m.LAT) ** R * (R**2 + 3 * R + 2) * np.cos(R * lon)

    zs = m.to_spec(torch.as_tensor(zeta(m.LON), device=DEV))[None]
    nsteps = int(86400 / m.dt)
    for _ in range(nsteps):
        zs = m.step(zs)
    nu = (R * (3 + R) * w - 2 * OMEGA) / ((1 + R) * (2 + R))
    expect = zeta(m.LON - nu * nsteps * m.dt)
    got = m.to_grid(zs)[0].cpu().numpy()
    wave = expect - 2 * w * mu
    assert np.abs(got - expect).max() < 2e-3 * np.abs(wave).max()


def test_angular_momentum_conserved_with_hyperdiffusion(m):
    u = gaussian_jet(m.LAT, 25.0, 45.0, 15.0)
    pert = 3e-5 * np.cos(m.LAT) * np.exp(-((m.LAT - np.deg2rad(45)) / np.deg2rad(10)) ** 2) * np.cos(6 * m.LON)
    zs0 = m.zeta_from_u(torch.as_tensor(u, device=DEV))
    zs = (zs0 + m.to_spec(torch.as_tensor(pert, device=DEV)))[None]
    def am(z):
        uu = m.uv(z)[0, 0]
        return (uu * m.coslat * m.area).sum().item()
    a0 = am(zs)
    for _ in range(int(3 * 86400 / m.dt)):
        zs = m.step(zs, ref_s=zs0)
    assert abs(am(zs) - a0) < 1e-6 * abs(a0)
