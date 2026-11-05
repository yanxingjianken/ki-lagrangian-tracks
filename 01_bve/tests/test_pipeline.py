import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ki_tracks import OMEGA, great_circle_km, integrate, lonlat_to_stereo, stereo_to_lonlat, training_batch  # noqa: E402
from patches import seed_patches  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_stereo_round_trip():
    lon = np.array([0.0, 45.0, 181.0, 359.0])
    lat = np.array([10.0, 45.0, 70.0, 89.0])
    lo, la = stereo_to_lonlat(lonlat_to_stereo(lon, lat))
    assert np.allclose(lo, lon, atol=1e-9) and np.allclose(la, lat, atol=1e-9)


def test_patches_disks_and_separation():
    lat, lon, pid, centres = seed_patches(n_patches=8, n_per=500, seed=3)
    for k, (clat, clon) in enumerate(centres):
        r = great_circle_km(clon, clat, lon[pid == k], lat[pid == k])
        assert r.max() <= 300.0 + 1e-6
    for i in range(8):
        for j in range(i + 1, 8):
            assert great_circle_km(centres[i, 1], centres[i, 0], centres[j, 1], centres[j, 0]) >= 2000.0


def test_regular_grid_solid_body():
    from stirred_jet import RegularGrid
    from bve_core import SphereBVE
    m = SphereBVE(device=DEV)
    u0 = 20.0
    zs = m.zeta_from_u(torch.as_tensor(u0 * np.cos(m.LAT), device=DEV))
    g = RegularGrid(m)
    uv = g.uv(zs[None])[0].cpu().numpy()
    assert np.abs(uv[0] - u0 * np.cos(np.deg2rad(g.lat))[:, None]).max() < 1e-6 * u0
    assert np.abs(uv[1]).max() < 1e-6 * u0


def test_noise_targets_are_harmonic_about_the_track():
    from types import SimpleNamespace
    T, N = 50, 30
    rng = np.random.default_rng(0)
    d = SimpleNamespace(t=np.linspace(0, 1, T), x=rng.normal(size=(T, N, 2)), v=rng.normal(size=(T, N, 2)),
                        a=rng.normal(size=(T, N, 2)), h=0.0625, sx=1e-3, sv=1e-3 * OMEGA / 0.0625)
    t, x, v, a, q = (z.cpu().numpy() for z in training_batch(d, np.random.default_rng(1), np.arange(N), 256, DEV))
    k = np.abs(d.t[None] - t[:, None]).argmin(1)
    i = np.array([np.argmin(np.abs(d.x[kk] - xx).sum(-1)) for kk, xx in zip(k, x)])
    dx, da = x - d.x[k, i], a - d.a[k, i]
    assert np.allclose(da, -(OMEGA / d.h) ** 2 * dx, atol=1e-2)


class Const(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, t, x, v):
        return self.g.expand_as(x)


@pytest.mark.parametrize("backward", [False, True])
def test_integrate_constant_acceleration_exact(backward):
    g = torch.tensor([[0.3, -0.2]], device=DEV)
    net = torch.nn.ModuleList([Const(g), Const(g)])
    x0 = torch.zeros(4, 2, device=DEV)
    v0 = torch.ones(4, 2, device=DEV)
    t1 = -0.5 if backward else 0.5
    end = integrate(net, None, x0, v0, 0.0, t1, 37)[-1]
    expect = x0 + v0 * t1 + 0.5 * g * t1**2
    assert torch.allclose(end, expect, atol=1e-5)


@pytest.mark.skipif(not (ROOT / "data/tracks_lagranto_w0.npz").exists(), reason="dataset not built")
def test_dataset_velocity_integrates_to_displacement():
    f = np.load(ROOT / "data/tracks_lagranto_w0.npz")
    t, x, v = f["time_h"], f["x_km"][:, ::50], f["v_kmh"][:, ::50]
    disp = np.trapezoid(v, t, axis=0)
    assert np.median(np.linalg.norm(disp - (x[-1] - x[0]), axis=-1)) < 1.0
    assert np.isfinite(f["a_kmh2"]).all()
    assert set(np.unique(f["split"])) == {0, 1, 2}
