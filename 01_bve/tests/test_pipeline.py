import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from geo import great_circle_km, lonlat_to_stereo, stereo_to_lonlat  # noqa: E402
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
