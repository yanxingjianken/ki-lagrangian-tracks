"""Checks of the omega integral, the spectral mass-flux divergence, the regridding and the written files."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("omega_regrid", ROOT / "scripts" / "04_omega_regrid.py")
om = importlib.util.module_from_spec(spec)
spec.loader.exec_module(om)
from common import A_SI, LAT, LON, SIGMA  # noqa: E402

NH = LAT >= 0
PHI, LAM = np.meshgrid(np.deg2rad(LAT[NH]), np.deg2rad(LON), indexing="ij")


def test_integral_weights():
    w = om.integral_weights(om.EDGES, SIGMA)
    assert np.allclose(w.sum(1), SIGMA)
    assert np.allclose(w[-1, :-1], np.diff(om.EDGES)[:-1]) and np.isclose(w[-1, -1], SIGMA[-1] - om.EDGES[-2])


def test_mass_flux_divergence_matches_analytic():
    """u = 0, v = V0 cos(phi) sin(phi), ps uniform: div(ps V) = ps V0 (cos^2 - 2 sin^2) / a; omega from the integral."""
    v0, ps0 = 10.0, 1.0e5
    v = np.broadcast_to(v0 * np.cos(PHI) * np.sin(PHI), (15,) + PHI.shape)
    omega, col = om.omega_sigma(np.zeros_like(v), v, np.full(PHI.shape, ps0))
    div = ps0 * v0 * (np.cos(PHI) ** 2 - 2 * np.sin(PHI) ** 2) / A_SI
    assert np.allclose(col, div, atol=1e-9 * ps0)
    assert np.allclose(omega, -SIGMA[:, None, None] * div[None], atol=1e-9 * ps0)


def test_advection_term_with_zonal_ps_wave():
    """Non-divergent u = U cos(phi), ps = p0 + dp cos(phi)^4 cos(4 lambda): div(ps V) = V.grad(ps), constant in
    sigma, so omega = sigma V.grad(ps) - sigma div(ps V) = 0 (the two spectral paths must agree)."""
    u0, p0, dp = 20.0, 1.0e5, 500.0
    u = np.broadcast_to(u0 * np.cos(PHI), (15,) + PHI.shape)
    ps = p0 + dp * np.cos(PHI) ** 4 * np.cos(4 * LAM)
    omega, col = om.omega_sigma(u, np.zeros_like(u), ps)
    adv = u0 * np.cos(PHI) * (-4 * dp * np.cos(PHI) ** 4 * np.sin(4 * LAM)) / (A_SI * np.cos(PHI))
    assert np.allclose(col, adv, atol=1e-6 * np.abs(adv).max())
    assert np.abs(omega).max() < 1e-6 * np.abs(adv).max()


def test_regrid_is_exact_for_smooth_fields():
    f = lambda la, lo: np.cos(np.deg2rad(la)) ** 2 * np.cos(5 * np.deg2rad(lo) + 0.3) + np.sin(np.deg2rad(la)) ** 2
    out = om.regrid(f(PHI * 180 / np.pi, LAM * 180 / np.pi), LAT[NH], 1.0, vector=False)
    la, lo = np.meshgrid(om.LAT_OUT, om.LON_OUT, indexing="ij")
    assert out.shape == (91, 360)
    assert np.abs(out[:-1] - f(la, lo)[:-1]).max() < 1e-4
    assert np.allclose(out[-1], np.sin(np.deg2rad(LAT[-1])) ** 2)


def test_regrid_odd_field_vanishes_at_equator_and_vector_pole():
    v = np.sin(PHI) * np.cos(PHI) * np.cos(LAM)
    out = om.regrid(v, LAT[NH], -1.0, vector=True)
    assert np.abs(out[0]).max() < 1e-12
    assert np.allclose(out[-1], np.sin(PHI[-1, 0]) * np.cos(PHI[-1, 0]) * np.cos(np.deg2rad(om.LON_OUT)))


FLOW_FILES = sorted((ROOT / "data").glob("dino_*.nc"))


@pytest.mark.parametrize("path", FLOW_FILES[:3] + FLOW_FILES[-2:], ids=lambda p: p.stem)
def test_flow_file_layout(path):
    ds = xr.open_dataset(path)
    assert ds.u.dims == ("time", "level", "lat", "lon") and ds.ps.dims == ("time", "lat", "lon")
    assert ds.sizes == {"time": 361, "level": 15, "lat": 91, "lon": 360}
    assert np.allclose(ds.level, SIGMA) and np.allclose(ds.lat, np.arange(91)) and np.allclose(ds.lon, np.arange(360))
    assert np.array_equal(ds.time, np.arange(361)) and str(ds.datetime.values[0])[:13] == "2000-01-01T00"
    units = dict(u="m s-1", v="m s-1", omega="Pa s-1", t="K", ps="Pa")
    for k, un in units.items():
        assert ds[k].dtype == np.float32 and ds[k].attrs["units"] == un
        assert ds[k].encoding["chunksizes"][0] == 1
    cfg = next(f for f in json.loads((ROOT / "configs" / "flows.json").read_text()) if f["name"] == ds.attrs["name"])
    assert ds.attrs["m"] == cfg["m"] and ds.attrs["ujet"] == cfg["ujet"] and ds.attrs["amp"] == cfg["amp"]
    assert 8.0e4 < float(ds.ps.min()) and float(ds.ps.max()) < 1.15e5


@pytest.mark.parametrize("path", sorted((ROOT / "data").glob("check_*.json")), ids=lambda p: p.stem)
def test_continuity(path):
    """-int div(ps V) dsigma against centred hourly ps differences, 30-60N. The reference is the weak part: ps carries
    oscillations of a few hours that hourly sampling aliases, largest relative to the eddy signal in the polar-cap runs
    and in weak life cycles (polar cap, hour 168: 2-h and 20-min centred differences differ by ~13 %, while the omega
    column matches the model's own instantaneous tendency to 0.2 %)."""
    c = json.loads(path.read_text())
    if c["status"] != "ok":
        pytest.skip(c["status"])
    k = c["continuity_30_60N_day2_on"]
    assert k["corr"] > 0.98 and k["rel_rms"] < 0.25
    assert c["equator_v_over_max_v"] < 1e-5
