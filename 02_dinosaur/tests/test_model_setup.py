"""Checks of the model set-up, the wavenumber projection and the flow set."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import COORDS, GRID, L_AXIS, M_AXIS, SIGMA, SIGMA_FULL, SPECS, U, nd, project_wave  # noqa: E402


def load(name: str):
    spec = importlib.util.spec_from_file_location(name[:-3].lstrip("0123456789_"), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FLOWS = load("00_flows.py")


def test_levels_are_thm93():
    assert len(SIGMA) == 15 and np.max(np.abs(SIGMA - SIGMA_FULL)) < 2e-3
    edges = np.asarray(COORDS.vertical.boundaries)
    assert edges[0] == 0.0 and edges[-1] == 1.0 and np.allclose((edges[1:] + edges[:-1]) / 2, SIGMA)


def test_grid_is_t95():
    assert GRID.nodal_shape == (289, 145) and GRID.modal_shape == (191, 97)
    assert abs(SPECS.dimensionalize(SPECS.radius, U.m).magnitude - 6.37122e6) < 1.0


def test_diffusion_is_one_hour_at_n95():
    eig = -GRID.laplacian_eigenvalues
    scale = nd(600.0, U.s) / (nd(3600.0, U.s) * (95.0 * 96.0) ** 3)
    assert np.isclose(np.exp(-6 * scale * eig[95] ** 3), np.exp(-1.0))


@pytest.mark.parametrize("m", [3, 4, 6, 9])
def test_projection_keeps_wavenumber_m_symmetric(m):
    from dinosaur import primitive_equations as pe
    rng = np.random.default_rng(m)
    shape = (15,) + GRID.modal_shape
    x = pe.State(*(rng.normal(size=s) for s in (shape, shape, shape, (1,) + GRID.modal_shape)))
    y = project_wave(x, m)
    keep = np.broadcast_to(np.abs(M_AXIS)[:, None] == m, GRID.modal_shape)
    even = np.broadcast_to(((L_AXIS[None, :] - m) % 2) == 0, GRID.modal_shape)
    lnps = np.asarray(y.log_surface_pressure)[0]
    assert np.all(lnps[~keep] == 0) and np.all(lnps[keep & ~even] == 0) and np.any(lnps[keep & even] != 0)
    vor = np.asarray(y.vorticity)
    assert np.all(vor[:, ~keep] == 0) and np.all(vor[:, keep & even] == 0)


def test_flow_set():
    flows = json.loads((ROOT / "configs" / "flows.json").read_text())
    fresh = FLOWS.draw_flows()
    assert [f["name"] for f in flows] == [f["name"] for f in fresh]
    assert len({f["name"] for f in flows}) == 56
    train = [f for f in flows if f["split"] == "train"]
    test = [f for f in flows if f["split"] == "test"]
    assert len(train) == 48 and len(test) == 8
    typ = lambda f: f["basic"].split("_")[0]
    for key, levels in (("ujet", FLOWS.UJETS), ("m", FLOWS.MS), ("amp", FLOWS.AMPS)):
        c = Counter(f[key] for f in train)
        assert all(c[lv] >= 5 for lv in levels), (key, c)
    assert all(Counter(typ(f) for f in train)[t] >= 5 for t in FLOWS.TYPES)
    inside = [f for f in test if f["label"].startswith("new combination")]
    seen = {(f["ujet"], typ(f), f["m"]) for f in train}
    assert len(inside) == 4 and all((f["ujet"], typ(f), f["m"]) not in seen and f["amp"] == 1.0 for f in inside)
    assert {(typ(f), f["ujet"], f["m"]) for f in test if f not in inside} == {
        ("lc1", 30.0, 6), ("lc2lobes", 65.0, 5), ("lc2polarcap", 45.0, 9), ("lc1", 65.0, 3)}
    for f in flows:
        assert {"name", "case", "z2", "ujet", "m", "amp", "split", "label", "status"} <= set(f)
        assert (f["case"], f["z2"]) == FLOWS.CASE[typ(f)]


def _basic_states():
    import xarray as xr
    for path in sorted((ROOT / "data").glob("basic_*.nc")):
        strong = xr.open_dataset(path).attrs["ujet"] >= 50
        mark = pytest.mark.xfail(strong, reason="thermal-wind temperature of jets >= 50 m/s is statically unstable "
                                                "in the tropical upper troposphere (README, Basic states)", strict=True)
        yield pytest.param(path, id=path.stem, marks=mark)


@pytest.mark.parametrize("path", list(_basic_states()))
def test_basic_state(path):
    import xarray as xr
    ds = xr.open_dataset(path)
    p = ds.level.values[:, None] * np.exp(ds.lnps.values)[None]
    theta = ds.t.values * (1e5 / p) ** (2 / 7)
    assert np.all(theta[:-1] > theta[1:]), "statically unstable"
    nh = ds.lat.values > 0
    u = ds.u.values[:, nh]
    if ds.attrs["case"] == "lc1":
        assert abs(u.max() - ds.attrs["ujet"]) < 0.5 and abs(ds.lat.values[nh][np.argmax(u.max(0))] - 45) < 1.5


@pytest.mark.parametrize("path", sorted((ROOT / "data").glob("mode_*.npz")), ids=lambda p: p.stem)
def test_mode_is_symmetric_wavenumber_m(path):
    import pandas as pd
    m = int(pd.read_csv(path.with_suffix(".csv")).m.iloc[0])
    mode = np.load(path)
    shape = (len(M_AXIS), len(L_AXIS))
    off = np.broadcast_to(np.abs(M_AXIS)[:, None] != m, shape)
    odd = np.broadcast_to(((L_AXIS[None, :] - m) % 2) == 1, shape)
    lnps = mode["log_surface_pressure"]
    assert np.abs(lnps[..., off]).max() == 0.0 and np.abs(lnps[..., ~off & odd]).max() == 0.0
    assert np.abs(mode["vorticity"][..., ~off & ~odd]).max() == 0.0
