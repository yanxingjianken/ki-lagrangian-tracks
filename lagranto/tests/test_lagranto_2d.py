"""Analytic checks of caltra -2d and trace on synthetic solid-body winds (1 deg global grid, hourly, 48 h).

Run:  micromamba run -n isca python -m pytest -q tests/test_lagranto_2d.py
Needs a built LAGRANTO (bash build.sh). Scratch files go to $LAGRANTO_TEST_DIR (default: a pytest temporary directory);
set LAGRANTO_TEST_KEEP=1 to keep them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import netCDF4
import numpy as np
import pytest

LAGRANTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAGRANTO / "python"))
import lagranto_io as lio  # noqa: E402

SCRATCH = Path(os.environ.get("LAGRANTO_TEST_DIR", tempfile.mkdtemp(prefix="lagranto_test_")))
A_LAG = 1.112e5 * 180.0 / np.pi  # sphere radius implied by caltra's 1.112e5 m per degree
U0 = 30.0
NHOURS = 48

# Bounds: measured values (README accuracy table) with a margin of about 1.5.
BOUNDS_T5 = {  # (case, hour): (median km, max km), 5-min step
    ("zonal", 24): (0.10, 0.15), ("zonal", 48): (0.20, 0.30),
    ("tilted", 24): (0.60, 4.5), ("tilted", 48): (1.0, 7.5),
}

pytestmark = pytest.mark.skipif(not (LAGRANTO / "caltra" / "caltra").exists(), reason="run bash build.sh first")


def unit(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def axis(case: str) -> np.ndarray:
    return unit(np.array(90.0), np.array(0.0)) if case == "zonal" else unit(np.array(45.0), np.array(0.0))


def wind(case: str, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solid-body rotation about the case axis, peak speed U0."""
    vel = U0 * np.cross(axis(case), unit(lat, lon))
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    e_lon = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], -1)
    e_lat = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], -1)
    return (vel * e_lon).sum(-1), (vel * e_lat).sum(-1)


def analytic(case: str, lat0: np.ndarray, lon0: np.ndarray, hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact positions: rotation of the start points about the axis by U0 t / a."""
    ax, r0 = axis(case), unit(lat0, lon0)
    ang = (U0 / A_LAG) * hours[:, None] * 3600.0
    c, s = np.cos(ang)[..., None], np.sin(ang)[..., None]
    r = r0 * c + np.cross(ax, r0) * s + ax * (r0 @ ax)[..., None] * (1.0 - c)
    return np.rad2deg(np.arcsin(np.clip(r[..., 2], -1, 1))), np.rad2deg(np.arctan2(r[..., 1], r[..., 0]))


def gc_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    a, b = unit(lat1, lon1), unit(lat2, lon2)
    return 6371.0 * np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(-1))


def run(cmd: list[str], cwd: Path) -> str:
    env = dict(os.environ, LAGRANTO=str(LAGRANTO), PATH=f"{LAGRANTO / 'bin'}:{os.environ['PATH']}")
    res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    return res.stdout


def caltra(d: Path, out: str, step: int) -> dict[str, np.ndarray]:
    run(["caltra", "20000101_00", "20000103_00", "startf", out, "-2d", "-i", "1", "-t", str(step), "-o", "10"], d)
    return lio.read_traj(d / out)


@pytest.fixture(scope="module")
def cases():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="pytest_", dir=SCRATCH))
    lon = np.arange(0.0, 360.0, 1.0)
    lat = np.arange(90.0, -90.5, -1.0)  # N->S on purpose: the writer must flip it
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    times = np.datetime64("2000-01-01T00") + np.arange(NHOURS + 1) * np.timedelta64(1, "h")
    slon = np.tile(np.arange(0.0, 360.0, 18.0), 10)
    slat = np.repeat(np.linspace(20.0, 75.0, 10), 20)
    dirs = {}
    for case in ("zonal", "tilted"):
        u, v = wind(case, lat2, lon2)
        shape = (times.size,) + u.shape
        d = root / case
        lio.write_pfiles(d, times, lon, lat, np.broadcast_to(u, shape), np.broadcast_to(v, shape))
        lio.startfile(d / "startf", slon, slat, 500.0)
        dirs[case] = d
    yield dirs, slat, slon
    if not os.environ.get("LAGRANTO_TEST_KEEP"):
        shutil.rmtree(root, ignore_errors=True)


def errors_at(case, tr, slat, slon, hour):
    alat, alon = analytic(case, slat, slon, tr["time"])
    err = gc_km(tr["lat"], tr["lon"], alat, alon)
    i = int(np.argmin(np.abs(tr["time"] - hour)))
    assert abs(tr["time"][i] - hour) < 1e-6
    return err[i]


def test_pfile_layout(cases):
    dirs, _, _ = cases
    files = sorted(dirs["zonal"].glob("P*"))
    assert [f.name for f in files[:2]] == ["P20000101_00", "P20000101_01"] and len(files) == NHOURS + 1
    with netCDF4.Dataset(files[0]) as nc:
        assert nc["U"].dimensions == ("time", "nz", "ny", "nx") and nc["U"].dtype == np.float32
        assert nc["U"].shape == (1, 2, 181, 361)
        assert (nc["U"].xmin, nc["U"].xmax, nc["U"].ymin, nc["U"].ymax) == (0.0, 360.0, -90.0, 90.0)
        np.testing.assert_array_equal(nc["U"][0, 0, :, 0], nc["U"][0, 0, :, 360])
        np.testing.assert_array_equal(nc["P"][0, :, 0, 0], [600.0, 400.0])
        assert nc.constants_file_name == files[0].name


def test_zonal_solid_body(cases):
    dirs, slat, slon = cases
    tr = caltra(dirs["zonal"], "tr_t5.3", 5)
    assert np.isfinite(tr["lat"]).all()
    assert np.abs(tr["lat"] - slat).max() < 1e-4
    dlon = np.mod(np.diff(tr["lon"], axis=0) + 180.0, 360.0) - 180.0
    exact = np.rad2deg(U0 / A_LAG * 600.0)
    assert np.abs(dlon - exact).max() < 2e-4
    np.testing.assert_allclose(tr["p"], 500.0, atol=1e-3)
    for hour in (24, 48):
        e = errors_at("zonal", tr, slat, slon, hour)
        med, mx = BOUNDS_T5[("zonal", hour)]
        assert np.median(e) < med and e.max() < mx, (hour, np.median(e), e.max())


def test_tilted_solid_body_and_step(cases):
    dirs, slat, slon = cases
    tr5 = caltra(dirs["tilted"], "tr_t5.3", 5)
    tr1 = caltra(dirs["tilted"], "tr_t1.3", 1)
    for hour in (24, 48):
        e = errors_at("tilted", tr5, slat, slon, hour)
        med, mx = BOUNDS_T5[("tilted", hour)]
        assert np.median(e) < med and e.max() < mx, (hour, np.median(e), e.max())
    e5, e1 = errors_at("tilted", tr5, slat, slon, 48), errors_at("tilted", tr1, slat, slon, 48)
    assert 4.0 < e5.max() / e1.max() < 6.0
    assert 3.5 < np.median(e5) / np.median(e1) < 6.0


def test_ascii_output_matches_binary(cases):
    dirs, _, _ = cases
    d = dirs["tilted"]
    a, b = caltra(d, "tr_t5.1", 5), lio.read_traj(d / "tr_t5.3")
    assert a["refdate"] == np.datetime64("2000-01-01T00:00") and np.allclose(a["time"], b["time"])
    assert np.abs(a["lat"] - b["lat"]).max() <= 0.0051
    assert np.abs(np.mod(a["lon"] - b["lon"] + 180.0, 360.0) - 180.0).max() <= 0.0051


def test_trace_uv(cases):
    dirs, slat, slon = cases
    d = dirs["tilted"]
    if not (d / "tr_t5.3").exists():
        caltra(d, "tr_t5.3", 5)
    (d / "tracevars").write_text("U 1. 0 P\nV 1. 0 P\n")
    run(["trace", "tr_t5.3", "tr_t5_uv.3"], d)
    tr = lio.read_traj(d / "tr_t5_uv.3")
    assert list(tr["columns"]) == ["time", "lon", "lat", "p", "U", "V"]
    u, v = wind("tilted", tr["lat"], tr["lon"])
    assert np.abs(tr["U"] - u).max() < 0.01 and np.abs(tr["V"] - v).max() < 0.01
    run(["trace", "tr_t5.3", "tr_t5_uv.1"], d)
    asc = lio.read_traj(d / "tr_t5_uv.1")
    assert np.abs(asc["U"] - tr["U"]).max() <= 0.0006
