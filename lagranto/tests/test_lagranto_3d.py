"""Analytic checks of 3-D caltra and trace on synthetic sigma-coordinate winds (15 sigma levels, 1 deg grid over
0-90N, hourly files, 48 h, 1-min step, 10-min output).

Run:  LAGRANTO_TEST_DIR=/path/to/scratch micromamba run -n isca python -m pytest -q tests/test_lagranto_3d.py
Needs a built LAGRANTO (bash build.sh). The P files (about 1.3 GB) go to a new directory under $LAGRANTO_TEST_DIR,
removed afterwards unless LAGRANTO_TEST_KEEP=1; without LAGRANTO_TEST_DIR the module is skipped (/tmp is not used).
"""

from __future__ import annotations

import filecmp
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

BASE = os.environ.get("LAGRANTO_TEST_DIR")
SIGMA = np.array([0.018, 0.060, 0.106, 0.152, 0.197, 0.241, 0.287, 0.338, 0.400, 0.477, 0.569, 0.674, 0.784,
                  0.887, 0.967])
A_LAG = 1.112e5 * 180.0 / np.pi  # sphere radius implied by caltra's 1.112e5 m per degree
U0 = 30.0  # m/s, solid-body rotation about the polar axis
PS_AMP = 2000.0  # Pa, surface-pressure wave of case a3
NHOURS = 48
TIMES = np.datetime64("2000-01-01T00") + np.arange(NHOURS + 1) * np.timedelta64(1, "h")
LON = np.arange(0.0, 360.0, 1.0)
LAT = np.arange(90.0, -0.5, -1.0)  # N->S on purpose: the writer must flip it

# Bounds: measured values (README 3-D accuracy table) with a margin of about 1.5.
BOUNDS = {  # (case, hour): (median, max); km for a1, hPa otherwise
    ("a1", 24): (0.45, 2.9), ("a1", 48): (0.9, 5.8),
    ("a3", 24): (0.0015, 0.015), ("a3", 48): (0.003, 0.03),
    ("b_strong", 24): (1.6, 3.3), ("b_strong", 48): (1.4, 3.3),
    ("b_weak", 24): (0.015, 0.11), ("b_weak", 48): (0.03, 0.2),
}
OMEGA_C = {"b_strong": 2.0, "b_weak": 0.02}  # Pa/s: omega = c cos(lon) sigma (1 - sigma)

pytestmark = [pytest.mark.skipif(not (LAGRANTO / "caltra" / "caltra").exists(), reason="run bash build.sh first"),
              pytest.mark.skipif(BASE is None, reason="set LAGRANTO_TEST_DIR to a scratch directory")]


def unit(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def gc_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    a, b = unit(lat1, lon1), unit(lat2, lon2)
    return 6371.0 * np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(-1))


def ps_wave(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Surface pressure of case a3 in Pa."""
    return 1e5 + PS_AMP * np.sin(2 * np.deg2rad(lon)) * np.cos(np.deg2rad(lat))


def omega_a3(lat, lon, sig):
    """omega = sigma u . grad(ps) for u = U0 cos(lat): sigma surfaces are material (sigma-dot = 0)."""
    return sig * U0 * 2 * PS_AMP * np.cos(2 * np.deg2rad(lon)) * np.cos(np.deg2rad(lat)) / A_LAG


def temperature(lat, sig):
    return 220.0 + 70.0 * sig + 10.0 * np.cos(np.deg2rad(lat))


def logistic(s0, kt):
    """Solution of d sigma / dt = k sigma (1 - sigma)."""
    return 1.0 / (1.0 + (1.0 / s0 - 1.0) * np.exp(-kt))


def run(cmd: list[str], cwd: Path, log: str | None = None) -> str:
    env = dict(os.environ, LAGRANTO=str(LAGRANTO), PATH=f"{LAGRANTO / 'bin'}:{os.environ['PATH']}")
    res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    if log:
        (cwd / log).write_text(res.stdout)
    return res.stdout


def caltra(d: Path, out: str, *flags: str) -> dict[str, np.ndarray]:
    log = out.replace(".3", ".log")
    run(["caltra", "20000101_00", "20000103_00", "startf", out, "-i", "1", "-t", "1", "-o", "10", "-j", *flags], d, log)
    assert (d / out).exists()
    return lio.read_traj(d / out)


def write_case(root: Path, case: str) -> tuple[Path, dict[str, np.ndarray]]:
    lat2, lon2 = np.meshgrid(LAT, LON, indexing="ij")
    sig3 = SIGMA[:, None, None]
    nt, shape = TIMES.size, (SIGMA.size,) + lat2.shape
    zero = np.zeros(shape, np.float32)
    ps = np.full(lat2.shape, 1e5)
    t = None
    if case in ("a1", "a3"):
        u, v, om = U0 * np.cos(np.deg2rad(lat2)) * np.ones(shape), zero, zero
        if case == "a3":
            ps, om = ps_wave(lat2, lon2), omega_a3(lat2[None], lon2[None], sig3)
            t = np.broadcast_to(temperature(lat2[None], sig3), shape)
        slat = np.repeat(np.linspace(20.0, 75.0, 10), 20)
        slon = np.tile(np.arange(0.0, 360.0, 18.0), 10)
        ssig = np.repeat([0.967, 0.8, 0.5, 0.152], slat.size)
        slat, slon = np.tile(slat, 4), np.tile(slon, 4)
    else:
        u = v = zero
        om = OMEGA_C[case] * np.cos(np.deg2rad(lon2))[None] * sig3 * (1.0 - sig3)
        slon = np.tile(np.arange(0.0, 360.0, 30.0), 4)
        slat = np.full(slon.size, 45.0)
        ssig = np.repeat([0.4, 0.5, 0.6, 0.7], 12)
    d = root / case
    u, v, om, ps = (np.broadcast_to(a, (nt,) + np.shape(a)) for a in (u, v, om, ps))
    lio.write_pfiles_3d(d, TIMES, LON, LAT, SIGMA, u, v, om, ps, None if t is None else np.broadcast_to(t, om.shape))
    sps = ps_wave(slat, slon) / 100.0 if case == "a3" else np.full(slat.size, 1000.0)
    lio.startfile(d / "startf", slon, slat, ssig * sps)
    return d, {"lat": slat, "lon": slon, "sig": ssig}


@pytest.fixture(scope="module")
def cases():
    Path(BASE).mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="pytest3d_", dir=BASE))
    dirs = {case: write_case(root, case) for case in ("a1", "a3", "b_strong", "b_weak")}
    yield dirs
    if not os.environ.get("LAGRANTO_TEST_KEEP"):
        shutil.rmtree(root, ignore_errors=True)


def at_hour(tr: dict[str, np.ndarray], err: np.ndarray, hour: int) -> np.ndarray:
    i = int(np.argmin(np.abs(tr["time"] - hour)))
    assert abs(tr["time"][i] - hour) < 1e-6
    return err[i]


def zonal_exact(st, hours):
    """Positions under solid-body rotation about the polar axis (caltra's sphere radius)."""
    lon = st["lon"] + np.rad2deg(U0 / A_LAG * hours[:, None] * 3600.0)
    return np.broadcast_to(st["lat"], lon.shape), lon


def check(case, err, tr):
    for hour in (24, 48):
        e = at_hour(tr, err, hour)
        med, mx = BOUNDS[(case, hour)]
        assert np.isfinite(e).all() and np.median(e) < med and e.max() < mx, (case, hour, np.median(e), e.max())


def test_pfile_layout(cases):
    d, _ = cases["a3"]
    files = sorted(d.glob("P*"))
    assert [f.name for f in files[:2]] == ["P20000101_00", "P20000101_01"] and len(files) == NHOURS + 1
    with netCDF4.Dataset(files[0]) as nc:
        assert set(nc.variables) == {"time", "aklev", "aklay", "bklev", "bklay", "pollon", "pollat", "U", "V",
                                     "OMEGA", "T", "PS"}
        for name in ("U", "V", "OMEGA", "T"):
            assert nc[name].dimensions == ("time", "nz", "ny", "nx") and nc[name].dtype == np.float32
            assert nc[name].shape == (1, 15, 91, 361) and nc[name].zstag == 0.0
        assert (nc["U"].xmin, nc["U"].xmax, nc["U"].ymin, nc["U"].ymax) == (0.0, 360.0, 0.0, 90.0)
        np.testing.assert_array_equal(nc["bklev"][:], SIGMA[::-1].astype(np.float32))  # bottom level first
        np.testing.assert_array_equal(nc["bklay"][:], nc["bklev"][:])
        assert not nc["aklev"][:].any() and not nc["aklay"][:].any()
        ps = nc["PS"][0, 0]
        np.testing.assert_allclose(ps, ps_wave(np.arange(0.0, 90.5)[:, None], np.arange(0.0, 361.0)) / 100.0,
                                   rtol=1e-6)
        np.testing.assert_array_equal(nc["U"][0, :, :, 0], nc["U"][0, :, :, 360])
        np.testing.assert_allclose(nc["T"][0, 0, 45, 10], temperature(45.0, 0.967), rtol=1e-6)
        assert nc.constants_file_name == files[0].name
    # Bottom-to-top sigma input (arrays flipped with it) gives identical files.
    alt = d.parent / "a3_flip"
    lat2, lon2 = np.meshgrid(LAT, LON, indexing="ij")
    sig = SIGMA[::-1][:, None, None]
    om = np.broadcast_to(omega_a3(lat2[None], lon2[None], sig), (2,) + sig.shape[:1] + lat2.shape)
    u = np.broadcast_to(U0 * np.cos(np.deg2rad(lat2)), om.shape)
    t = np.broadcast_to(temperature(lat2[None], sig), om.shape)
    lio.write_pfiles_3d(alt, TIMES[:2], LON, LAT, SIGMA[::-1], u, np.zeros_like(u), om,
                        np.broadcast_to(ps_wave(lat2, lon2), (2,) + lat2.shape), t)
    assert filecmp.cmp(alt / "P20000101_01", d / "P20000101_01", shallow=False)


def test_write_rejects_hpa_and_surface_level():
    d = Path(tempfile.mkdtemp(prefix="reject_", dir=BASE))
    try:
        lon, lat, sig = np.arange(0.0, 360.0, 90.0), np.array([0.0, 45.0, 90.0]), np.array([0.9, 0.5])
        f = np.zeros((1, 2, 3, 4))
        with pytest.raises(ValueError, match="Pa"):
            lio.write_pfiles_3d(d, TIMES[:1], lon, lat, sig, f, f, f, np.full((1, 3, 4), 1000.0))
        with pytest.raises(ValueError, match="sigma"):
            lio.write_pfiles_3d(d, TIMES[:1], lon, lat, np.array([1.0, 0.5]), f, f, f, np.full((1, 3, 4), 1e5))
        assert not list(d.iterdir())
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_caltra_wfactor_parser():
    sh = "Vertical scaling factor (wfactor)  : 1. (default: OMEGA in Pa/s)\n"
    f = "   Vertical wind (scale)  :    1.00000000\n"
    assert lio.caltra_wfactor(sh + f) == 1.0
    with pytest.raises(ValueError, match="expected 1"):
        lio.caltra_wfactor(sh.replace("1.", "100.") + f.replace("1.000", "100.0"))
    with pytest.raises(ValueError, match="caltra.sh"):
        lio.caltra_wfactor(f)
    with pytest.raises(ValueError, match="different"):
        lio.caltra_wfactor(sh + f.replace("1.000", "100.0"))


def test_a1_horizontal_solid_body_keeps_pressure(cases):
    d, st = cases["a1"]
    tr = caltra(d, "tr.3")
    assert lio.caltra_wfactor(d / "tr.log") == 1.0
    np.testing.assert_allclose(tr["p"], np.broadcast_to(1000.0 * st["sig"], tr["p"].shape), atol=1e-3)
    lat, lon = zonal_exact(st, tr["time"])
    check("a1", gc_km(tr["lat"], tr["lon"], lat, lon), tr)


def test_a3_sigma_surfaces_follow_ps(cases):
    d, st = cases["a3"]
    tr = caltra(d, "tr.3")
    assert lio.caltra_wfactor(d / "tr.log") == 1.0
    lat, lon = zonal_exact(st, tr["time"])
    check("a3", np.abs(tr["p"] - st["sig"] * ps_wave(lat, lon) / 100.0), tr)
    (d / "tracevars").write_text("U 1. 0 P\nV 1. 0 P\nOMEGA 1. 0 P\nT 1. 0 P\n")
    run(["trace", "tr.3", "tr_uvwt.3"], d)
    tt = lio.read_traj(d / "tr_uvwt.3")
    assert list(tt["columns"]) == ["time", "lon", "lat", "p", "U", "V", "OMEGA", "T"]
    sig = tt["p"] / (ps_wave(tt["lat"], tt["lon"]) / 100.0)
    inside = sig <= SIGMA.max() + 1e-6  # trace keeps the lowest-level value below sigma = 0.967
    assert np.abs(tt["U"] - U0 * np.cos(np.deg2rad(tt["lat"]))).max() < 0.002 and np.abs(tt["V"]).max() < 1e-6
    assert np.abs(tt["OMEGA"] - omega_a3(tt["lat"], tt["lon"], sig))[inside].max() < 1e-4
    assert np.abs(tt["T"] - temperature(tt["lat"], sig))[inside].max() < 1e-3


def piecewise_linear_p(c, g, p0, hours, dt=60.0):
    """RK4 (double precision) of dp/dt = omega(p) as caltra sees it: linear in p between levels, constant above
    the top level, tapered linearly to zero between the lowest level and ps = 1000 hPa."""
    plev = 1000.0 * SIGMA

    def om(p):
        w = g * np.interp(p, plev, c * SIGMA * (1 - SIGMA))
        return np.where(p > plev[-1], w * (1000.0 - p) / (1000.0 - plev[-1]), w)

    p, out = np.asarray(p0, float), {0: np.asarray(p0, float)}
    for n in range(1, int(hours[-1] * 3600 / dt) + 1):
        k1 = om(p)
        k2 = om(p + 0.5 * dt * k1 / 100)
        k3 = om(p + 0.5 * dt * k2 / 100)
        p = p + dt * (k1 + 2 * k2 + 2 * k3 + om(p + dt * k3 / 100)) / 600.0
        out[n] = p
    return np.array([out[int(round(h * 3600 / dt))] for h in hours])


@pytest.mark.parametrize("case", ["b_strong", "b_weak"])
def test_b_vertical_motion(cases, case):
    d, st = cases[case]
    tr = caltra(d, "tr.3")
    assert lio.caltra_wfactor(d / "tr.log") == 1.0
    c, g = OMEGA_C[case], np.cos(np.deg2rad(st["lon"]))
    exact = 1000.0 * logistic(st["sig"], c * g * tr["time"][:, None] * 3600.0 / 1e5)
    check(case, np.abs(tr["p"] - exact), tr)
    assert np.abs(tr["lon"] - st["lon"]).max() == 0 and np.abs(tr["lat"] - st["lat"]).max() == 0
    if case == "b_strong":  # the error is the linear-in-p interpolation of omega, not the time step
        hours = np.array([24.0, 48.0])
        ref = piecewise_linear_p(c, g, 1000.0 * st["sig"], hours)
        p = np.array([at_hour(tr, tr["p"], h) for h in (24, 48)])
        assert np.abs(p - ref).max() < 0.002
    else:  # the trunk's guess: 850-hPa OMEGA range 0.005 Pa/s < 0.4 Pa/s, so wfactor 100
        tr_auto = caltra(d, "tr_auto.3", "-wfactor", "auto")
        assert lio.caltra_wfactor(d / "tr_auto.log", expect=None) == 100.0
        assert np.median(at_hour(tr_auto, np.abs(tr_auto["p"] - exact), 48)) > 100.0
