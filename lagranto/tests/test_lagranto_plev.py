"""3-D caltra on pressure-level P files: the ground must come from PS (write_pfile_plev), not from 1000 hPa.

Synthetic winds on a 2-degree global grid, 6 pressure levels, hourly files over 6 h: u = 10 m/s, v = 0 and uniform
descent omega = 0.5 Pa/s (18 hPa/h). Surface pressure is 1000 hPa except over a plateau (80-100E, 25-45N) at
600 hPa. Parcels start at 560 hPa over the plateau and over the lowland. With write_pfile_plev and `-j` the plateau
parcels stay above 600 hPa; the same files with every bk = 0 (the trap) let them sink to about 668 hPa.

Run:  LAGRANTO_TEST_DIR=/path/to/scratch micromamba run -n kinematic_interpolants python -m pytest -q \
      tests/test_lagranto_plev.py
Needs a built LAGRANTO (bash build.sh); files go to a new directory under $LAGRANTO_TEST_DIR, removed afterwards.
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

BASE = os.environ.get("LAGRANTO_TEST_DIR")
PLEV = np.array([200.0, 300.0, 500.0, 700.0, 850.0, 1000.0])  # top first on purpose: the writer must reorder
LON = np.arange(0.0, 360.0, 2.0)
LAT = np.arange(90.0, -90.5, -2.0)  # N->S on purpose
TIMES = np.datetime64("2016-03-01T00") + np.arange(7) * np.timedelta64(1, "h")
OMEGA = 0.5  # Pa/s, descent of 18 hPa/h

pytestmark = [pytest.mark.skipif(not (LAGRANTO / "caltra" / "caltra").exists(), reason="run bash build.sh first"),
              pytest.mark.skipif(BASE is None, reason="set LAGRANTO_TEST_DIR to a scratch directory")]


def fields():
    shape = (PLEV.size, LAT.size, LON.size)
    u, v, w = np.full(shape, 10.0), np.zeros(shape), np.full(shape, OMEGA)
    ps = np.full((LAT.size, LON.size), 1.0e5)
    box = (LAT[:, None] >= 25) & (LAT[:, None] <= 45) & (LON[None] >= 80) & (LON[None] <= 100)
    ps[box] = 6.0e4
    return u, v, w, ps


def run_caltra(wdir: Path, lon, lat, p) -> dict:
    lio.startfile(wdir / "startf", lon, lat, p, lon_min=0.0)
    env = dict(os.environ, LAGRANTO=str(LAGRANTO), PATH=f"{LAGRANTO / 'bin'}:{os.environ['PATH']}")
    first, last = (lio._pname(t)[1:] for t in (TIMES[0], TIMES[-1]))
    out = subprocess.run(["caltra", first, last, "startf", "traj.3", "-i", "1", "-t", "1", "-o", "10", "-j"],
                         cwd=wdir, env=env, capture_output=True, text=True)
    (wdir / "caltra.log").write_text(out.stdout + out.stderr)
    assert out.returncode == 0 and (wdir / "traj.3").exists(), out.stdout[-2000:] + out.stderr[-2000:]
    lio.caltra_wfactor(wdir / "caltra.log", expect=1.0)
    return lio.read_traj(wdir / "traj.3")


@pytest.fixture(scope="module")
def work():
    d = Path(tempfile.mkdtemp(prefix="plev_", dir=BASE))
    yield d
    shutil.rmtree(d, ignore_errors=True)


START_LON = np.array([84.0, 88.0, 92.0, 200.0, 220.0])
START_LAT = np.array([35.0, 35.0, 35.0, 35.0, 35.0])
PLATEAU = START_LON < 100


def test_file_layout(work):
    u, v, w, ps = fields()
    path = lio.write_pfile_plev(work / "plev", TIMES[0], LON, LAT, PLEV, u, v, w, ps)
    assert not list((work / "plev").glob("*.part"))
    with netCDF4.Dataset(path) as nc:
        ak, bk = nc["aklev"][:], nc["bklev"][:]
        assert np.allclose(ak + bk * 1000.0, PLEV[::-1]) and bk[0] == lio.BK_BOTTOM and np.all(bk[1:] == 0)
        assert nc["U"].shape == (1, PLEV.size, LAT.size, LON.size + 1)
        psf = nc["PS"][0, 0]
        assert psf.min() == pytest.approx(600.0) and psf.max() == pytest.approx(1000.0)
        assert np.allclose(nc["PS"][0, 0, :, 0], nc["PS"][0, 0, :, -1])  # closed ring


def test_parcels_stay_above_the_plateau(work):
    u, v, w, ps = fields()
    wdir = work / "plev"
    for t in TIMES:
        lio.write_pfile_plev(wdir, t, LON, LAT, PLEV, u, v, w, ps)
    tr = run_caltra(wdir, START_LON, START_LAT, np.full(START_LON.size, 560.0))
    p_end = tr["p"][-1]
    assert np.all(np.isfinite(p_end))
    assert p_end[PLATEAU].max() <= 600.0, p_end
    # lowland parcels descend freely: 560 hPa + 18 hPa/h x 6 h
    np.testing.assert_allclose(p_end[~PLATEAU], 560.0 + OMEGA * 36.0 * 6.0, atol=1.0)


def test_pure_pressure_levels_ignore_ps(work):
    """The trap the bottom-level bk avoids: with every bk = 0 the plateau parcels sink through the ground."""
    u, v, w, ps = fields()
    wdir = work / "pure"
    lon_w, lat_w, flip, close = lio._horizontal(LON, LAT)
    p = PLEV[::-1].astype(np.float32)
    wdir.mkdir()
    for t in TIMES:
        hour = np.float32((t - TIMES[0]) / np.timedelta64(1, "h"))
        lio._write_pfile(wdir / lio._pname(t), hour, lon_w, lat_w, p, np.zeros_like(p),
                         {n: lio._orient(a[::-1], flip, close) for n, a in (("U", u), ("V", v), ("OMEGA", w))},
                         lio._orient(ps, flip, close) / 100.0, (float(p[0]), float(p[-1])))
    tr = run_caltra(wdir, START_LON, START_LAT, np.full(START_LON.size, 560.0))
    assert tr["p"][-1][PLATEAU].min() > 650.0
