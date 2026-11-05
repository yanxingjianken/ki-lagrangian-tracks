"""Seed the 3-D parcel patches of one flow and window and compute their tracks with LAGRANTO (full 3-D caltra).

Patches: 1000 parcels in a 300-km disk on one model level (start pressure = sigma x local surface pressure); 8 patch
centres per level (30-65N, >= 2000 km apart; a different draw per level). Writes data/lagranto/<flow>_w<k>/traj.3,
caltra.log and data/lagranto_<flow>_w<k>.npz (time_h, lon, lat, p, level, patch, centres, sigma_start).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
LAGRANTO = ROOT.parent / "lagranto"
sys.path.insert(0, str(LAGRANTO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lagranto_io as lio  # noqa: E402
from patches import seed_patches  # noqa: E402

LEVELS = (0.967, 0.887, 0.674, 0.477, 0.241)
p = argparse.ArgumentParser()
p.add_argument("--flow", required=True)
p.add_argument("--window", type=int, default=0)
p.add_argument("--n-patches", type=int, default=8)
p.add_argument("--n-per", type=int, default=1000)
p.add_argument("--step-min", type=int, default=1)
p.add_argument("--out-min", type=int, default=10)
a = p.parse_args()

wdir = ROOT / f"data/lagranto/{a.flow}_w{a.window}"
pfiles = sorted(wdir.glob("P*_*"))
first, last = pfiles[0].name[1:], pfiles[-1].name[1:]
ds = xr.open_dataset(ROOT / f"data/dino_{a.flow}.nc")
model_sigma = ds.level.sel(level=list(LEVELS), method="nearest").values
assert np.allclose(model_sigma, LEVELS, atol=2e-3), model_sigma
h0 = 24 * (2 + 6 * a.window)
ps0 = ds.ps.isel(time=h0).values / 100.0
lon_c = np.r_[ds.lon.values, 360.0]
ps_at = RegularGridInterpolator((ds.lat.values, lon_c), np.concatenate([ps0, ps0[:, :1]], 1))

lat, lon, pres, lev, pid, cen = [], [], [], [], [], []
for k, sigma in enumerate(model_sigma):
    la, lo, q, c = seed_patches(a.n_patches, a.n_per, 300.0, (30.0, 65.0), 2000.0, seed=100 * a.window + k)
    lat.append(la)
    lon.append(lo)
    pres.append(sigma * ps_at(np.c_[la, lo % 360.0]))
    lev.append(np.full(la.size, k))
    pid.append(q)
    cen.append(c)
lat, lon, pres, lev, pid = map(np.concatenate, (lat, lon, pres, lev, pid))
lio.startfile(wdir / "startf", lon, lat, pres, lon_min=0.0)
env = dict(os.environ, LAGRANTO=str(LAGRANTO), PATH=f"{LAGRANTO / 'bin'}:{os.environ['PATH']}")
cmd = ["caltra", first, last, "startf", "traj.3", "-i", "1", "-t", str(a.step_min), "-o", str(a.out_min), "-j"]
log = subprocess.run(cmd, cwd=wdir, env=env, capture_output=True, text=True)
(wdir / "caltra.log").write_text(log.stdout + log.stderr)
if log.returncode or not (wdir / "traj.3").exists():
    raise RuntimeError(f"caltra failed, see {wdir / 'caltra.log'}")
lio.caltra_wfactor(wdir / "caltra.log", expect=1.0)
tr = lio.read_traj(wdir / "traj.3")
np.savez(ROOT / f"data/lagranto_{a.flow}_w{a.window}.npz", time_h=tr["time"] - tr["time"][0], lon=tr["lon"],
         lat=tr["lat"], p=tr["p"], level=lev, patch=pid, centres=np.stack(cen), sigma_start=model_sigma,
         start_hour=h0)
print(f"{tr['lon'].shape[1]} tracks, {len(tr['time'])} times, {np.isnan(tr['lon']).any(0).sum()} left the domain")
