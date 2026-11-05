"""Seed the parcel patches and compute their 2-D tracks with LAGRANTO (caltra -2d) from the window's P files.

Writes data/lagranto/w<k>/startf, traj.3 (LAGRANTO binary) and data/lagranto_w<k>.npz (time_h, lon, lat, patch,
centres, start_hour).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAGRANTO = ROOT.parent / "lagranto"
sys.path.insert(0, str(LAGRANTO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lagranto_io import read_traj, startfile  # noqa: E402
from patches import seed_patches  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--window", type=int, default=0)
p.add_argument("--run", default="stirred_jet", help="data/bve_<run>.nc")
p.add_argument("--n-patches", type=int, default=8)
p.add_argument("--n-per", type=int, default=1000)
p.add_argument("--radius-km", type=float, default=300.0)
p.add_argument("--min-sep-km", type=float, default=2000.0)
p.add_argument("--lat-range", type=float, nargs=2, default=[35.0, 60.0])
p.add_argument("--step-min", type=int, default=1, help="1 min: parcels near the pole stay accurate")
p.add_argument("--out-min", type=int, default=10)
a = p.parse_args()

suffix = "" if a.run == "stirred_jet" else f"_{a.run}"
wdir = ROOT / f"data/lagranto/w{a.window}{suffix}"
info = dict(line.split(" ", 1) for line in (wdir / "window.txt").read_text().splitlines())
pfiles = sorted(wdir.glob("P*_*"))
first, last = pfiles[0].name[1:], pfiles[-1].name[1:]
assert first == np.datetime_as_string(np.datetime64(info["first"]), unit="h").replace("-", "").replace("T", "_")
lat, lon, pid, centres = seed_patches(a.n_patches, a.n_per, a.radius_km, tuple(a.lat_range), a.min_sep_km, seed=a.window)
startfile(wdir / "startf", lon, lat, 500.0, lon_min=0.0)
env = dict(os.environ, LAGRANTO=str(LAGRANTO), PATH=f"{LAGRANTO / 'bin'}:{os.environ['PATH']}")
cmd = ["caltra", first, last, "startf", "traj.3", "-2d", "-i", "1", "-t", str(a.step_min), "-o", str(a.out_min)]
log = subprocess.run(cmd, cwd=wdir, env=env, capture_output=True, text=True)
(wdir / "caltra.log").write_text(log.stdout + log.stderr)
if log.returncode or not (wdir / "traj.3").exists():
    raise RuntimeError(f"caltra failed, see {wdir / 'caltra.log'}")
tr = read_traj(wdir / "traj.3")
np.savez(ROOT / f"data/lagranto{suffix}_w{a.window}.npz", time_h=tr["time"] - tr["time"][0], lon=tr["lon"], lat=tr["lat"],
         patch=pid, centres=centres, start_hour=int(info["start_hour"]))
print(f"{tr['lon'].shape[1]} tracks, {len(tr['time'])} times, {np.isnan(tr['lon']).any(0).sum()} left the domain")
