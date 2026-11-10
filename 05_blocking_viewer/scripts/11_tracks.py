# Adapted from roger/03_era5/scripts/11_tracks.py (patch centres of the event releases, 3 start levels).
"""Seed the parcel patches of one event release and compute their 3-D tracks with LAGRANTO caltra.

Per start level (sigma x local surface pressure, configs/releases.json) and patch centre of the release: 480 parcels
uniform in a 300-km disk. Parcels are ordered by level and patch and split into --jobs caltra runs, each in its own
directory under data/lagranto/<release>/ with links to the P files of the 157-h window. caltra: full 3-D, -j, 1-min
step, 10-min output, vertical scaling factor checked = 1. Writes data/lagranto/<release>.npz (time_h, lon, lat, p,
level, patch, centres, sigma, release time); the job directories are removed.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import netCDF4
import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
LAGRANTO = ROOT.parent / "lagranto"
sys.path.insert(0, str(LAGRANTO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lagranto_io as lio  # noqa: E402
from patches import destination  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--release", required=True)
p.add_argument("--jobs", type=int, default=8)
p.add_argument("--step-min", type=int, default=1)
p.add_argument("--out-min", type=int, default=10)
p.add_argument("--pdir", default=str(ROOT / "data/pfiles"))
a = p.parse_args()

cfg = json.loads((ROOT / "configs/releases.json").read_text())
rel = next(r for r in cfg["releases"] if r["name"] == a.release)
t0 = np.datetime64(rel["time"], "h")
hours = t0 + np.arange(cfg["window_h"] + 1) * np.timedelta64(1, "h")
pfiles = [Path(a.pdir) / lio._pname(t) for t in hours]
missing = [f.name for f in pfiles if not f.exists()]
if missing:
    raise FileNotFoundError(f"{a.release}: {len(missing)} P files missing, first {missing[0]}")

with netCDF4.Dataset(pfiles[0]) as nc:  # surface pressure at release time, as caltra sees it (hPa, S->N, closed)
    ps0 = np.asarray(nc["PS"][0, 0], dtype=np.float64)
    box = (nc["PS"].xmin, nc["PS"].xmax, nc["PS"].ymin, nc["PS"].ymax)
lon_g = np.linspace(box[0], box[1], ps0.shape[1])
lat_g = np.linspace(box[2], box[3], ps0.shape[0])
ps_at = RegularGridInterpolator((lat_g, lon_g), ps0)

rng = np.random.default_rng(1000 * rel["index"] + sum(map(ord, rel["event"])))
lat, lon, pres, lev, pid, cen = [], [], [], [], [], []
for k, sigma in enumerate(cfg["sigma"]):
    for q, (clat, clon) in enumerate(rel["centres"]):
        r = cfg["radius_km"] * np.sqrt(rng.uniform(0.0, 1.0, cfg["n_per"]))
        la, lo = destination(clat, clon, rng.uniform(0.0, 2 * np.pi, cfg["n_per"]), r)
        lat.append(la)
        lon.append(lo)
        pres.append(sigma * ps_at(np.c_[la, lo % 360.0]))
        lev.append(np.full(la.size, k))
        pid.append(np.full(la.size, q))
        cen.append((k, q, clat, clon))
lat, lon, pres, lev, pid = map(np.concatenate, (lat, lon, pres, lev, pid))
cen = np.asarray(cen)

wdir = ROOT / f"data/lagranto/{a.release}"
shutil.rmtree(wdir, ignore_errors=True)
env = dict(os.environ, LAGRANTO=str(LAGRANTO), PATH=f"{LAGRANTO / 'bin'}:{os.environ['PATH']}",
           OMP_NUM_THREADS="1")
first, last = pfiles[0].name[1:], pfiles[-1].name[1:]
chunks = np.array_split(np.arange(lat.size), a.jobs)
procs = []
for j, idx in enumerate(chunks):
    jd = wdir / f"j{j}"
    jd.mkdir(parents=True)
    for f in pfiles:
        (jd / f.name).symlink_to(f)
    lio.startfile(jd / "startf", lon[idx], lat[idx], pres[idx], lon_min=0.0)
    cmd = ["caltra", first, last, "startf", "traj.3", "-i", "1", "-t", str(a.step_min), "-o", str(a.out_min), "-j"]
    log = open(jd / "caltra.log", "w")
    procs.append((subprocess.Popen(cmd, cwd=jd, env=env, stdout=log, stderr=subprocess.STDOUT), log, jd))
parts = []
for proc, log, jd in procs:
    rc = proc.wait()
    log.close()
    if rc or not (jd / "traj.3").exists():
        raise RuntimeError(f"caltra failed in {jd}, see {jd / 'caltra.log'}")
    lio.caltra_wfactor(jd / "caltra.log", expect=1.0)
    parts.append(lio.read_traj(jd / "traj.3"))
time_h = parts[0]["time"] - parts[0]["time"][0]
assert all(np.array_equal(q["time"], parts[0]["time"]) for q in parts)
out = {k: np.concatenate([q[k] for q in parts], axis=1).astype(np.float32) for k in ("lon", "lat", "p")}
assert out["lon"].shape[1] == lat.size
np.savez(ROOT / f"data/lagranto/{a.release}.npz", time_h=time_h, **out, level=lev.astype(np.int16),
         patch=pid.astype(np.int16), centres=cen, sigma=np.asarray(cfg["sigma"]), release_time=str(t0))
shutil.rmtree(wdir)
bad = (~np.isfinite(out["lon"])).any(0)
print(f"{a.release} {t0}: {lat.size} tracks in {a.jobs} caltra runs, {len(time_h)} times, {bad.sum()} incomplete")
