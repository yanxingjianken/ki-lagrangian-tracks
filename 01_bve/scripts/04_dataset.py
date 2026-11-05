"""LAGRANTO tracks -> KI dataset: polar-stereographic positions, spline velocity and acceleration, split by parcel.

Reads data/<source>_w<k>.npz (time_h, lon, lat, patch); writes data/tracks_<source>_w<k>.npz with x_km, v_kmh, a_kmh2
(T, N, 2), lon, lat (T, N), time_h (T,), patch (N,), split (N,: 0 train, 1 validation, 2 test).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ki_tracks import lonlat_to_stereo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--window", type=int, default=0)
p.add_argument("--run", default="stirred_jet", help="data/bve_<run>.nc")
p.add_argument("--source", default="lagranto", help="lagranto, or heun for the cross-check tracks")
p.add_argument("--fractions", type=float, nargs=3, default=[0.7, 0.1, 0.2])
p.add_argument("--trim-h", type=float, default=0.0, help="drop this much at both ends")
p.add_argument("--every-h", type=float, default=None, help="store every this many hours (default: every track time)")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

suffix = "" if a.run == "stirred_jet" else f"_{a.run}"
with np.load(ROOT / f"data/{a.source}{suffix}_w{a.window}.npz") as f:
    time_h, lon, lat, patch = f["time_h"], f["lon"], f["lat"], f["patch"]
    start_hour, centres = int(f["start_hour"]), f["centres"]
ok = np.isfinite(lon).all(0) & np.isfinite(lat).all(0)
lon_u = np.rad2deg(np.unwrap(np.deg2rad(lon[:, ok]), axis=0))
x = lonlat_to_stereo(lon_u, lat[:, ok])
spl = CubicSpline(time_h, x, axis=0)
keep = (time_h >= time_h[0] + a.trim_h) & (time_h <= time_h[-1] - a.trim_h)
if a.every_h:
    keep &= np.isclose(np.mod(time_h - time_h[0] + 1e-6, a.every_h), 0.0, atol=1e-4)
t = time_h[keep]
patch = patch[ok]
rng = np.random.default_rng(a.seed)
split = np.empty(len(patch), np.int8)
for k in np.unique(patch):
    idx = rng.permutation(np.where(patch == k)[0])
    n_tr, n_va = int(round(a.fractions[0] * len(idx))), int(round(a.fractions[1] * len(idx)))
    split[idx[:n_tr]], split[idx[n_tr:n_tr + n_va]], split[idx[n_tr + n_va:]] = 0, 1, 2
np.savez(ROOT / f"data/tracks_{a.source}{suffix}_w{a.window}.npz", time_h=t - t[0], time_h_abs=t, x_km=spl(t).astype(np.float32),
         v_kmh=spl(t, 1).astype(np.float32), a_kmh2=spl(t, 2).astype(np.float32),
         lon=(lon_u[keep] % 360.0).astype(np.float32), lat=lat[keep][:, ok].astype(np.float32), patch=patch,
         split=split, dropped=int((~ok).sum()), start_hour=start_hour, centres=centres)
print(f"window {a.window}: {ok.sum()} parcels kept, {(~ok).sum()} dropped, {len(t)} times")
