# Copy of roger/03_era5/scripts/12_dataset.py.
"""LAGRANTO tracks of one release -> hourly dataset: position, spline velocity and acceleration (float32, as 02).

Position: north-polar stereographic x, y (km) and log-pressure height z = -H ln(p / 1000 hPa) (km, H = 7 km).
Southern-hemisphere patches are mirrored to the north first (latitude -> -latitude; longitude kept), which turns a
southern flow into an equally valid northern one (the Coriolis parameter changes sign with latitude, v with the
mirror). A cubic spline through the 10-min positions gives hourly x, v, a. Parcels with a missing position (start
below the ground, or outside the domain) are dropped.

Reads data/lagranto/<release>.npz; writes data/tracks/tracks_<release>.npz with time_h (T,), x_km, v_kmh, a_kmh2
(T, N, 3), lat (true latitude), p_hpa (T, N), level, patch, hemisphere (N,), sigma, release_time, dropped; then
removes the LAGRANTO file.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import lonlat_to_stereo  # noqa: E402

H_KM = 7.0
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--release", required=True)
p.add_argument("--every-h", type=float, default=1.0)
p.add_argument("--chunk", type=int, default=20000, help="parcels per spline call (memory)")
p.add_argument("--keep-lagranto", action="store_true")
a = p.parse_args()

src = ROOT / f"data/lagranto/{a.release}.npz"
with np.load(src) as f:
    t, lon, lat, pres = (f[k] for k in ("time_h", "lon", "lat", "p"))
    level, patch, sigma, rtime = f["level"], f["patch"], f["sigma"], str(f["release_time"])
ok = np.isfinite(lon).all(0) & np.isfinite(lat).all(0) & np.isfinite(pres).all(0)
idx = np.flatnonzero(ok)
hemi = np.where(lat[0, idx] >= 0, 1, -1).astype(np.int8)
keep = np.isclose(np.mod(t + 1e-6, a.every_h), 0.0, atol=1e-4)
th = t[keep]
T, N = th.size, idx.size
X, V, A = (np.empty((T, N, 3), np.float32) for _ in range(3))
for s in range(0, N, a.chunk):
    c = idx[s:s + a.chunk]
    lam = np.rad2deg(np.unwrap(np.deg2rad(lon[:, c].astype(np.float64)), axis=0))
    phi = lat[:, c].astype(np.float64) * hemi[s:s + a.chunk]
    xyz = np.concatenate([lonlat_to_stereo(lam, phi), (-H_KM * np.log(pres[:, c] / 1000.0))[..., None]], -1)
    spl = CubicSpline(t, xyz, axis=0)
    X[:, s:s + c.size], V[:, s:s + c.size], A[:, s:s + c.size] = spl(th), spl(th, 1), spl(th, 2)
out = ROOT / "data/tracks"
out.mkdir(parents=True, exist_ok=True)
tmp = out / f"tracks_{a.release}.part.npz"
np.savez(tmp, time_h=th, x_km=X, v_kmh=V, a_kmh2=A, lat=lat[keep][:, idx], p_hpa=pres[keep][:, idx],
         level=level[idx], patch=patch[idx], hemisphere=hemi, sigma=sigma, release_time=rtime,
         dropped=int((~ok).sum()))
tmp.rename(out / f"tracks_{a.release}.npz")
if not a.keep_lagranto:
    src.unlink()
print(f"{a.release} {rtime}: {N} parcels kept, {(~ok).sum()} dropped; {T} hourly times")
