"""3-D LAGRANTO tracks -> hourly dataset per start level: position, spline velocity and acceleration.

Position: polar-stereographic x, y (km) and log-pressure height z = -H ln(p / 1000 hPa) (km, H = 7 km). Reads
data/lagranto_<flow>_w<k>.npz; writes data/tracks_<flow>_w<k>_l<level>.npz with x_km, v_kmh, a_kmh2 (T, N, 3), lat,
p_hpa (T, N), time_h (T,), patch (N,), sigma_start; parcels that left the domain are dropped.
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
p.add_argument("--flow", required=True)
p.add_argument("--window", type=int, default=0)
p.add_argument("--every-h", type=float, default=1.0)
a = p.parse_args()

with np.load(ROOT / f"data/lagranto_{a.flow}_w{a.window}.npz") as f:
    t, lon, lat, pres, level, patch, sig = (f[k] for k in ("time_h", "lon", "lat", "p", "level", "patch", "sigma_start"))
keep = np.isclose(np.mod(t + 1e-6, a.every_h), 0.0, atol=1e-4)
for k in np.unique(level):
    sel = (level == k) & np.isfinite(lon).all(0) & np.isfinite(lat).all(0) & np.isfinite(pres).all(0)
    lon_u = np.rad2deg(np.unwrap(np.deg2rad(lon[:, sel]), axis=0))
    xyz = np.concatenate([lonlat_to_stereo(lon_u, lat[:, sel]), (-H_KM * np.log(pres[:, sel] / 1000.0))[..., None]], -1)
    spl = CubicSpline(t, xyz, axis=0)
    np.savez(ROOT / f"data/tracks_{a.flow}_w{a.window}_l{k}.npz", time_h=t[keep], x_km=spl(t[keep]).astype(np.float32),
             v_kmh=spl(t[keep], 1).astype(np.float32), a_kmh2=spl(t[keep], 2).astype(np.float32),
             lat=lat[keep][:, sel].astype(np.float32), p_hpa=pres[keep][:, sel].astype(np.float32),
             patch=patch[sel], sigma_start=sig[k], dropped=int((level == k).sum() - sel.sum()))
    print(f"{a.flow} w{a.window} level {k} (sigma {sig[k]}): {sel.sum()} parcels, {(level == k).sum() - sel.sum()} dropped")
