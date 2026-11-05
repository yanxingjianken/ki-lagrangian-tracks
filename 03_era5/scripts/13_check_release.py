"""Check figure and numbers for one tracked release: are the tracks physical and above the ground?

Left: the tracks of one patch per hemisphere at four start levels on a map (true positions; southern patches are
stored mirrored and are mirrored back here). Right: pressure against time for the same parcels. Printed: parcels
kept and dropped, median 24-h displacement per start level, speed and acceleration percentiles, and, where P files
of the window still exist, how far any parcel is below the surface pressure (bilinear, as caltra).
Writes figures/check_<release>.png.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import great_circle_km, stereo_to_lonlat  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--release", required=True)
p.add_argument("--levels", type=int, nargs="+", default=[0, 7, 11, 16])
a = p.parse_args()

f = np.load(ROOT / f"data/tracks/tracks_{a.release}.npz")
t, x, v, acc, lat, pres = f["time_h"], f["x_km"], f["v_kmh"], f["a_kmh2"], f["lat"], f["p_hpa"]
level, patch, hemi, sigma = f["level"], f["patch"], f["hemisphere"], f["sigma"]
lon, _ = stereo_to_lonlat(x[..., :2])
print(f"{a.release} {f['release_time']}: {lat.shape[1]} parcels kept, {int(f['dropped'])} dropped; T = {t.size}")
d24 = great_circle_km(lon[0], lat[0], lon[24], lat[24])
for k in np.unique(level):
    s = level == k
    print(f"  level {k:2d} sigma {sigma[k]:.3f}: median 24-h displacement {np.median(d24[s]):5.0f} km, "
          f"median |dp| 24 h {np.median(np.abs(pres[24, s] - pres[0, s])):5.1f} hPa, start p {np.median(pres[0, s]):4.0f} hPa")
spd = np.hypot(v[..., 0], v[..., 1])
print(f"  horizontal speed km/h p50/p99/max {np.percentile(spd, 50):.0f}/{np.percentile(spd, 99):.0f}/{spd.max():.0f}; "
      f"|a_h| km/h^2 p99 {np.percentile(np.hypot(acc[..., 0], acc[..., 1]), 99):.0f}; z km {x[..., 2].min():.2f}..{x[..., 2].max():.2f}")

rel = next(r for r in json.loads((ROOT / "configs/releases.json").read_text())["releases"] if r["name"] == a.release)
t0 = np.datetime64(rel["time"], "h")
for h in (0, 24, 72, 156):
    pf = ROOT / "data/pfiles" / ("P" + (t0 + np.timedelta64(h, "h")).astype(object).strftime("%Y%m%d_%H"))
    if not pf.exists():
        continue
    with netCDF4.Dataset(pf) as nc:
        ps = np.asarray(nc["PS"][0, 0], float)
        box = (nc["PS"].xmin, nc["PS"].xmax, nc["PS"].ymin, nc["PS"].ymax)
    it = RegularGridInterpolator((np.linspace(box[2], box[3], ps.shape[0]), np.linspace(box[0], box[1], ps.shape[1])), ps)
    gap = pres[h] - it(np.c_[lat[h], lon[h] % 360.0])
    print(f"  hour {h:3d}: parcels below the surface {int((gap > 0).sum())}, max p - ps {gap.max():.2f} hPa")

fig = plt.figure(figsize=(11, 4.6), layout="constrained")
ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
ax.coastlines(lw=0.4, color="0.5")
ax.set_global()
bx = fig.add_subplot(1, 2, 2)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(a.levels)))
for c, k in zip(colors, a.levels):
    for q in (0, 8):
        s = np.flatnonzero((level == k) & (patch == q))[:40]
        for i in s:
            ax.plot(np.rad2deg(np.unwrap(np.deg2rad(lon[:, i]))), lat[:, i], color=c, lw=0.4,
                    transform=ccrs.Geodetic())
            bx.plot(t, pres[:, i], color=c, lw=0.4)
        ax.plot(lon[0, s], lat[0, s], "k.", ms=1, transform=ccrs.PlateCarree())
    bx.plot([], [], color=c, label=f"start sigma {sigma[k]:.3f}")
bx.invert_yaxis()
bx.set(xlabel="hours after release", ylabel="pressure (hPa)")
bx.legend(fontsize=7, frameon=False)
ax.set_title(f"{a.release} ({f['release_time']}): patch 0 (north) and 8 (south), 40 parcels each; black: start",
             fontsize=8)
bx.set_title("pressure of the same parcels", fontsize=8)
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / f"figures/check_{a.release}.png", dpi=150)
print("wrote", ROOT / f"figures/check_{a.release}.png")
