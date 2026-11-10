"""Locate each event's block on ERA5 500-hPa height: maps and the area where the height stands far above its
zonal mean, for choosing the region and life-cycle window of the track cloud.

Z500 every 6 h at 1 deg (ARCO ERA5: the 1-deg hourly store to 2021, the 0.25-deg store thinned to 1 deg after) over a
search area and window per event (configs/events_search.json). Anomaly = Z500 minus its zonal mean at the same
latitude and time (ridges and blocks positive). Writes data/z500_<event>.npz and figures/events_<event>.png (maps
every 2 days, shading = anomaly, contours = Z500) and prints, per day, the largest anomaly in the search area and its
position.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STORE_1DEG = "gcp-public-data-arco-era5/ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr"
STORE_025 = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
G = 9.80665


def z500(t0, t1):
    import gcsfs
    import xarray as xr
    fs = gcsfs.GCSFileSystem(token="anon")
    times = np.arange(np.datetime64(t0, "h"), np.datetime64(t1, "h") + 1, np.timedelta64(6, "h"))
    if times[-1] < np.datetime64("2022-01-01T00"):
        ds = xr.open_zarr(fs.get_mapper(STORE_1DEG), chunks=None)
        z = ds["geopotential"].sel(time=times, level=500).transpose("time", "latitude", "longitude")
        lat, lon = z.latitude.values, z.longitude.values
        z = z.values
    else:
        ds = xr.open_zarr(fs.get_mapper(STORE_025), chunks=None)
        z = ds["geopotential"].sel(time=times, level=500).isel(latitude=slice(None, None, 4), longitude=slice(None, None, 4))
        lat, lon = z.latitude.values, z.longitude.values
        z = z.values
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[:, ::-1]
    return times, lat, lon, (z / G).astype(np.float32)


def main(name):
    import matplotlib
    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    ev = json.loads((ROOT / "configs/events_search.json").read_text())[name]
    path = ROOT / f"data/z500_{name}.npz"
    if path.exists():
        d = np.load(path)
        times, lat, lon, z = d["times"].astype("datetime64[h]"), d["lat"], d["lon"], d["z"]
    else:
        times, lat, lon, z = z500(ev["window"][0], ev["window"][1])
        path.parent.mkdir(exist_ok=True)
        np.savez(path, times=times.astype(str), lat=lat, lon=lon, z=z)
    anom = z - z.mean(axis=2, keepdims=True)
    (la0, la1), (lo0, lo1) = ev["lat"], ev["lon"]
    mlat = (lat >= la0) & (lat <= la1)
    mlon = ((lon >= lo0) & (lon <= lo1)) if lo0 < lo1 else ((lon >= lo0) | (lon <= lo1))
    sub = anom[:, mlat][:, :, mlon]
    print(f"== {name}")
    for k in range(0, len(times), 4):
        i = np.unravel_index(np.argmax(sub[k]), sub[k].shape)
        print(f"  {str(times[k])[:13]}  max anomaly {sub[k][i]:5.0f} m at {lat[mlat][i[0]]:.0f}N {lon[mlon][i[1]]:.0f}E")
    show = list(range(0, len(times), 8))[:8]
    clon = float((lo0 + ((lo1 - lo0) % 360) / 2) % 360)
    proj = ccrs.NorthPolarStereo(central_longitude=clon)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), subplot_kw=dict(projection=proj), layout="constrained")
    for ax, k in zip(axes.flat, show):
        ax.set_extent([clon - 70, clon + 70, 25, 90], ccrs.PlateCarree())
        lw = np.r_[lon, 360.0]
        cyc = lambda a: np.concatenate([a, a[:, :1]], 1)  # noqa: E731
        cf = ax.contourf(lw, lat, cyc(anom[k]), levels=np.arange(-400, 401, 50), cmap="RdBu_r", extend="both",
                         transform=ccrs.PlateCarree())
        ax.contour(lw, lat, cyc(z[k]), levels=np.arange(4800, 6000, 80), colors="k", linewidths=0.5,
                   transform=ccrs.PlateCarree())
        ax.plot([lo0, lo0, lo1 if lo1 > lo0 else lo1 + 360, lo1 if lo1 > lo0 else lo1 + 360, lo0], [la0, la1, la1, la0, la0],
                color="m", lw=1, transform=ccrs.PlateCarree())
        ax.coastlines(lw=0.4, color="0.4")
        ax.set_title(str(times[k])[:13], fontsize=9)
    fig.colorbar(cf, ax=axes, shrink=0.6, label="Z500 minus zonal mean (m)")
    fig.suptitle(f"{name}: 500-hPa height (contours every 80 m) and its departure from the zonal mean; magenta: search area",
                 fontsize=10)
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / f"figures/events_{name}.png", dpi=110)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)
