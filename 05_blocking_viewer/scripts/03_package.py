"""Package one event's track cloud and maps for the Hugging Face dataset read by the Space and the page.

From data/tracks/tracks_<release>.npz (12_dataset.py; hourly x, v of every parcel) of the event's releases:
  pkg/<event>/tracks.npy   float16 (P, 53, 480, 6): x, y, z (km) and u, v, w (km/h) of the 480 parcels of each patch
                           every 3 h from release to 156 h (memory-mappable);
  pkg/<event>/centroids.bin float32 (P, 53, 2): latitude and longitude of each patch's centroid every 3 h (the
                           page picks candidate patches from these before fetching one patch's rows of tracks.npy);
  pkg/<event>/index.json   per patch row: release name and time, start level (sigma index), patch centre;
                           event label, window, region, basis, sigma, the hours stored;
  pkg/<event>/z500.bin     int16 metres (2, T, ny, nx): ERA5 500-hPa height and its departure from the zonal mean
                           (global, same latitude and time), every 6 h at 1 deg over the region (latitude ascending,
                           longitude from 20 deg west of the region eastward), with z500.json (times, lat, lon, shape)
                           — the page's map background.
Positions are north-polar stereographic x, y and z = -7 km ln(p / 1000 hPa); parcels that left the domain or started
below the ground (none expected) are filled by repeating others so every patch has 480.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
HOURS = np.arange(0, 157, 3)


def z500_maps(ev, t0, t1, out):
    import importlib.util
    spec = importlib.util.spec_from_file_location("events", Path(__file__).resolve().parent / "01_events.py")
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    times, lat, lon, z = E.z500(t0, t1)
    (la0, la1), (lo0, lo1) = ev["region_lat"], ev["region_lon"]
    la0, la1 = max(la0 - 15, 0), min(la1 + 10, 90)
    span = (lo1 - lo0) % 360 or 360
    lons = (lo0 - 20 + np.arange(int(span) + 41)) % 360
    jl = (lat >= la0) & (lat <= la1)
    il = np.searchsorted(lon, lons) % len(lon)
    anom = z - z.mean(axis=2, keepdims=True)
    sub = np.stack([z[:, jl][:, :, il], anom[:, jl][:, :, il]])
    sub.round().astype(np.int16).tofile(out / "z500.bin")
    (out / "z500.json").write_text(json.dumps(dict(times=[str(t)[:13] for t in times], lat=lat[jl].tolist(),
                                                   lon=lons.tolist(), shape=list(sub.shape), units="m",
                                                   fields=["500-hPa height", "500-hPa height minus its zonal mean"])))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("event")
    p.add_argument("--no-maps", action="store_true")
    p.add_argument("--limit", type=int, help="first N releases only, into pkg_test/ (tests)")
    a = p.parse_args()
    cfg = json.loads((ROOT / "configs/releases.json").read_text())
    ev = json.loads((ROOT / "configs/events.json").read_text())[a.event]
    rels = [r for r in cfg["releases"] if r["event"] == a.event][: a.limit]
    out = ROOT / ("pkg_test" if a.limit else "pkg") / a.event
    out.mkdir(parents=True, exist_ok=True)
    rows, n_per = [], cfg["n_per"]
    n_p = len(rels) * len(rels[0]["centres"]) * len(cfg["sigma"])
    arr = np.lib.format.open_memmap(out / "tracks.npy", mode="w+", dtype=np.float16, shape=(n_p, len(HOURS), n_per, 6))
    rng, i = np.random.default_rng(0), 0
    for r in rels:
        with np.load(ROOT / f"data/tracks/tracks_{r['name']}.npz") as f:
            x, v, level, patch, th = f["x_km"], f["v_kmh"], f["level"], f["patch"], f["time_h"]
        sel = np.searchsorted(th, HOURS)
        assert np.allclose(th[sel], HOURS)
        for k in range(len(cfg["sigma"])):
            for q, (clat, clon) in enumerate(r["centres"]):
                idx = np.flatnonzero((level == k) & (patch == q))
                if len(idx) < n_per:
                    idx = np.r_[idx, rng.choice(idx, n_per - len(idx))]
                arr[i, ..., :3] = x[sel][:, idx[:n_per]]
                arr[i, ..., 3:] = v[sel][:, idx[:n_per]]
                rows.append(dict(row=i, release=r["name"], release_time=r["time"], level=k, patch=q,
                                 centre=[clat, clon]))
                i += 1
    arr.flush()
    assert i == n_p
    xy = np.asarray(arr[..., :2], np.float32).mean(axis=2)          # (P, 53, 2) stereographic centroid
    r = np.hypot(xy[..., 0], xy[..., 1])
    clat = 90.0 - np.rad2deg(2 * np.arctan(r / (2 * 6371.0)))
    clon = np.rad2deg(np.arctan2(xy[..., 0], -xy[..., 1])) % 360.0
    np.stack([clat, clon], -1).astype(np.float32).tofile(out / "centroids.bin")
    (out / "index.json").write_text(json.dumps(dict(
        event=a.event, label=ev["label"], window=ev["window"], region_lat=ev["region_lat"], region_lon=ev["region_lon"],
        basis=ev["basis"], sigma=cfg["sigma"], hours=HOURS.tolist(), n_per=n_per, row_shape=[len(HOURS), n_per, 6],
        patches=rows)))
    if not a.no_maps:
        t0 = np.datetime64(rels[0]["time"], "h") + np.timedelta64(24, "h")
        t1 = np.datetime64(rels[-1]["time"], "h") + np.timedelta64(132, "h")
        z500_maps(ev, str(t0), str(t1), out)
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e9
    print(f"{a.event}: {n_p} patches from {len(rels)} releases -> {out} ({size:.2f} GB)")
