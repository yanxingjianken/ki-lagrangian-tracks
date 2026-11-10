"""Patch releases of the track cloud of each event -> configs/releases.json.

Per event (configs/events.json: life-cycle window and region located on ERA5 500-hPa height by 01_events.py):
releases every 12 h from 4 days before the window to its end; patch centres about 1000 km apart over the region
(rows every 8 deg of latitude, longitude spacing 1000 km / (111 km cos lat)), which reaches 40 deg upstream (west) of
the block; 480 parcels in a 300-km disk per patch; tracks 157 h, as in 03_era5. Six start levels, all among the
training levels of 03_era5, in two releases per time: the upper one at sigma 0.38, 0.285, 0.20 (about 380, 285 and
200 hPa at sea-level pressure) and the lower one (`_lower`) at sigma 0.967, 0.85, 0.477 (near the surface, about 850
and 480 hPa). Each release carries its start levels and the seed of its random parcel positions.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIGMA = [0.967, 0.85, 0.477, 0.38, 0.285, 0.20]       # every start level, surface to top
SETS = {"": [0.38, 0.285, 0.20], "_lower": [0.967, 0.85, 0.477]}
EVERY_H, LEAD_D, WINDOW_H = 12, 4, 156


def centres(lat_range, lon_range):
    out = []
    lo0, lo1 = lon_range
    span = (lo1 - lo0) % 360 or 360
    for lat in np.arange(lat_range[0] + 4.0, lat_range[1] - 3.0, 8.0):
        step = 1000.0 / (111.2 * np.cos(np.deg2rad(lat)))
        n = max(int(span // step), 1)
        off = (span - (n - 1) * step) / 2 if n > 1 else span / 2
        out += [(round(float(lat), 2), round(float((lo0 + off + k * step) % 360), 2)) for k in range(n)]
    return out


if __name__ == "__main__":
    events = json.loads((ROOT / "configs/events.json").read_text())
    rel = []
    for name, ev in events.items():
        t0 = np.datetime64(ev["window"][0], "h") - np.timedelta64(24 * LEAD_D, "h")
        t1 = np.datetime64(ev["window"][1], "h")
        cen = centres(ev["region_lat"], ev["region_lon"])
        for j, (suffix, sig) in enumerate(SETS.items()):
            for k, t in enumerate(np.arange(t0, t1 + 1, np.timedelta64(EVERY_H, "h"))):
                rel.append(dict(name=f"{name}_r{k:03d}{suffix}", event=name, index=k, time=str(t),
                                end=str(t + np.timedelta64(WINDOW_H, "h")), centres=cen, sigma=sig,
                                seed=1000 * k + sum(map(ord, name)) + 500 * j))
        print(f"{name}: {sum(r['event'] == name for r in rel)} releases {str(t0)}..{str(t1)}, {len(cen)} centres x "
              f"{len(SIGMA)} levels")
    cfg = dict(sigma=SIGMA, n_per=480, radius_km=300.0, window_h=WINDOW_H, every_h=EVERY_H, releases=rel)
    (ROOT / "configs/releases.json").write_text(json.dumps(cfg, indent=1))
    n = sum(len(r["centres"]) * len(r["sigma"]) for r in rel)
    print(f"total {len(rel)} releases, {n} patches, {n * 480:.3g} parcels")
