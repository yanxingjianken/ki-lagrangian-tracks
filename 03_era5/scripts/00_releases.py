"""Release schedule -> configs/releases.json.

A release every 5 days from the start of each period, its hour cycling through 00, 06, 12 and 18 UTC; each release
is tracked for 157 h. Training period 2015-01-01 .. 2017-12-31, test year 2021. Every release has 20 start levels
(sigma x surface pressure) with 16 patches each (8 per hemisphere, 25-70 deg); test releases add 8 patches per level
at latitudes never used in training (per hemisphere two at 10-25 deg and two at 70-85 deg). 480 parcels per patch.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIGMA = [0.967, 0.93, 0.887, 0.85, 0.80, 0.75, 0.71, 0.674, 0.62, 0.57, 0.52, 0.477, 0.43, 0.38, 0.33, 0.285, 0.241,
         0.20, 0.15, 0.10]
PERIODS = {"train": ("2015-01-01T00", "2018-01-01T00"), "test": ("2021-01-01T00", "2022-01-01T00")}
# patch bands: (first patch id, count, hemisphere, latitude range)
BANDS = {"main": [(0, 8, 1, (25.0, 70.0)), (8, 8, -1, (25.0, 70.0))],
         "unseen_latitudes": [(16, 2, 1, (10.0, 25.0)), (18, 2, 1, (70.0, 85.0)),
                              (20, 2, -1, (10.0, 25.0)), (22, 2, -1, (70.0, 85.0))]}
WINDOW_H = 156
EVERY_D = 5


def releases():
    out = []
    for split, (start, end) in PERIODS.items():
        t0, t1 = np.datetime64(start, "h"), np.datetime64(end, "h")
        k = 0
        while True:
            t = t0 + np.timedelta64(24 * EVERY_D * k + 6 * (k % 4), "h")
            if t >= t1:
                break
            bands = BANDS["main"] + (BANDS["unseen_latitudes"] if split == "test" else [])
            out.append(dict(name=f"{split[:2]}{k:03d}", split=split, index=k, time=str(t),
                            end=str(t + np.timedelta64(WINDOW_H, "h")), bands=bands))
            k += 1
    return out


if __name__ == "__main__":
    rel = releases()
    cfg = dict(sigma=SIGMA, n_per=480, radius_km=300.0, min_sep_km=2000.0, window_h=WINDOW_H, every_days=EVERY_D,
               split_by_patch=dict(train=[0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13], validation=[6, 14], test=[7, 15],
                                   unseen_latitudes=list(range(16, 24))),
               releases=rel)
    (ROOT / "configs").mkdir(exist_ok=True)
    (ROOT / "configs/releases.json").write_text(json.dumps(cfg, indent=1))
    for s in PERIODS:
        r = [x for x in rel if x["split"] == s]
        print(f"{s}: {len(r)} releases, {r[0]['time']} .. {r[-1]['time']} (last window ends {r[-1]['end']})")
