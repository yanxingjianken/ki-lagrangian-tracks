"""Hourly BVE wind of one window -> LAGRANTO P files (2-D case: the wind copied to two identical levels).

Writes data/lagranto/w<k>/P<YYYYMMDD_HH> for hours [start, start + days] of data/bve_stirred_jet.nc.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "lagranto/python"))
from lagranto_io import write_pfiles  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--window", type=int, default=0)
p.add_argument("--run", default="stirred_jet", help="data/bve_<run>.nc")
p.add_argument("--start-day", type=float, default=None, help="default: 2 + 7 * window")
p.add_argument("--days", type=float, default=6.5)
a = p.parse_args()

ds = xr.open_dataset(ROOT / f"data/bve_{a.run}.nc")
suffix = "" if a.run == "stirred_jet" else f"_{a.run}"
start_day = 2.0 + 7.0 * a.window if a.start_day is None else a.start_day
h0, nh = int(round(start_day * 24)), int(round(a.days * 24))
times = ds.time.values[h0:h0 + nh + 1]
assert len(times) == nh + 1, f"record has {ds.sizes['time']} hours, window needs {h0 + nh + 1}"
sub = ds.isel(time=slice(h0, h0 + nh + 1))
out = ROOT / f"data/lagranto/w{a.window}{suffix}"
files = write_pfiles(out, times, ds.lon.values, ds.lat.values, sub.u.values, sub.v.values)
(out / "window.txt").write_text(f"start_hour {h0}\nhours {nh}\nfirst {times[0]}\nlast {times[-1]}\n")
print(f"{len(files)} P files in {out}: {times[0]} .. {times[-1]}")
