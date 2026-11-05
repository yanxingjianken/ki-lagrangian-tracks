"""Hourly Dinosaur fields of one flow and window -> LAGRANTO 3-D P files (u, v, omega, ps, T on the sigma levels).

Writes data/lagranto/<flow>_w<k>/P<YYYYMMDD_HH> for hours [start, start + 156] of data/dino_<flow>.nc; window k starts
on day 2 + 6 k of the life cycle.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "lagranto/python"))
from lagranto_io import write_pfiles_3d  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--flow", required=True)
p.add_argument("--window", type=int, default=0)
p.add_argument("--hours", type=int, default=156)
a = p.parse_args()

ds = xr.open_dataset(ROOT / f"data/dino_{a.flow}.nc")
h0 = 24 * (2 + 6 * a.window)
assert ds.sizes["time"] >= h0 + a.hours + 1, f"{a.flow} has {ds.sizes['time']} hours, window needs {h0 + a.hours + 1}"
sub = ds.isel(time=slice(h0, h0 + a.hours + 1))
times = sub.time.values
if not np.issubdtype(times.dtype, np.datetime64):
    times = np.datetime64("2000-01-01T00") + np.round(times).astype("timedelta64[h]")
out = ROOT / f"data/lagranto/{a.flow}_w{a.window}"
files = write_pfiles_3d(out, times, ds.lon.values, ds.lat.values, ds.level.values, sub.u.values, sub.v.values,
                        sub.omega.values, sub.ps.values, t=sub.t.values)
(out / "window.txt").write_text(f"start_hour {h0}\nhours {a.hours}\nfirst {times[0]}\nlast {times[-1]}\n")
print(f"{len(files)} P files in {out}: {times[0]} .. {times[-1]}")
