"""Life cycle: the basic state plus the normal mode scaled to `amp` hPa of maximum surface-pressure amplitude, integrated
for `days` days. Writes <out> with u, v, T on (time, level, lat, lon) and ps on (time, lat, lon), northern hemisphere
including the equator row, SI units, every `out_hours` hours; exits with code 3 if the state stops being finite.
Adapted from literature_review/rwb/thorncroft93_baroclinic/thm93_dinosaur/03_life_cycle.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import numpy as np
import xarray as xr
from dinosaur import primitive_equations as pe

from common import LAT, LON, SIGMA, make_step, nodal_fields, propagator, state_from_basic



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basic", type=Path, required=True)
    ap.add_argument("--mode", type=Path, required=True)
    ap.add_argument("--amp", type=float, default=1.0)
    ap.add_argument("--days", type=float, default=15.0)
    ap.add_argument("--out-hours", type=int, default=6)
    ap.add_argument("--dt", type=float, default=600.0, help="time step (s); diffusion e-folding stays 1 h at n = 95")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    base, t_ref = state_from_basic(xr.open_dataset(a.basic))
    mode = pe.State(**{k: v for k, v in np.load(a.mode).items()})
    x = jax.tree_util.tree_map(lambda b, m: b + a.amp * m, base, mode)
    advance = propagator(make_step(t_ref, dt_s=a.dt), a.out_hours * int(round(3600.0 / a.dt)))
    nh = LAT >= 0
    n_out = int(round(a.days * 24 / a.out_hours))
    fields = {k: [] for k in ("u", "v", "t", "ps")}
    for i in range(n_out + 1):
        f = nodal_fields(x, t_ref)
        if not (np.isfinite(f["ps"]).all() and np.isfinite(f["u"]).all()):
            print(f"blew up: non-finite state at hour {i * a.out_hours}", flush=True)
            sys.exit(3)
        for k in fields:
            arr = f[k][..., nh]
            fields[k].append(np.moveaxis(arr, -1, -2).astype(np.float32))
        if i < n_out:
            x = advance(x)

    coords = dict(time=np.arange(n_out + 1) * a.out_hours / 24.0, level=SIGMA, lat=LAT[nh], lon=LON)
    units = dict(u="m s-1", v="m s-1", t="K", ps="Pa")
    ds = xr.Dataset({k: ((("time", "lat", "lon") if k == "ps" else ("time", "level", "lat", "lon")),
                         np.stack(v), {"units": units[k]}) for k, v in fields.items()}, coords=coords,
                    attrs=dict(basic=str(a.basic), mode=str(a.mode), amp_hpa=a.amp, dt_s=a.dt))
    ds.time.attrs["units"] = "days"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(a.out)


if __name__ == "__main__":
    main()
