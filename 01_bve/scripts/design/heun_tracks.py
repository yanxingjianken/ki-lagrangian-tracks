"""Cross-check tracks: the same patches advected with the hourly BVE wind by a Heun integrator (linear in time).

Writes data/heun_w<k>.npz in the layout of data/lagranto_w<k>.npz (time_h, lon, lat, patch, centres).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parcel_stats import cartesian, heun, lat_lon  # noqa: E402
from patches import seed_patches  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--window", type=int, default=0)
p.add_argument("--start-day", type=float, default=2.0)
p.add_argument("--days", type=float, default=6.5)
p.add_argument("--step-min", type=float, default=10.0)
p.add_argument("--out-min", type=float, default=10.0)
a = p.parse_args()

ds = xr.open_dataset(ROOT / "data/bve_stirred_jet.nc")
h0, nh = int(a.start_day * 24), int(a.days * 24)
uv = torch.tensor(np.stack([ds.u.values[h0:h0 + nh + 1], ds.v.values[h0:h0 + nh + 1]], 1), dtype=torch.float64,
                  device="cuda")
lat0, lon0, pid, centres = seed_patches(seed=a.window)
r = cartesian(torch.as_tensor(lat0, device="cuda")[None], torch.as_tensor(lon0, device="cuda")[None])
sub = int(round(60 / a.step_min))
every = int(round(a.out_min / a.step_min))
out, dt = [r], a.step_min * 60.0
for n in range(nh * sub):
    h, f0, f1 = n // sub, (n % sub) / sub, (n % sub + 1) / sub
    w0 = (1 - f0) * uv[h] + f0 * uv[h + 1]
    w1 = (1 - f1) * uv[h] + f1 * uv[h + 1]
    r = heun(w0[None], w1[None], r, dt)
    if (n + 1) % every == 0:
        out.append(r)
la, lo = lat_lon(torch.cat(out))
time_h = np.arange(len(out)) * a.out_min / 60.0
np.savez(ROOT / f"data/heun_w{a.window}.npz", time_h=time_h, lon=lo.cpu().numpy(), lat=la.cpu().numpy(), patch=pid,
         centres=centres, start_hour=h0)
print("tracks", la.shape, "patches", centres.round(1).tolist())
