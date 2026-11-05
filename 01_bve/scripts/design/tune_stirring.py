"""Stirring-amplitude sweep: spin up the stirred jet, track parcels, report Lagrangian scales and the eddy decorrelation time.

Writes outputs/design/tune_stirring.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parcel_stats import cartesian, heun, lagrangian_scales, seed_equal_area  # noqa: E402
from stirred_jet import RegularGrid, StirredJet  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--amps", type=float, nargs="+", default=[1e-10, 2e-10, 4e-10, 8e-10])
p.add_argument("--u0", type=float, default=40.0)
p.add_argument("--spinup-days", type=float, default=30.0)
p.add_argument("--track-days", type=float, default=6.0)
p.add_argument("--n-parcels", type=int, default=500)
p.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "outputs/design/tune_stirring.json"))
a = p.parse_args()

sj = StirredJet(a.amps, u0=a.u0)
grid = RegularGrid(sj.m)
spd = int(round(86400 / sj.m.dt))
sj.run(int(a.spinup_days * spd))

lat0, lon0 = seed_equal_area(a.n_parcels, 25.0, 70.0)
r = cartesian(torch.as_tensor(lat0, device=sj.device)[None].repeat(sj.B, 1),
              torch.as_tensor(lon0, device=sj.device)[None].repeat(sj.B, 1))
uv = grid.uv(sj.zeta)
track, zeta_h = [r.cpu().numpy()], []
for n in range(int(a.track_days * spd)):
    sj.step()
    uv_new = grid.uv(sj.zeta)
    r = heun(uv, uv_new, r, sj.m.dt)
    uv = uv_new
    track.append(r.cpu().numpy())
    if (n + 1) % (spd // 24) == 0:
        zeta_h.append(grid.zeta(sj.zeta)[:, 30:66].cpu().numpy())
track = np.asarray(track)
zeta_h = np.asarray(zeta_h)
speed, turn = lagrangian_scales(track, sj.m.dt / 3600.0)
uv_np = uv.cpu().numpy()
rows = []
for b, amp in enumerate(a.amps):
    z = zeta_h[:, b] - zeta_h[:, b].mean(axis=(0, 2), keepdims=True)
    lags = np.arange(0, min(96, len(z) - 1))
    acf = np.array([np.mean(z[: len(z) - L] * z[L:]) for L in lags]) / np.mean(z * z)
    below = np.where(acf < np.exp(-1))[0]
    rows.append(dict(amp=amp, u0=a.u0, speed_median=float(np.median(speed[:, b])),
                     speed_p90=float(np.percentile(speed[:, b], 90)),
                     turn_median_h=float(np.median(turn[:, b])), turn_p10_h=float(np.percentile(turn[:, b], 10)),
                     zonal_u_max=float(uv_np[b, 0].mean(-1).max()),
                     eddy_rms_speed_30_60N=float(np.sqrt(((uv_np[b, :, 30:61] - uv_np[b, :, 30:61].mean(-1, keepdims=True)) ** 2).sum(0).mean())),
                     zeta_efold_h=float(lags[below[0]]) if len(below) else None))
    print(rows[-1], flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(rows, indent=1))
