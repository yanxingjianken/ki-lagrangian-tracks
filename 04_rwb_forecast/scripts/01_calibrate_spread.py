"""Choose the ensemble noise eps_rel so that the spread matches the error of the ensemble mean (training years).

Validation patches (ids 6 and 14) of every 11th 2015-2017 release, all start levels; anchor 48 h after release
(context 0-48 h, 192 parcels every 3 h), 64 other parcels forecast 24 h, 16 members per eps_rel. At 12 and 24 h:
median and rms great-circle error of the ensemble mean, and the spread (rms distance of the members from their
mean). The eps_rel whose 24-h spread / rms error is closest to 1 goes to configs/ensemble.json; the deterministic
forecast (no noise) is listed for reference. Writes outputs/calibration.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import great_circle_km, stereo_to_lonlat  # noqa: E402
from kiforecast import Forecaster, load_patches  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--eps", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
p.add_argument("--members", type=int, default=16)
p.add_argument("--max-patches", type=int, default=320)
p.add_argument("--chunk", type=int, default=32)
a = p.parse_args()

cfg = json.loads((ROOT.parent / "03_era5/configs/releases.json").read_text())
names = [r["name"] for r in cfg["releases"] if r["split"] == "train"][::11]
F = Forecaster()
X, V, LAT, P, meta = load_patches(names, cfg["split_by_patch"]["validation"], max_patches=a.max_patches)
n = X.shape[0]
g = torch.Generator(device="cuda").manual_seed(0)
perm = torch.rand(n, X.shape[2], device="cuda", generator=g).argsort(1)
kidx, qidx = perm[:, :192], perm[:, 192:256]
t0 = torch.full((n,), 48, device="cuda")
b = torch.arange(n, device="cuda")[:, None]
x0, v0 = X[b, t0[:, None], qidx], V[b, t0[:, None], qidx]
truth = {h: X[b, (t0 + h)[:, None], qidx].cpu().numpy() for h in (12, 24)}


def km(xa, xb):
    (la, pa), (lb, pb) = stereo_to_lonlat(xa[..., :2]), stereo_to_lonlat(xb[..., :2])
    return great_circle_km(la, pa, lb, pb)


rows = []
for eps in [0.0] + a.eps:
    paths = []
    for s in range(0, n, a.chunk):
        sl = slice(s, s + a.chunk)
        tok, cen = F.context(X[sl], V[sl], t0[sl], kidx[sl])
        if eps == 0.0:
            paths.append(F.ode(tok, cen, x0[sl], v0[sl])[:, :, None].cpu().numpy())
        else:
            paths.append(F.ensemble(tok, cen, x0[sl], v0[sl], members=a.members, eps_rel=eps, gen=g).cpu().numpy())
    path = np.concatenate(paths, axis=1)                      # (H + 1, n, M, Q, 3)
    for h in (12, 24):
        mean = path[h].mean(1)
        err = km(mean, truth[h])
        spread = np.sqrt((km(path[h], mean[:, None]) ** 2).mean(1)) if path.shape[2] > 1 else np.zeros_like(err)
        upper = np.array([m["level"] >= 14 for m in meta])[:, None].repeat(err.shape[1], 1)
        rows.append(dict(eps_rel=eps, horizon_h=h, median_err_km=np.median(err), rms_err_km=np.sqrt((err**2).mean()),
                         spread_km=np.sqrt((spread**2).mean()), ratio=np.sqrt((spread**2).mean() / (err**2).mean()),
                         upper_rms_err_km=np.sqrt((err[upper] ** 2).mean()),
                         upper_ratio=np.sqrt((spread[upper] ** 2).mean() / (err[upper] ** 2).mean())))
    print(pd.DataFrame(rows[-2:]).round(2).to_string(index=False, header=eps == 0.0), flush=True)
df = pd.DataFrame(rows)
(ROOT / "outputs").mkdir(exist_ok=True)
df.to_csv(ROOT / "outputs/calibration.csv", index=False)
d24 = df[(df.horizon_h == 24) & (df.eps_rel > 0)]
best = float(d24.iloc[(d24.ratio - 1).abs().argmin()].eps_rel)
(ROOT / "configs/ensemble.json").write_text(json.dumps(dict(
    eps_rel=best, members=a.members, chosen_on=f"{n} validation patches of releases {names[0]}..{names[-1]} (every 11th)",
    rule="24-h spread closest to the rms error of the ensemble mean"), indent=1))
print("chosen eps_rel", best)
