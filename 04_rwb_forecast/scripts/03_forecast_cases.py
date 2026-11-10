"""Forecast the upper-level patches of a year and measure how each forecast stretches the patch.

Cases: Northern-Hemisphere patches at start sigma 0.33, 0.285, 0.241 (levels 14-16) of the 03_era5 releases of one
year, patch ids not used in training (2015-2017: 6, 7; 2021: 0-7), anchors 48, 72 and 96 h after release. For each
case, 192 parcels observed over the 48 h before the anchor form the context and the other 288 parcels are forecast
24 h by (1) their true tracks, (2) straight motion at the anchor velocity, (3) the deterministic model and (4) the
stochastic ensemble (configs/ensemble.json). Patch shape at 0, 12, 24 h: covariance of the parcels' east/north
offsets from their centroid (local tangent plane), elongation E = sqrt(l1 / l2) and long-axis angle alpha from east
in (-90, 90] deg (alpha > 0: SW-NE). Also stored: the true centroid track (hourly). Writes outputs/cases_<year>.npz.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import stereo_to_lonlat  # noqa: E402
from kiforecast import Forecaster, load_patches  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
H = (0, 12, 24)


def shape(lon, lat):
    """lon, lat (..., N) deg -> elongation E, angle alpha (deg) of the parcel cloud, centroid lon/lat."""
    lat0 = lat.mean(-1, keepdims=True)
    lon0 = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(lon)).mean(-1, keepdims=True)))
    dx = 6371.0 * np.cos(np.deg2rad(lat0)) * np.deg2rad((lon - lon0 + 180.0) % 360.0 - 180.0)
    dy = 6371.0 * np.deg2rad(lat - lat0)
    cxx, cyy, cxy = (dx * dx).mean(-1), (dy * dy).mean(-1), (dx * dy).mean(-1)
    tr, det = cxx + cyy, cxx * cyy - cxy**2
    l1 = tr / 2 + np.sqrt(np.maximum(tr**2 / 4 - det, 0))
    l2 = np.maximum(tr - l1, 1e-6)
    alpha = np.rad2deg(0.5 * np.arctan2(2 * cxy, cxx - cyy))       # long axis, in (-90, 90]
    return np.sqrt(l1 / l2), alpha, lon0[..., 0] % 360.0, lat0[..., 0]


def lonlat(x):
    lon, lat = stereo_to_lonlat(x[..., :2])
    return lon, lat


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("year", type=int)
    p.add_argument("--anchors", type=int, nargs="+", default=[48, 72, 96])
    p.add_argument("--chunk", type=int, default=24)
    a = p.parse_args()
    cfg = json.loads((ROOT.parent / "03_era5/configs/releases.json").read_text())
    ens = json.loads((ROOT / "configs/ensemble.json").read_text())
    names = [r["name"] for r in cfg["releases"] if r["time"].startswith(str(a.year))]
    ids = [6, 7] if a.year <= 2017 else list(range(8))
    F = Forecaster()
    res = {k: [] for k in ("E", "alpha", "clon", "clat", "track_lon", "track_lat", "anchor_time", "release",
                           "level", "patch", "anchor_h")}
    g = torch.Generator(device="cuda").manual_seed(a.year)
    for name in names:
        X, V, LAT, P, meta = load_patches([name], ids, levels=[14, 15, 16])
        keep = [i for i, m in enumerate(meta) if m["hemisphere"] == 1]
        X, V, meta = X[keep], V[keep], [meta[i] for i in keep]
        rt = np.datetime64(meta[0]["release_time"], "h")
        for t0h in a.anchors:
            n = X.shape[0]
            perm = torch.rand(n, X.shape[2], device="cuda", generator=g).argsort(1)
            kidx, qidx = perm[:, :192], perm[:, 192:]
            t0 = torch.full((n,), t0h, device="cuda")
            b = torch.arange(n, device="cuda")[:, None]
            x0, v0 = X[b, t0[:, None], qidx], V[b, t0[:, None], qidx]
            tru = X[b[:, :, None], (t0[:, None] + torch.arange(25, device="cuda"))[:, :, None], qidx[:, None]]  # (n, 25, Q, 3)
            tru = tru.transpose(0, 1).cpu().numpy()
            straight = np.stack([(x0 + h * v0).cpu().numpy() for h in range(25)])
            ode, ensm = [], []
            for s in range(0, n, a.chunk):
                sl = slice(s, s + a.chunk)
                tok, cen = F.context(X[sl], V[sl], t0[sl], kidx[sl])
                ode.append(F.ode(tok, cen, x0[sl], v0[sl]).cpu().numpy())
                ensm.append(F.ensemble(tok, cen, x0[sl], v0[sl], members=ens["members"], eps_rel=ens["eps_rel"],
                                       gen=g).cpu().numpy())
            ode, ensm = np.concatenate(ode, 1), np.concatenate(ensm, 1)     # (25, n, Q, 3), (25, n, M, Q, 3)
            E, AL, CL, CA = [], [], [], []
            for h in H:
                fields = [tru[h], straight[h], ode[h], ensm[h].mean(1)] + [ensm[h][:, m] for m in range(ensm.shape[2])]
                sh = [shape(*lonlat(f)) for f in fields]
                E.append([s[0] for s in sh]); AL.append([s[1] for s in sh]); CL.append(sh[0][2]); CA.append(sh[0][3])
            # E, alpha: (3 horizons, 4 + M predictors, n) -> stored as (n, 3, 4 + M)
            res["E"].append(np.transpose(np.array(E), (2, 0, 1))); res["alpha"].append(np.transpose(np.array(AL), (2, 0, 1)))
            tl, ta = lonlat(tru.mean(2))                                      # true centroid track (25, n)
            res["track_lon"].append(tl.T); res["track_lat"].append(ta.T)
            res["clon"].append(tl[0]); res["clat"].append(ta[0])
            at = rt + np.timedelta64(t0h, "h")
            res["anchor_time"] += [str(at)] * n
            res["release"] += [name] * n
            res["level"] += [m["level"] for m in meta]
            res["patch"] += [m["patch"] for m in meta]
            res["anchor_h"] += [t0h] * n
        print(f"{name} {meta[0]['release_time']}: {X.shape[0]} patches x {len(a.anchors)} anchors", flush=True)
    out = {k: (np.concatenate(v) if k in ("E", "alpha", "clon", "clat", "track_lon", "track_lat") else np.array(v))
           for k, v in res.items()}
    out["predictors"] = np.array(["truth", "straight", "model", "ensemble mean"] + [f"member {m}" for m in range(ens["members"])])
    out["horizons_h"] = np.array(H)
    np.savez(ROOT / f"outputs/cases_{a.year}.npz", **out)
    print(f"{len(out['release'])} cases -> outputs/cases_{a.year}.npz")
