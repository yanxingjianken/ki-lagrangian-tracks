"""Launch logic of the Space: pick tracked parcels near a point and forecast/backcast them 24 h as an ensemble.

For the chosen event, start level and time t (on the 3-hourly grid), the patch of that level whose parcels at t lie
nearest the point is found among those released 48-132 h before t (forecast: 48 h of observed tracks before t) and
among those released 24-108 h before t (backcast: 48 h after t). The 16 parcels of the patch nearest the point are
the ones launched; 192 other parcels of the patch, observed every 3 h over the 48 h, are the model's context; the
patch centroid at t is the reference position (as in training). Each direction returns the ensemble members and mean
(members every 3 h, mean hourly) and the true ERA5 tracks of the same parcels (every 3 h), as longitude, latitude
and pressure.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from kimodel import Model, stereo_to_lonlat

DATA_REPO = os.environ.get("DATA_REPO", "yanxingjianken/blocking-tracks")
MODEL_REPO = os.environ.get("MODEL_REPO", "yanxingjianken/ki-era5-tracks")
LOCAL_DATA = os.environ.get("LOCAL_DATA")          # a local copy of the dataset (tests); else the Hub
LOCAL_MODEL = os.environ.get("LOCAL_MODEL")
N_LAUNCH, N_CONTEXT, NEAR_KM, MIN_NEAR = 16, 192, 300.0, 16
H_KM = 7.0


def _fetch(repo, name, repo_type):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, name, repo_type=repo_type, token=os.environ.get("HF_TOKEN"))


def gc_km(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon2 - lon1)
    return 6371.0 * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1.0, 1.0))


class Launcher:
    def __init__(self, device="cpu"):
        mpath = LOCAL_MODEL or _fetch(MODEL_REPO, "context_ki_era5.pt", "model")
        epath = (Path(LOCAL_MODEL).with_name("ensemble.json") if LOCAL_MODEL else _fetch(MODEL_REPO, "ensemble.json", "model"))
        self.model = Model(mpath, device)
        self.eps_rel = float(json.loads(Path(epath).read_text())["eps_rel"])
        self.events = {}

    def event(self, name):
        if name not in self.events:
            get = (lambda f: str(Path(LOCAL_DATA) / name / f)) if LOCAL_DATA else (lambda f: _fetch(DATA_REPO, f"{name}/{f}", "dataset"))
            idx = json.loads(Path(get("index.json")).read_text())
            arr = np.load(get("tracks.npy"), mmap_mode="r")
            rel = np.array([np.datetime64(p["release_time"], "h") for p in idx["patches"]])
            lev = np.array([p["level"] for p in idx["patches"]])
            self.events[name] = (idx, arr, rel, lev)
        return self.events[name]

    def launch(self, event, time_utc, level, lat, lon, seed=0):
        idx, arr, rel, lev = self.event(event)
        t = np.datetime64(str(time_utc)[:13], "h")
        level, lat, lon = int(level), float(lat), float(lon) % 360.0
        out = dict(event=event, time=str(t), level=level, sigma=idx["sigma"][level], lat=lat, lon=lon, directions={})
        rng = np.random.default_rng(seed)
        for name, d, (a0, a1) in (("forecast", 1, (48, 132)), ("backcast", -1, (24, 108))):
            age = (t - rel).astype(int)
            cand = np.flatnonzero((lev == level) & (age >= a0) & (age <= a1) & (age % 3 == 0))
            best = None
            for c in cand:
                lo, la = stereo_to_lonlat(np.asarray(arr[c, age[c] // 3, :, :2], np.float32))
                dist = gc_km(la, lo, lat, lon)
                n = int((dist < NEAR_KM).sum())
                if best is None or (n, -dist.min()) > (best[0], -best[2].min()):
                    best = (n, c, dist)
            if best is None or best[0] < MIN_NEAR:
                out["directions"][name] = dict(error=f"fewer than {MIN_NEAR} tracked parcels within {NEAR_KM:.0f} km of "
                                                     f"this point at this time and level; try a point closer to the flow "
                                                     f"or another time")
                continue
            n, c, dist = best
            row = age[c] // 3
            P = np.asarray(arr[c], np.float32)                       # (53, 480, 6): x, y, z, u, v, w
            tgt = np.argsort(dist)[:N_LAUNCH]
            ctx = rng.choice(np.setdiff1d(np.arange(P.shape[1]), tgt), N_CONTEXT, replace=False)
            rows = row - d * np.arange(17)
            tok, cen = self.model.context(P[rows][:, ctx, :3], P[rows][:, ctx, 3:], d, P[row, :, :2].mean(0))
            ens = self.model.ensemble(tok, cen, P[row, tgt, :3], P[row, tgt, 3:], direction=d, eps_rel=self.eps_rel,
                                      seed=seed)                    # (25, M, Q, 3)
            truth = P[row + d * np.arange(9)][:, tgt, :3]            # every 3 h, 0..24 h
            mean = ens.mean(1)
            lo_t, la_t = stereo_to_lonlat(truth[-1])
            lo_m, la_m = stereo_to_lonlat(mean[-1])
            err = gc_km(la_t, lo_t, la_m, lo_m)
            out["directions"][name] = dict(
                release=idx["patches"][c]["release"], release_time=idx["patches"][c]["release_time"],
                hours_since_release=int(age[c]), parcels_within_km=n, near_km=NEAR_KM,
                hours=(d * np.arange(25)).tolist(), member_hours=(d * 3 * np.arange(9)).tolist(),
                truth_hours=(d * 3 * np.arange(9)).tolist(),
                members=_llp(ens[::3]), mean=_llp(mean), truth=_llp(truth),
                median_error_24h_km=round(float(np.median(err)), 1))
        return out


def _llp(x):
    """(..., 3) stereographic x, y, z -> nested lists [lon, lat, p hPa] rounded."""
    lon, lat = stereo_to_lonlat(x[..., :2])
    p = 1000.0 * np.exp(-x[..., 2] / H_KM)
    a = np.stack([np.round(lon, 2), np.round(lat, 2), np.round(p, 1)], -1)
    return a.tolist()
