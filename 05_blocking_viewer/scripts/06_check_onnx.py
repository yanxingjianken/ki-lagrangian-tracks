"""Check the ONNX graphs on real tracks and choose the in-browser settings (precision, members, steps per hour).

A NumPy + onnxruntime integrator that mirrors the page's JavaScript step by step (same token layout, centroid,
Euler-Maruyama update and score term) forecasts and backcasts 16 parcels of 2021 patches (03_era5 tracks, unseen
year); the median 24-h error of the ensemble mean is compared for: PyTorch (space/kimodel.py), ONNX fp32, ONNX with an
8-bit decoder, ONNX with 8-bit encoder and decoder; 16 or 8 members; 6 or 4 steps per hour. Also the time per launch
(both directions) on one CPU thread. Prints a table.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "space"))
OX = ROOT / "hf_model/onnx"
CFG = json.loads((OX / "config.json").read_text())
SC = CFG["scales"]
POS, VEL, ACC = (np.array(SC[k], np.float32) for k in ("pos_km", "vel_kmh", "acc_kmh2"))
TH = float(SC["time_h"])
SIGV = np.array(CFG["sx_km"], np.float32) * CFG["omega"] / CFG["delta_h"]


def sin_lat(x):
    r = np.hypot(x[..., 0], x[..., 1])
    return np.cos(2 * np.arctan(r / (2 * CFG["a_km"])))


def tokens(Xc, Vc, d, cen):
    n_t, k = Xc.shape[:2]
    tau = np.broadcast_to((-d * 3.0 * np.arange(n_t) / TH)[:, None, None], (n_t, k, 1))
    rel = np.concatenate([Xc[..., :2] - cen, Xc[..., 2:]], -1)
    tok = np.concatenate([tau, rel / POS, sin_lat(Xc)[..., None], Vc / VEL], -1)
    return tok.reshape(1, n_t * k, 8).astype(np.float32)


def ensemble_ort(enc, dec, tok, cen, x0, v0, d, members, steps_per_h, rng, eps_rel=CFG["eps_rel"]):
    z = enc.run(None, {"tokens": tok})[0]
    q = x0.shape[0]
    x = np.tile(x0, (members, 1)).astype(np.float32)
    v = np.tile(v0, (members, 1)).astype(np.float32)
    eps = eps_rel * SIGV
    dt = d / steps_per_h
    out = [x.copy()]
    for k in range(24 * steps_per_h):
        rel = np.concatenate([x[:, :2] - cen, x[:, 2:]], -1)
        qry = np.concatenate([np.full((len(x), 1), k * dt / TH), rel / POS, sin_lat(x)[:, None], v / VEL], -1)
        a, s = dec.run(None, {"latents": z, "queries": qry[None].astype(np.float32)})
        a, s = a[0] * ACC, s[0]
        noise = rng.standard_normal(x.shape).astype(np.float32)
        x, v = x + dt * v, v + dt * a + abs(dt) * 0.5 * eps**2 * s / SIGV + eps * np.sqrt(abs(dt)) * noise
        if (k + 1) % steps_per_h == 0:
            out.append(x.copy())
    return np.stack(out).reshape(25, members, q, 3)


def gc(xa, xb):
    from kimodel import stereo_to_lonlat
    (lo1, la1), (lo2, la2) = stereo_to_lonlat(xa), stereo_to_lonlat(xb)
    p1, p2 = np.deg2rad(la1), np.deg2rad(la2)
    return 6371 * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(np.deg2rad(lo2 - lo1)), -1, 1))


if __name__ == "__main__":
    import onnxruntime as ort
    import torch
    from kimodel import Model
    torch.set_num_threads(1)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = lambda f: ort.InferenceSession(str(OX / f), so, providers=["CPUExecutionProvider"])  # noqa: E731
    variants = {"onnx fp32": ("encoder.onnx", "decoder.onnx"), "onnx 8-bit decoder": ("encoder.onnx", "decoder.int8.onnx"),
                "onnx 8-bit both": ("encoder.int8.onnx", "decoder.int8.onnx")}
    S = {k: (sess(e), sess(d)) for k, (e, d) in variants.items()}
    M = Model(ROOT.parent / "03_era5/outputs/models/context_ki_era5.pt", "cpu")
    rng0 = np.random.default_rng(0)
    cases = []
    for name in ["te010", "te030", "te050", "te065"]:
        f = np.load(ROOT.parent / f"03_era5/data/tracks/tracks_{name}.npz")
        X, V, lev, pat = f["x_km"], f["v_kmh"], f["level"], f["patch"]
        for q in rng0.choice(8, 3, replace=False):
            for k in (12, 15, 18):
                idx = np.flatnonzero((lev == k) & (pat == q))
                cases.append((X, V, idx))
    configs = [("pytorch", 16, 6)] + [(v, 16, 6) for v in variants] + [("onnx fp32", 8, 6), ("onnx fp32", 16, 4), ("onnx fp32", 8, 4)]
    rows = []
    for var, mem, sph in configs:
        err, tt = {1: [], -1: []}, []
        rng = np.random.default_rng(1)
        for X, V, idx in cases:
            t_launch = 0.0
            for d, t0 in ((1, 72), (-1, 60)):
                p = np.random.default_rng(len(idx) + t0).permutation(idx)
                ctx, tgt = p[:192], p[192:208]
                rows_c = t0 - d * 3 * np.arange(17)
                cen = X[t0, idx, :2].mean(0)
                t1 = time.time()
                if var == "pytorch":
                    tok, c = M.context(X[rows_c][:, ctx], V[rows_c][:, ctx], d, cen)
                    ens = M.ensemble(tok, c, X[t0, tgt], V[t0, tgt], direction=d, members=mem, steps_per_h=sph)
                else:
                    tok = tokens(X[rows_c][:, ctx], V[rows_c][:, ctx], d, cen)
                    ens = ensemble_ort(*S[var], tok, cen, X[t0, tgt], V[t0, tgt], d, mem, sph, rng)
                t_launch += time.time() - t1
                err[d] += list(gc(ens[24].mean(0), X[t0 + 24 * d, tgt]))
            tt.append(t_launch)
        rows.append((var, mem, sph, np.median(err[1]), np.median(err[-1]), np.median(tt)))
        print(f"{var:20s} members {mem:2d} steps/h {sph}: 24-h ensemble-mean median error forecast {rows[-1][3]:5.0f} km, "
              f"backcast {rows[-1][4]:5.0f} km; launch (both directions, 1 thread) {rows[-1][5]:5.1f} s", flush=True)
