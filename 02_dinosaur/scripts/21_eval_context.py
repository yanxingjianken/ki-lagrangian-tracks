# Adapted from roger/01_bve/scripts/09_eval_context.py (three-dimensional tracks, start levels).
"""Score the 3-D conditional KI: 6-24 h conditional forecasts and backcasts of target parcels not in the context.

Cases: unseen patches of the training flows (patch 7 of every window and start level) and every patch of each unseen
flow. For each patch and anchor (48, 72, 96 h into the window) both directions are scored; the context is 192 other
parcels of the patch over the 48 h on the observed side. Horizontal error: great-circle km; vertical: hPa. Writes
outputs/eval/context_ki_3d.csv.gz (one row per target parcel, direction, anchor, horizon and model).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctx_data import context, load_patches, read_flows, rollout  # noqa: E402
from ctx_model import ContextKI  # noqa: E402
from geo import great_circle_km, stereo_to_lonlat  # noqa: E402

H_KM = 7.0
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--flows", default=str(ROOT / "configs/flows.json"))
p.add_argument("--model", default=str(ROOT / "outputs/models/context_ki_3d.pt"))
p.add_argument("--windows", type=int, nargs="+", default=[0, 1])
p.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4])
p.add_argument("--anchors-h", type=int, nargs="+", default=[48, 72, 96])
p.add_argument("--horizons-h", type=int, nargs="+", default=[6, 12, 24])
p.add_argument("--targets", type=int, default=64)
p.add_argument("--chunk", type=int, default=48)
a = p.parse_args()
dev = "cuda"
ck = torch.load(a.model, map_location=dev, weights_only=False)
g = ck["args"]
model = ContextKI(d=g["d_model"]).to(dev).eval()
model.load_state_dict(ck["state"])
ok = read_flows(a.flows)
cases = [("trained flows, unseen patches", [f["name"] for f in ok if f["split"] == "train"], [7])]
cases += [(f.get("label", f["name"]), [f["name"]], list(range(8))) for f in ok if f["split"] == "test"]


class Straight(torch.nn.Module):
    def encode(self, tok):
        return None

    def decode(self, z, q):
        return torch.zeros_like(q[..., 1:4]), None


rows = []
for label, names, patches in cases:
    D = load_patches(ROOT, names, a.windows, a.levels, patches, device=dev)
    nP, H = D.X.shape[0], max(a.horizons_h)
    gen = torch.Generator(device=dev).manual_seed(7)
    pp, dd, tt = (torch.tensor(z, device=dev).flatten() for z in np.meshgrid(
        np.arange(nP), [1, -1], a.anchors_h, indexing="ij"))
    perm = torch.rand(len(pp), D.X.shape[2], device=dev, generator=gen).argsort(1)
    kidx, qidx = perm[:, : g["context_parcels"]], perm[:, g["context_parcels"]: g["context_parcels"] + a.targets]
    for s in range(0, len(pp), a.chunk):
        sl = slice(s, s + a.chunk)
        p_, d_, t_, q_ = pp[sl], dd[sl], tt[sl], qidx[sl]
        tok, cen = context(D, p_, t_, d_, kidx[sl], g["span_h"])
        x0, v0 = D.X[p_[:, None], t_[:, None], q_], D.V[p_[:, None], t_[:, None], q_]
        steps = t_[:, None] + d_[:, None] * torch.arange(H + 1, device=dev)
        truth = D.X[p_[:, None, None], steps[:, :, None], q_[:, None]].transpose(0, 1).cpu().numpy()
        for mlabel, m in (("conditional KI", model), ("no learning (straight motion)", Straight())):
            path = rollout(m, tok, cen, x0, v0, d_, H).cpu().numpy()
            for h in a.horizons_h:
                (lp, ap), (lt, at) = stereo_to_lonlat(path[h][..., :2]), stereo_to_lonlat(truth[h][..., :2])
                err = great_circle_km(lp, ap, lt, at)
                disp = great_circle_km(*stereo_to_lonlat(truth[0][..., :2]), lt, at)
                pres = lambda z: 1000.0 * np.exp(-z / H_KM)
                dp = pres(path[h][..., 2]) - pres(truth[h][..., 2])
                dp_true = pres(truth[h][..., 2]) - pres(truth[0][..., 2])
                for i in range(len(p_)):
                    name, w, lev, patch = D.meta[int(p_[i])]
                    rows += [dict(case=label, flow=name, window=w, level=lev, patch=patch, anchor_h=int(t_[i]),
                                  task="forecast" if d_[i] > 0 else "backcast", horizon_h=h, model=mlabel,
                                  parcel=int(q), error_km=float(e), displacement_km=float(dk), error_hpa=float(ep),
                                  dp_true_hpa=float(dq))
                             for q, e, dk, ep, dq in zip(q_[i].tolist(), err[i], disp[i], dp[i], dp_true[i])]
    print(f"{label}: {nP} patches", flush=True)
out = ROOT / "outputs/eval"
out.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(out / "context_ki_3d.csv.gz", index=False)
pd.set_option("display.width", 220)
print(df.pivot_table(index=["task", "horizon_h"], columns=["case", "model"], values="error_km", aggfunc="median").round(0))
df["abs_error_hpa"] = df.error_hpa.abs()
print(df.pivot_table(index=["task", "horizon_h"], columns=["case", "model"], values="abs_error_hpa", aggfunc="median").round(1))
