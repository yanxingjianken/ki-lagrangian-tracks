"""Score the per-patch models on three abilities, in the training flow and in flows the models never saw.

Abilities (test parcels; times relative to the training window [t_start, t_end] after release):
fill a removed segment inside the window, forecast from t_end, backcast from t_start. Each patch's model is applied
to: its own patch (in distribution), the next patch of the same flow (another region), and the same patch position
and release time in other BVE runs (unseen perturbation; unseen jet and perturbation). The trained model's own
centre and scale are used everywhere. Writes outputs/eval/<data stem>_abilities.csv (one row per parcel and case).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ki_tracks import KINet, great_circle_km, infill, integrate, load_patch, stereo_to_lonlat, to_km  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--data", default=str(ROOT / "data/tracks_lagranto_w0.npz"))
p.add_argument("--ood", nargs="*", default=["new perturbation=data/tracks_lagranto_seed2_w0.npz",
                                            "new jet and perturbation=data/tracks_lagranto_jet40n_w0.npz"],
               help="label=path of tracks from other runs, same patch positions and release time")
p.add_argument("--models", default=str(ROOT / "outputs/models"))
p.add_argument("--patches", type=int, nargs="+", default=list(range(8)))
p.add_argument("--t-start-h", type=float, default=24.0)
p.add_argument("--t-end-h", type=float, default=120.0)
p.add_argument("--horizons-h", type=float, nargs="+", default=[6.0, 12.0, 24.0])
p.add_argument("--gaps-h", type=float, nargs="+", default=[6.0, 12.0, 24.0, 48.0])
p.add_argument("--gaps-per-parcel", type=int, default=3)
p.add_argument("--steps-per-hour", type=int, default=6)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
stem = Path(a.data).stem


class Zero(nn.Module):
    def forward(self, t, x, v):
        return torch.zeros_like(x)


def km(d, path, truth):
    (lp, ap), (lt, at) = stereo_to_lonlat(to_km(d, path)), stereo_to_lonlat(to_km(d, truth))
    return great_circle_km(lp, ap, lt, at), great_circle_km(lt[0], at[0], lt[-1], at[-1])


def score(net, d, rng, base):
    rows, pj = [], d.test
    tt = lambda arr, k: torch.tensor(arr[k, pj], dtype=torch.float32, device=dev)
    for L_h in a.gaps_h:
        L, n = L_h / d.clock, int(L_h * a.steps_per_hour)
        for _ in range(a.gaps_per_parcel):
            k0 = np.searchsorted(d.t, rng.uniform(0.0, 1.0 - L, len(pj)))
            k1 = np.searchsorted(d.t, d.t[k0] + L - 1e-9)
            path = infill(net, d, tt(d.x, k0), tt(d.v, k0), tt(d.x, k1), tt(d.v, k1), d.t[k0], d.t[k1], n).cpu().numpy()
            tq = d.t[k0][None] + (d.t[k1] - d.t[k0])[None] * np.linspace(0, 1, n + 1)[:, None]
            truth = np.stack([np.stack([np.interp(tq[:, j], d.t, d.x[:, q, c]) for c in (0, 1)], -1) for j, q in enumerate(pj)], 1)
            err, disp = km(d, path, truth)
            rows += [dict(base, ability="fill", horizon_h=L_h, parcel=int(q), error_km=float(err[:, j].mean()),
                          displacement_km=float(disp[j])) for j, q in enumerate(pj)]
    for ability, t0, sign in (("forecast", 1.0, 1), ("backcast", 0.0, -1)):
        k0 = int(np.argmin(np.abs(d.t_all - t0)))
        for H in a.horizons_h:
            k1 = int(np.argmin(np.abs(d.t_all - (d.t_all[k0] + sign * H / d.clock))))
            if not 0 <= k1 < len(d.t_all) or abs(abs(d.t_all[k1] - d.t_all[k0]) * d.clock - H) > 0.01:
                continue
            end = integrate(net, d, torch.tensor(d.x_all[k0, pj], dtype=torch.float32, device=dev),
                            torch.tensor(d.v_all[k0, pj], dtype=torch.float32, device=dev), d.t_all[k0], d.t_all[k1],
                            int(H * a.steps_per_hour))[-1].cpu().numpy()
            err, disp = km(d, np.stack([d.x_all[k0, pj], end]), d.x_all[[k0, k1]][:, pj])
            rows += [dict(base, ability=ability, horizon_h=H, parcel=int(q), error_km=float(err[-1, j]),
                          displacement_km=float(disp[j])) for j, q in enumerate(pj)]
    return rows


rows = []
for patch in a.patches:
    ck = torch.load(Path(a.models) / f"{stem}_p{patch}_s{int(a.t_start_h)}_t{int(a.t_end_h)}.pt", map_location=dev,
                    weights_only=False)
    g = ck["args"]
    net = KINet(width=g["width"], depth=g["depth"], freqs=g["freqs"]).to(dev)
    net.load_state_dict(ck["state"])
    cases = [("trained flow, same patch", a.data, patch), ("trained flow, other patch", a.data, (patch + 1) % 8)]
    cases += [(lab, str(ROOT / pth) if not Path(pth).is_absolute() else pth, patch)
              for lab, pth in (s.split("=", 1) for s in a.ood) if (ROOT / pth).exists() or Path(pth).exists()]
    for label, path, q in cases:
        d = load_patch(path, q, a.t_end_h, t_start_h=a.t_start_h, center=ck["center"], scale=ck["scale"])
        for mlabel, m in (("trained", net.eval()), ("zero acceleration", nn.ModuleList([Zero(), Zero()]))):
            rows += score(m, d, np.random.default_rng(100 + patch), dict(data=label, patch=patch, model=mlabel))
    print(f"patch {patch}: {len(cases)} cases", flush=True)
out = ROOT / "outputs/eval"
out.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(out / f"{stem}_abilities.csv", index=False)
pd.set_option("display.width", 200)
print(df.pivot_table(index=["ability", "horizon_h"], columns=["data", "model"], values="error_km", aggfunc="median").round(1))
