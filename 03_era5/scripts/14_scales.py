"""Per-axis scales of the model's velocities and accelerations from the ERA5 training tracks -> configs/scales.json.

The model normalizes inputs and targets per axis (x, y, z). Each velocity and acceleration scale is the standard
deviation of that component after capping its largest 0.5 % (winsorized), over 8000 random parcels from each of
--n-releases training releases spread over 2015-2017. Hourly vertical accelerations from 0.25-degree winds are
dominated by short fluctuations; with a scale of the size of the mean vertical acceleration they would outweigh
the horizontal ones by two orders of magnitude in the loss, so each axis gets its own spread. Positions keep fixed
scales: 1000 km for the horizontal offset from the patch centroid and 10 km for the absolute height z.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctx_data import release_names  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--n-releases", type=int, default=10)
p.add_argument("--per-release", type=int, default=8000)
p.add_argument("--cap-percent", type=float, default=0.5)
a = p.parse_args()

names = release_names(ROOT, "train")
pick = [names[i] for i in np.linspace(0, len(names) - 1, a.n_releases).round().astype(int)]
rng = np.random.default_rng(0)
V, A = [], []
for name in pick:
    with np.load(ROOT / f"data/tracks/tracks_{name}.npz") as f:
        idx = np.sort(rng.choice(f["patch"].size, a.per_release, replace=False))
        V.append(f["v_kmh"][:, idx].reshape(-1, 3))
        A.append(f["a_kmh2"][:, idx].reshape(-1, 3))
V, A = np.concatenate(V), np.concatenate(A)


def capped_std(x):
    c = np.percentile(np.abs(x), 100.0 - a.cap_percent, axis=0)
    return np.clip(x, -c, c).std(axis=0)


vel, acc = capped_std(V), capped_std(A)
n = A / acc
out = dict(pos_km=[1000.0, 1000.0, 10.0], vel_kmh=[float(f"{s:.4g}") for s in vel],
           acc_kmh2=[float(f"{s:.4g}") for s in acc], time_h=24.0,
           method=f"standard deviation after capping the largest {a.cap_percent} % of each component; "
                  f"{a.per_release} random parcels x 157 h from each release",
           releases=pick, check=dict(zero_predictor_mse=[round(float(m), 3) for m in (np.clip(n, -30, 30) ** 2).mean(0)],
                                     fraction_beyond_30=[float(f"{q:.2e}") for q in (np.abs(n) > 30).mean(0)]))
(ROOT / "configs/scales.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
