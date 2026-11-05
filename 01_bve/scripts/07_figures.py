"""Result figures from outputs/eval/<stem>_abilities.csv.

figures/f1_abilities.png: fill, forecast and backcast errors against horizon in the training flow.
figures/f2_unseen_flows.png: the same models on another region of the flow and on unseen perturbations and jets.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--stem", default="tracks_lagranto_w0")
p.add_argument("--horizon-ood", type=float, default=24.0)
a = p.parse_args()
df = pd.read_csv(ROOT / f"outputs/eval/{a.stem}_abilities.csv")
(ROOT / "figures").mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8.5})
ID = "trained flow, same patch"
TITLES = {"fill": "fill a removed segment\n(both ends known)", "forecast": "forecast\n(after the training window)",
          "backcast": "backcast\n(before the training window)"}

idf = df[(df.data == ID) & (df.horizon_h <= 24)]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3), sharey=True, layout="constrained")
for ax, ability in zip(axes, TITLES):
    g = idf[idf.ability == ability]
    for model, color, marker in (("trained", "tab:red", "o"), ("zero acceleration", "0.45", "s")):
        q = g[g.model == model].groupby("horizon_h")["error_km"].quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=0.2, lw=0)
        label = "trained Kinematic Interpolant (KI)" if model == "trained" else "no learning (straight motion)"
        ax.plot(q.index, q[0.5], marker=marker, color=color, label=label)
        if model == "trained":
            for h, v in q[0.5].items():
                ax.annotate(f"{v:.0f}" if v >= 10 else f"{v:.1f}", (h, v), textcoords="offset points",
                            xytext=(0, -12) if ability != "fill" else (0, 6), ha="center", color=color, fontsize=7.5)
    disp = g[g.model == "trained"].groupby("horizon_h")["displacement_km"].median()
    ax.plot(disp.index, disp, ls=":", color="k", label="distance travelled")
    ax.set(xscale="log", yscale="log", xlabel="horizon (h)", title=TITLES[ability])
    ax.set_xticks(disp.index, [f"{int(h)}" for h in disp.index])
    ax.xaxis.set_minor_locator(plt.NullLocator())
axes[0].set_ylabel("median position error of test parcels (km)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, fontsize=7.5)
f24 = idf[(idf.model == "trained") & (idf.horizon_h == 24)].groupby("ability")["error_km"].median()
fig.suptitle(f"In the training flow, 24-h gaps are filled to {f24['fill']:.1f} km, but 24-h forecasts and backcasts miss by "
             f"{f24['forecast']:.0f} and {f24['backcast']:.0f} km\n(8 patches of 1000 parcels, 1600 unseen test parcels; band = 25-75 %)",
             fontsize=8.5)
fig.savefig(ROOT / "figures/f1_abilities.png", dpi=200)
plt.close(fig)

order = [ID, "trained flow, other patch", "new perturbation", "new jet and perturbation"]
order = [o for o in order if o in set(df.data)]
short = {ID: "own\npatch", "trained flow, other patch": "other\nregion",
         "new perturbation": "new\nperturb.", "new jet and perturbation": "new jet\n+ perturb."}
h = a.horizon_ood
fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.4), sharey=True, layout="constrained")
x = np.arange(len(order))
for ax, ability in zip(axes, TITLES):
    g = df[(df.ability == ability) & (df.horizon_h == h)]
    med = g.groupby(["data", "model"])["error_km"].median()
    tr = [med.get((o, "trained"), np.nan) for o in order]
    ze = [med.get((o, "zero acceleration"), np.nan) for o in order]
    ax.bar(x[:1] - 0.2, tr[:1], 0.4, color="tab:red", label="trained Kinematic Interpolant (KI), in its own patch")
    ax.bar(x[1:] - 0.2, tr[1:], 0.4, color="#f4a582", label="trained KI, unseen region or flow")
    ax.bar(x + 0.2, ze, 0.4, color="0.7", label="no learning (straight motion)")
    for xi, v in zip(x - 0.2, tr):
        ax.annotate(f"{v:.0f}" if v >= 10 else f"{v:.1f}", (xi, v), textcoords="offset points", xytext=(0, 2),
                    ha="center", fontsize=7)
    ax.set(yscale="log", title=TITLES[ability].replace("\n", f" {int(h)} h\n", 1))
    ax.set_xticks(x, [short[o] for o in order], fontsize=7)
axes[0].set_ylabel(f"median position error at {int(h)} h (km)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, fontsize=7.5)
tr_fill = df[(df.ability == "fill") & (df.horizon_h == h) & (df.model == "trained")].groupby("data")["error_km"].median()
worst = tr_fill.drop(ID).max() if len(tr_fill) > 1 else np.nan
g24 = df[df.horizon_h == h].groupby(["ability", "data", "model"])["error_km"].median()
worse = {ab: sum(g24[(ab, o, "trained")] > g24[(ab, o, "zero acceleration")] for o in order[1:]) for ab in TITLES}
verdict = "does not transfer its gap filling" if worse["fill"] == len(order) - 1 else "transfers its gap filling only in part"
fig.suptitle(f"A model trained on one flow {verdict}: {int(h)}-h gap error {tr_fill[ID]:.1f} km in its own patch, "
             f"{tr_fill.drop(ID).min():.0f}-{worst:.0f} km elsewhere\n(worse than straight motion in the unseen cases: "
             f"gaps {worse['fill']}/{len(order) - 1}, forecasts {worse['forecast']}/{len(order) - 1}, backcasts "
             f"{worse['backcast']}/{len(order) - 1}; 1600 test parcels per bar)", fontsize=8.5)
fig.savefig(ROOT / "figures/f2_unseen_flows.png", dpi=200)
plt.close(fig)
print("figures written")
