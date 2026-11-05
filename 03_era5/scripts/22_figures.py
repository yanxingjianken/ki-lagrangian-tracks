# Adapted from roger/02_dinosaur/scripts/22_figures.py (ERA5 test cases, hemispheres, 20 start levels).
"""Result figures of the ERA5 conditional KI from outputs/eval/context_ki_era5.csv.gz.

figures/f1_conditional_ki_era5.png: horizontal forecast and backcast error against horizon for the four test cases,
against straight motion.
figures/f2_by_level.png: 24-h horizontal and vertical errors by start level, unseen patches of the training releases.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIGMA = json.loads((ROOT / "configs/releases.json").read_text())["sigma"]
df = pd.read_csv(ROOT / "outputs/eval/context_ki_era5.csv.gz")
groups = [g for g in ["training releases, unseen patches", "2021, unseen year", "2021, unseen latitudes 10-25",
                      "2021, unseen latitudes 70-85"] if g in set(df.case)]
colors = dict(zip(groups, ["tab:red", "tab:purple", "tab:orange", "tab:blue"]))
KI, STRAIGHT = "conditional KI", "no learning (straight motion)"
plt.rcParams.update({"font.size": 8.5})
(ROOT / "figures").mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.9), sharey=True, layout="constrained")
for ax, task in zip(axes, ["forecast", "backcast"]):
    g = df[df.task == task]
    for grp in groups:
        for model, ls, mk in ((KI, "-", "o"), (STRAIGHT, ":", "s")):
            m = g[(g.case == grp) & (g.model == model)].groupby("horizon_h")["error_km"].median()
            ax.plot(m.index, m, ls=ls, marker=mk, ms=4, color=colors[grp], label=grp if model == KI else None)
    disp = g[(g.case == groups[0]) & (g.model == KI)].groupby("horizon_h")["displacement_km"].median()
    ax.plot(disp.index, disp, color="k", lw=0.8, ls="--", label="distance travelled" if task == "forecast" else None)
    ax.set(xscale="log", yscale="log", xlabel="horizon (h)", title=f"conditional {task}, ERA5 3-D tracks")
    ax.set_xticks(disp.index, [str(int(h)) for h in disp.index])
    ax.xaxis.set_minor_locator(plt.NullLocator())
axes[0].set_ylabel("median horizontal error (km)")
axes[0].plot([], [], "k-o", ms=4, label="solid: conditional Kinematic Interpolant (KI)")
axes[0].plot([], [], "k:s", ms=4, label="dotted: straight motion")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=7)
m24 = df[(df.horizon_h == 24) & (df.model == KI)].groupby("case")["error_km"].median()
s24 = df[(df.horizon_h == 24) & (df.model == STRAIGHT)].groupby("case")["error_km"].median()
gain = (1 - m24 / s24) * 100
verdict = "keeps its skill in every test case" if (gain[groups] > 0).all() else "loses its skill in some test cases"
fig.suptitle(f"The ERA5 conditional KI {verdict}: 24-h horizontal error below straight motion by "
             + ", ".join(f"{gain[g]:.0f} % ({g})" for g in groups) + "; forecast and backcast pooled", fontsize=7.5,
             wrap=True)
fig.savefig(ROOT / "figures/f1_conditional_ki_era5.png", dpi=200)
plt.close(fig)

g24 = df[(df.horizon_h == 24) & (df.case == groups[0])].copy()
g24["abs_hpa"] = g24.error_hpa.abs()
g24["abs_true_hpa"] = g24.dp_true_hpa.abs()
levels = sorted(g24.level.unique())
fig, axes = plt.subplots(1, 2, figsize=(7.4, 6.0), sharey=True, layout="constrained")
y = np.arange(len(levels))
med = g24.groupby(["level", "model"])[["error_km", "abs_hpa"]].median().unstack("model")
for ax, col, xlabel in ((axes[0], "error_km", "median horizontal error at 24 h (km)"),
                        (axes[1], "abs_hpa", "median vertical error at 24 h (hPa)")):
    ax.barh(y + 0.18, med[(col, KI)].reindex(levels).values, 0.36, color="tab:red", label="conditional KI")
    ax.barh(y - 0.18, med[(col, STRAIGHT)].reindex(levels).values, 0.36, color="0.7", label="straight motion")
    ax.set(xlabel=xlabel)
t = g24[g24.model == KI].groupby("level")["abs_true_hpa"].median().reindex(levels).values
axes[1].plot(t, y, "k|", ms=8, mew=2, label="true 24-h pressure change")
axes[0].set_yticks(y, [f"start sigma {SIGMA[k]}" for k in levels])
axes[0].invert_yaxis()
fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, fontsize=7.5)
ratio_h = (med[("error_km", KI)] / med[("error_km", STRAIGHT)]).values
ratio_v = (med[("abs_hpa", KI)] / med[("abs_hpa", STRAIGHT)]).values
fig.suptitle(f"At every start level the conditional KI's 24-h error is a fraction of straight motion's: "
             f"{ratio_h.min() * 100:.0f}-{ratio_h.max() * 100:.0f} % horizontally,\n{ratio_v.min() * 100:.0f}-"
             f"{ratio_v.max() * 100:.0f} % vertically (training releases, unseen patches; forecast and backcast pooled)",
             fontsize=8)
fig.savefig(ROOT / "figures/f2_by_level.png", dpi=200)
print(m24.round(0).to_dict(), s24.round(0).to_dict())
