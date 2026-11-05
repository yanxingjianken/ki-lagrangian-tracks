"""Result figure of the conditional KI from outputs/eval/context_ki.csv.gz: forecast and backcast error against horizon
on unseen patches of the training flows and in the three unseen flows, against straight motion.

Writes figures/f3_conditional_ki.png.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "outputs/eval/context_ki.csv.gz")
plt.rcParams.update({"font.size": 8.5})
cases = list(dict.fromkeys(df.case))
colors = dict(zip(cases, ["tab:red", "tab:orange", "tab:purple", "tab:brown"]))
short = {c: c.replace("new jet and wavenumber, ", "new jet + wavenumber,\n") for c in cases}
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True, layout="constrained")
for ax, task in zip(axes, ["forecast", "backcast"]):
    g = df[df.task == task]
    for case in cases:
        for model, ls, mk in (("conditional KI", "-", "o"), ("no learning (straight motion)", ":", "s")):
            m = g[(g.case == case) & (g.model == model)].groupby("horizon_h")["error_km"].median()
            ax.plot(m.index, m, ls=ls, marker=mk, ms=4, color=colors[case],
                    label=short[case] if model == "conditional KI" else None)
    disp = g[(g.case == cases[0]) & (g.model == "conditional KI")].groupby("horizon_h")["displacement_km"].median()
    ax.plot(disp.index, disp, color="k", lw=0.8, ls="--", label="distance travelled" if task == "forecast" else None)
    ax.set(xscale="log", yscale="log", xlabel="horizon (h)", title=f"conditional {task}")
    ax.set_xticks(disp.index, [str(int(h)) for h in disp.index])
    ax.xaxis.set_minor_locator(plt.NullLocator())
axes[0].set_ylabel("median position error of target parcels (km)")
axes[0].plot([], [], "k-o", ms=4, label="solid: conditional Kinematic Interpolant (KI)")
axes[0].plot([], [], "k:s", ms=4, label="dotted: straight motion")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, fontsize=7)
m24 = df[(df.horizon_h == 24) & (df.model == "conditional KI")].groupby("case")["error_km"].median()
s24 = df[(df.horizon_h == 24) & (df.model != "conditional KI")].groupby("case")["error_km"].median()
gain = (1 - m24 / s24) * 100
verdict = "keeps most of its skill in unseen flows" if (gain.drop(cases[0]) > 0).all() else "loses its skill in some unseen flows"
fig.suptitle(f"The conditional KI {verdict} (24-h error, forecast and backcast pooled):\n"
             f"{m24[cases[0]]:.0f} km on unseen patches of trained flows ({gain[cases[0]]:.0f} % below straight motion); "
             f"{m24.drop(cases[0]).min():.0f}-{m24.drop(cases[0]).max():.0f} km in unseen flows "
             f"({gain.drop(cases[0]).min():.0f}-{gain.drop(cases[0]).max():.0f} % below)", fontsize=8)
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures/f3_conditional_ki.png", dpi=200)
print(m24.round(0).to_dict(), s24.round(0).to_dict())
