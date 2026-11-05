"""Sanity figure figures/check_eke_omega.png: eddy kinetic energy of every flow (one panel per basic-state type) and
day-7 maps of omega at sigma 0.887 with surface-pressure contours for one LC1 and one LC2 flow (north polar
stereographic, 20-90N).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TYPES = {"lc1": "LC1 (Z1 only)", "lc2lobes": "LC2, Z2 lobes", "lc2polarcap": "LC2, Z2 polar cap"}


def polar_xy(lat, lon):
    r = np.tan(np.deg2rad(90.0 - lat) / 2) / np.tan(np.deg2rad(35.0))
    return r * np.sin(np.deg2rad(lon)), -r * np.cos(np.deg2rad(lon))


def draw_map(ax, path: Path, hour: int, level: float = 0.887):
    ds = xr.open_dataset(path).isel(time=hour)
    ds = ds.sel(lat=slice(20, 90))
    om = ds.omega.sel(level=level, method="nearest").values
    ps = ds.ps.values / 100.0
    lon = np.r_[ds.lon.values, 360.0]
    lo, la = np.meshgrid(lon, ds.lat.values)
    x, y = polar_xy(la, lo)
    wrap = lambda f: np.concatenate([f, f[:, :1]], 1)
    lim = float(np.percentile(np.abs(om), 99.5))
    cf = ax.contourf(x, y, wrap(om), np.linspace(-lim, lim, 21), cmap="RdBu_r", extend="both")
    cs = ax.contour(x, y, wrap(ps), np.arange(940, 1060, 5), colors="k", linewidths=0.6)
    ax.clabel(cs, fmt="%d", fontsize=6)
    for lat_c in (30, 45, 60, 75):
        cx, cy = polar_xy(np.full(361, lat_c), np.arange(361))
        ax.plot(cx, cy, color="0.6", lw=0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    sig = float(ds.level.sel(level=level, method="nearest"))
    ax.set_title(f"{path.stem[5:]}, day {hour / 24:g}\nomega at sigma {sig:.3f} (colour), ps every 5 hPa", fontsize=8)
    return cf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs=2, default=["t05_lc1_u50_m6_a1", "t16_lc2lobes_u45_m6_a1"])
    ap.add_argument("--hour", type=int, default=168)
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "check_eke_omega.png")
    a = ap.parse_args()

    flows = json.loads((ROOT / "configs" / "flows.json").read_text())
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.15], hspace=0.25, wspace=0.5)
    cmap, norm = plt.get_cmap("viridis"), plt.Normalize(30, 65)
    axes = [fig.add_subplot(gs[0, 2 * k:2 * k + 2]) for k in range(3)]
    for k, (typ, title) in enumerate(TYPES.items()):
        ax = axes[k]
        ax.set_yscale("log")
        n = 0
        for f in flows:
            check = ROOT / "data" / f"check_{f['name']}.json"
            if not f["basic"].startswith(typ + "_") or not check.exists():
                continue
            c = json.loads(check.read_text())
            if "eke" not in c:
                continue
            ls = "-" if f["split"] == "train" else "--"
            ax.plot(np.asarray(c["hours"]) / 24, np.asarray(c["eke"]) / 1e5, ls, color=cmap(norm(f["ujet"])),
                    lw=0.8 if ls == "-" else 1.4)
            n += 1
        ax.set_title(f"{title}: {n} flows", fontsize=9)
        ax.set_xlim(0, 15)
        ax.set_ylim(1e-3, 50)
        ax.set_xlabel("day")
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel("eddy kinetic energy (1e5 J m-2)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.01, label="jet speed (m/s); dashed = unseen flows")
    maps = [fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])]
    cfs = [draw_map(ax, ROOT / "data" / f"dino_{name}.nc", a.hour) for ax, name in zip(maps, a.maps)]
    for ax, cf in zip(maps, cfs):
        fig.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, label="omega (Pa/s), negative = ascent")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
