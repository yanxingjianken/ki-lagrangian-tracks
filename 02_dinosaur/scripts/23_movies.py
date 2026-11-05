"""Movies of 3-D conditional forecasts and backcasts over the evolving flow, one per test case.

Columns: forecast (context = the patch's observed tracks over the 48 h before the anchor) and backcast (48 h after).
Top: relative vorticity (colour), streamfunction (contours) and wind (arrows) on the patch's start level at the
frame's time, polar-stereographic; target parcels not in the context: truth blue, model red dashed. Bottom: their
pressure against hours from the anchor. Writes figures/movies/<case>.mp4 and a check frame outputs/movie_frames/<case>.png. FFMPEG = ffmpeg path.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctx_data import context, load_patches, rollout  # noqa: E402
from ctx_model import ContextKI  # noqa: E402
from flowfields import mirror_global, streamfunction, vorticity  # noqa: E402
from geo import lonlat_to_stereo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
H_KM = 7.0
SIGMA = (0.968, 0.887, 0.674, 0.477, 0.241)
CASES = [("training flows, unseen patch", "t05_lc1_u50_m6_a1"),
         ("unseen, inside the training ranges", "o3_lc2polarcap_u50_m6_a1"),
         ("unseen, outside: jet 30 m/s", "o5_lc1_u30_m6_a1"),
         ("unseen, outside: wavenumber 9", "o7_lc2polarcap_u45_m9_a1")]
p = argparse.ArgumentParser()
p.add_argument("--model", default=str(ROOT / "outputs/models/context_ki_3d.pt"))
p.add_argument("--window", type=int, default=0)
p.add_argument("--level", type=int, default=3)
p.add_argument("--patch", type=int, default=7)
p.add_argument("--anchor-h", type=int, default=96)
p.add_argument("--targets", type=int, default=12)
p.add_argument("--box-km", type=float, default=3500.0)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams["animation.ffmpeg_path"] = os.environ.get("FFMPEG", "ffmpeg")
ck = torch.load(a.model, map_location=dev, weights_only=False)
model = ContextKI(d=ck["args"]["d_model"]).to(dev).eval()
model.load_state_dict(ck["state"])
K, span = ck["args"]["context_parcels"], ck["args"]["span_h"]
pres = lambda z: 1000.0 * np.exp(-z / H_KM)

LEGEND = [Line2D([], [], color="0.55", lw=1, label="context: other parcels' observed tracks (model input)"),
          Line2D([], [], color="tab:blue", lw=2, label="truth: target parcels (not in the context)"),
          Line2D([], [], color="tab:red", lw=1.5, ls="--", label="conditional KI prediction")]

for label, flow in CASES:
    D = load_patches(ROOT, [flow], [a.window], [a.level], [a.patch], device=dev)
    gen = torch.Generator(device=dev).manual_seed(3)
    perm = torch.rand(D.X.shape[2], device=dev, generator=gen).argsort()
    kidx, qidx = perm[:K][None].repeat(2, 1), perm[K:K + a.targets][None].repeat(2, 1)
    pp, dd = torch.zeros(2, dtype=torch.long, device=dev), torch.tensor([1, -1], device=dev)
    tt = torch.full((2,), a.anchor_h, device=dev)
    tok, cen = context(D, pp, tt, dd, kidx, span)
    x0, v0 = D.X[pp[:, None], tt[:, None], qidx], D.V[pp[:, None], tt[:, None], qidx]
    path = rollout(model, tok, cen, x0, v0, dd, 24).cpu().numpy()
    hrs = tt[:, None] + dd[:, None] * torch.arange(25, device=dev)
    truth = D.X[pp[:, None, None], hrs[:, :, None], qidx[:, None]].transpose(0, 1).cpu().numpy()
    ctx = [D.X[0][h][:, kidx[0]].cpu().numpy() for h in
           (np.arange(a.anchor_h - span, a.anchor_h + 1), np.arange(a.anchor_h, a.anchor_h + span + 1))]
    ds = xr.open_dataset(ROOT / f"data/dino_{flow}.nc").sel(level=SIGMA[a.level], method="nearest")
    h0 = 24 * (2 + 6 * a.window)
    need = sorted({a.anchor_h + s * h for s in (1, -1) for h in range(25)})
    sub = ds.isel(time=[h0 + h for h in need])
    ug, vg = mirror_global(sub.u.values, sub.v.values, ds.lat.values)
    zeta, psi = vorticity(ug, vg)[:, :91], streamfunction(ug, vg)[:, :91] / 1e6
    lat_n = np.linspace(90, 0, 91)
    cyc = lambda f: np.concatenate([f, f[..., :1]], -1)
    LON, LAT = np.meshgrid(np.r_[ds.lon.values, 360.0], lat_n)
    XY = lonlat_to_stereo(LON, LAT) / 1e3
    un, vn = (cyc(f[:, :91]) for f in (ug, vg))
    lam = np.deg2rad(LON)
    fld = {h: (cyc(zeta[i]) * 1e4, cyc(psi[i]), un[i] * np.cos(lam) - vn[i] * np.sin(lam),
               un[i] * np.sin(lam) + vn[i] * np.cos(lam)) for i, h in enumerate(need)}
    centre = cen.cpu().numpy()[0] / 1e3
    stem = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    out = ROOT / "figures/movies"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.6), layout="constrained", gridspec_kw=dict(height_ratios=[2.2, 1]))
    writer = FFMpegWriter(fps=4, extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p"])
    frames = [None] * 6 + list(range(25))
    with writer.saving(fig, str(out / f"{stem}.mp4"), dpi=90):
        for n, h in enumerate(frames):
            k = 0 if h is None else h
            for b in range(2):
                ax, axp = axes[0, b], axes[1, b]
                ax.clear()
                axp.clear()
                z, ps_, wx, wy = fld[a.anchor_h + (1 if b == 0 else -1) * k]
                ax.pcolormesh(XY[..., 0], XY[..., 1], z, cmap="RdBu_r", vmin=-1.5, vmax=1.5, shading="gouraud")
                ax.contour(XY[..., 0], XY[..., 1], ps_, levels=20, colors="k", linewidths=0.4, negative_linestyles="solid")
                s = (slice(None, None, 4), slice(None, None, 4))
                keep = LAT[s] < 80
                ax.quiver(XY[s][..., 0][keep], XY[s][..., 1][keep], wx[s][keep], wy[s][keep], scale=1500, width=0.002, color="0.3")
                c = ctx[b] / 1e3
                ax.plot(c[:, :, 0], c[:, :, 1], color="0.55", lw=0.4)
                ax.plot(truth[: k + 1, b, :, 0] / 1e3, truth[: k + 1, b, :, 1] / 1e3, color="tab:blue", lw=2)
                ax.plot(path[: k + 1, b, :, 0] / 1e3, path[: k + 1, b, :, 1] / 1e3, color="tab:red", lw=1.5, ls="--")
                ax.plot(truth[k, b, :, 0] / 1e3, truth[k, b, :, 1] / 1e3, "o", color="tab:blue", ms=4)
                ax.plot(path[k, b, :, 0] / 1e3, path[k, b, :, 1] / 1e3, "o", color="tab:red", ms=3)
                bx = a.box_km / 1e3
                ax.set(xlim=(centre[0] - bx, centre[0] + bx), ylim=(centre[1] - bx, centre[1] + bx), aspect="equal",
                       xlabel="x (1000 km)")
                task, sign = ("forecast", 1) if b == 0 else ("backcast", -1)
                when = "context: other parcels' observed 48 h (grey)" if h is None else f"{sign * h:+d} h from the anchor"
                ax.set_title(f"{task}: {when}; fields on sigma {SIGMA[a.level]}", fontsize=10)
                hours = sign * np.arange(k + 1)
                axp.plot(hours, pres(truth[: k + 1, b, :, 2]), color="tab:blue", lw=1.5)
                axp.plot(hours, pres(path[: k + 1, b, :, 2]), color="tab:red", lw=1.2, ls="--")
                lim = pres(np.concatenate([truth[:, b, :, 2], path[:, b, :, 2]]))
                axp.set(xlim=(0, 24) if b == 0 else (-24, 0), ylim=(lim.max() + 10, lim.min() - 10),
                        xlabel="hours from the anchor", ylabel="pressure (hPa)")
            axes[0, 0].set_ylabel("y (1000 km)")
            axes[0, 0].legend(handles=LEGEND, loc="lower left", fontsize=7, framealpha=0.85)
            axes[1, 0].legend(handles=LEGEND[1:], loc="best", fontsize=7, framealpha=0.85)
            fig.suptitle(f"{label} ({flow}): truth blue, conditional KI red dashed; vorticity (colour, 1e-4 s-1), "
                         f"streamfunction (contours), wind (arrows)", fontsize=9)
            writer.grab_frame()
            if n == len(frames) - 13:
                (ROOT / "outputs/movie_frames").mkdir(parents=True, exist_ok=True)
                fig.savefig(ROOT / "outputs/movie_frames" / f"{stem}.png", dpi=90)
    plt.close(fig)
    print("wrote", out / f"{stem}.mp4")
