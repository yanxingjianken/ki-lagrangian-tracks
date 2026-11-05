"""Movies of conditional forecasts and backcasts over the evolving flow: one per test case (unseen patches of the
training flows, and each unseen flow).

Left: forecast (context = the patch's observed tracks over the 48 h before the anchor); right: backcast (context
over the 48 h after it). Background at the frame's time: relative vorticity (colour), streamfunction (contours) and
wind (arrows) in polar-stereographic coordinates; target parcels not in the context: truth blue, model red dashed.
Writes figures/movies/<case>.mp4 and a check frame outputs/movie_frames/<case>.png. FFMPEG = path of ffmpeg.
"""
import argparse
import json
import os
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
from flowfields import streamfunction  # noqa: E402
from geo import lonlat_to_stereo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--flows", default=str(ROOT / "configs/flows.json"))
p.add_argument("--model", default=str(ROOT / "outputs/models/context_ki.pt"))
p.add_argument("--window", type=int, default=0)
p.add_argument("--anchor-h", type=int, default=72)
p.add_argument("--targets", type=int, default=12)
p.add_argument("--box-km", type=float, default=3500.0)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams["animation.ffmpeg_path"] = os.environ.get("FFMPEG", "ffmpeg")
ck = torch.load(a.model, map_location=dev, weights_only=False)
model = ContextKI(d=ck["args"]["d_model"]).to(dev).eval()
model.load_state_dict(ck["state"])
K, span = ck["args"]["context_parcels"], ck["args"]["span_h"]
flows = json.loads(Path(a.flows).read_text())["flows"]
cases = [("training flows, unseen patch", next(f["name"] for f in flows if f["split"] == "train"), 18)]
cases += [(f.get("label", f["name"]), f["name"], 18) for f in flows if f["split"] == "test"]


LEGEND = [Line2D([], [], color="0.55", lw=1, label="context: other parcels' observed tracks (model input)"),
          Line2D([], [], color="tab:blue", lw=2, label="truth: target parcels (not in the context)"),
          Line2D([], [], color="tab:red", lw=1.5, ls="--", label="conditional KI prediction")]

for label, flow, patch in cases:
    D = load_patches(ROOT, [flow], [a.window], [patch], device=dev)
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
    ctx_hrs = [np.arange(a.anchor_h - span, a.anchor_h + 1), np.arange(a.anchor_h, a.anchor_h + span + 1)]
    ctx = [D.X[0][h][:, kidx[0]].cpu().numpy() for h in ctx_hrs]
    ds = xr.open_dataset(ROOT / f"data/bve_{flow}.nc")
    with np.load(ROOT / f"data/tracks_lagranto_{flow}_w{a.window}.npz") as f:
        h_release = int(f["start_hour"])
    nh = ds.lat.values >= 0
    cyc = lambda f: np.concatenate([f, f[..., :1]], -1)
    LON, LAT = np.meshgrid(np.r_[ds.lon.values, 360.0], ds.lat.values[nh])
    XY = lonlat_to_stereo(LON, LAT)
    centre = cen.cpu().numpy()[0]
    need = sorted({a.anchor_h + s * h for s in (1, -1) for h in range(25)})
    sel = ds.isel(time=[h_release + h for h in need])
    psi = dict(zip(need, cyc(streamfunction(sel.u.values, sel.v.values)[:, nh] / 1e6)))
    fld = {h: (cyc(sel.zeta.values[i][nh]) * 1e4, cyc(sel.u.values[i][nh]), cyc(sel.v.values[i][nh]))
           for i, h in enumerate(need)}
    stem = label.split(":")[0].replace(",", "").replace(" ", "_")
    out = ROOT / "figures/movies"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), layout="constrained")
    writer = FFMpegWriter(fps=4, extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p"])
    frames = [None] * 6 + list(range(25))
    with writer.saving(fig, str(out / f"{stem}.mp4"), dpi=100):
        for n, h in enumerate(frames):
            for b, ax in enumerate(axes):
                ax.clear()
                hh = a.anchor_h + (0 if h is None else (1 if b == 0 else -1) * h)
                z, u, v = fld[hh]
                ax.pcolormesh(XY[..., 0] / 1e3, XY[..., 1] / 1e3, z, cmap="RdBu_r", vmin=-1.5, vmax=1.5, shading="gouraud")
                ax.contour(XY[..., 0] / 1e3, XY[..., 1] / 1e3, psi[hh], levels=20, colors="k", linewidths=0.4,
                           negative_linestyles="solid")
                s = (slice(None, None, 4), slice(None, None, 4))
                lam = np.deg2rad(LON)
                wx, wy = u * np.cos(lam) - v * np.sin(lam), u * np.sin(lam) + v * np.cos(lam)
                keep = LAT[s] < 80
                ax.quiver(XY[s][..., 0][keep] / 1e3, XY[s][..., 1][keep] / 1e3, wx[s][keep], wy[s][keep], scale=1500, width=0.002, color="0.3")
                c = ctx[b] / 1e3
                ax.plot(c[:, :, 0], c[:, :, 1], color="0.55", lw=0.4)
                k = 0 if h is None else h
                ax.plot(truth[: k + 1, b, :, 0] / 1e3, truth[: k + 1, b, :, 1] / 1e3, color="tab:blue", lw=2)
                ax.plot(path[: k + 1, b, :, 0] / 1e3, path[: k + 1, b, :, 1] / 1e3, color="tab:red", lw=1.5, ls="--")
                ax.plot(truth[k, b, :, 0] / 1e3, truth[k, b, :, 1] / 1e3, "o", color="tab:blue", ms=4)
                ax.plot(path[k, b, :, 0] / 1e3, path[k, b, :, 1] / 1e3, "o", color="tab:red", ms=3)
                bx = a.box_km / 1e3
                ax.set(xlim=(centre[0] / 1e3 - bx, centre[0] / 1e3 + bx), ylim=(centre[1] / 1e3 - bx, centre[1] / 1e3 + bx),
                       aspect="equal", xlabel="x (1000 km)")
                task = "forecast" if b == 0 else "backcast"
                when = "context: other parcels' observed 48 h (grey)" if h is None else f"{'+' if b == 0 else '-'}{h} h from the anchor"
                ax.set_title(f"{task}: {when}", fontsize=10)
            axes[0].set_ylabel("y (1000 km)")
            axes[0].legend(handles=LEGEND, loc="lower left", fontsize=7, framealpha=0.85)
            fig.suptitle(f"{label} ({flow}): truth blue, conditional KI red dashed; vorticity (colour, 1e-4 s-1), "
                         f"streamfunction (contours), wind (arrows)", fontsize=9)
            writer.grab_frame()
            if n == len(frames) - 13:
                (ROOT / "outputs/movie_frames").mkdir(parents=True, exist_ok=True)
                fig.savefig(ROOT / "outputs/movie_frames" / f"{stem}.png", dpi=100)
    plt.close(fig)
    print("wrote", out / f"{stem}.mp4")
