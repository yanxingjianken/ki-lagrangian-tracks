# Adapted from roger/01_bve/scripts/08_train_context.py (three-dimensional tracks, start levels).
"""Train the 3-D conditional KI on the training flows (one GPU, or many with torchrun / DDP).

Training patches: patches 0-5 of every window and start level of the training flows; validation: patch 6 (24-h
conditional forecast and backcast error of fixed samples, horizontal in km and vertical in hPa). Rank 0 writes outputs/models/<name>.pt (best by validation),
<name>_latest.pt, <name>_history.json and outputs/logs/live_<name>_{loss,best,latest}.png.
"""
import argparse
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctx_data import context, flow_names, load_patches, rollout, sample_batch  # noqa: E402
from ctx_model import ContextKI  # noqa: E402
from geo import great_circle_km, stereo_to_lonlat  # noqa: E402
from live_panels import context_png, loss_png  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
H_KM = 7.0
p = argparse.ArgumentParser()
p.add_argument("--flows", default=str(ROOT / "configs/flows.json"))
p.add_argument("--windows", type=int, nargs="+", default=[0, 1])
p.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4])
p.add_argument("--steps", type=int, default=60000)
p.add_argument("--batch", type=int, default=32, help="patch samples per GPU")
p.add_argument("--context-parcels", type=int, default=192)
p.add_argument("--queries", type=int, default=256)
p.add_argument("--span-h", type=int, default=48)
p.add_argument("--horizon-h", type=int, default=24)
p.add_argument("--d-model", type=int, default=384)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--warmup", type=int, default=1000)
p.add_argument("--ema", type=float, default=0.999)
p.add_argument("--val-every", type=int, default=2000)
p.add_argument("--val-patches", type=int, default=96)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--name", default="context_ki_3d")
a = p.parse_args()

distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
if distributed:
    dist.init_process_group("nccl")
    rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
else:
    rank, world, local = 0, 1, 0
dev, main = f"cuda:{local}", rank == 0
names = flow_names(a.flows, "train")
train = load_patches(ROOT, names, a.windows, a.levels, range(6), device=dev)
gen = torch.Generator(device=dev).manual_seed(a.seed + 1000 * rank)
torch.manual_seed(a.seed)
model = ContextKI(d=a.d_model).to(dev)
net = DDP(model, device_ids=[local]) if distributed else model
ema = deepcopy(model).requires_grad_(False)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))

if main:
    val = load_patches(ROOT, names, a.windows, a.levels, [6], device=dev)
    vg = torch.Generator(device=dev).manual_seed(1234)
    keep = torch.randperm(val.X.shape[0], device=dev, generator=vg)[: a.val_patches]
    nv = len(keep)
    vp = keep.repeat(2)
    vdir = torch.cat([torch.ones(nv), -torch.ones(nv)]).long().to(dev)
    vt0 = torch.where(vdir > 0, a.span_h + a.horizon_h, a.span_h)
    vk = torch.rand(len(vp), val.X.shape[2], device=dev, generator=vg).argsort(1)
    vctx, vq = vk[:, : a.context_parcels], vk[:, -64:]
    vtok, vcen = context(val, vp, vt0, vdir, vctx, a.span_h)
    vx0, vv0 = val.X[vp[:, None], vt0[:, None], vq], val.V[vp[:, None], vt0[:, None], vq]
    vend = val.X[vp[:, None], (vt0 + vdir * a.horizon_h)[:, None], vq].cpu().numpy()
    vtracks = val.X[vp[:, None, None], (vt0[:, None] + vdir[:, None] * torch.arange(a.horizon_h + 1, device=dev))[:, :, None],
                    vq[:, None]].cpu().numpy()


def validate(m):
    path = rollout(m, vtok, vcen, vx0, vv0, vdir, a.horizon_h).cpu().numpy()
    (lp, ap), (lt, at) = stereo_to_lonlat(path[-1][..., :2]), stereo_to_lonlat(vend[..., :2])
    err = great_circle_km(lp, ap, lt, at)
    dp = np.abs(1000.0 * (np.exp(-path[-1][..., 2] / H_KM) - np.exp(-vend[..., 2] / H_KM)))
    h = len(err) // 2
    fc, bc = float(np.median(err[:h])), float(np.median(err[h:]))
    hist["forecast_hpa"].append(float(np.median(dp[:h])))
    hist["backcast_hpa"].append(float(np.median(dp[h:])))
    return fc, bc, path


hist = dict(step=[], train=[], val_step=[], val_km=[], forecast_km=[], backcast_km=[],
            forecast_hpa=[], backcast_hpa=[], world_size=world, flows=names)
best = np.inf
for step in range(1, a.steps + 1):
    tok, qry, acc, score = sample_batch(train, a.batch, a.context_parcels, a.queries, a.span_h, a.horizon_h, gen=gen)
    a_hat, s_hat = net(tok, None, qry)
    loss = ((a_hat - acc) ** 2).mean() + ((s_hat - score) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    sched.step()
    with torch.no_grad():
        for e, w in zip(ema.parameters(), model.parameters()):
            e.lerp_(w, 1 - a.ema)
    hist["step"].append(step)
    hist["train"].append(float(loss))
    if step % a.val_every and step != a.steps:
        continue
    finite = torch.tensor(float(torch.isfinite(loss)), device=dev)
    if distributed:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if finite.item() < 1:
        raise FloatingPointError("nonfinite training loss")
    if not main:
        continue
    fc, bc, path = validate(ema)
    score_km = 0.5 * (fc + bc)
    hist["val_step"].append(step)
    hist["val_km"].append(score_km)
    hist["forecast_km"].append(fc)
    hist["backcast_km"].append(bc)
    out, logs = ROOT / "outputs/models", ROOT / "outputs/logs"
    out.mkdir(parents=True, exist_ok=True)
    ck = dict(state=ema.state_dict(), step=step, forecast_km=fc, backcast_km=bc, args=vars(a), world_size=world)
    for which in ["latest"] + (["best"] if score_km < best else []):
        path_ck = out / (f"{a.name}.pt" if which == "best" else f"{a.name}_latest.pt")
        torch.save(ck, path_ck.with_suffix(".tmp"))
        os.replace(path_ck.with_suffix(".tmp"), path_ck)
        context_png(logs / f"live_{a.name}_{which}.png", vtok, vcen, path, vtracks, nv,
                    f"{a.name} {which}: step {step}, validation 24-h forecast {fc:.0f} km, backcast {bc:.0f} km")
    best = min(best, score_km)
    loss_png(logs / f"live_{a.name}_loss.png", hist, a.name, val_label="mean of 24-h forecast and backcast error (km)")
    (out / f"{a.name}_history.json").write_text(json.dumps(hist))
    print(f"step {step} loss {loss.item():.4g} forecast {fc:.1f} km {hist['forecast_hpa'][-1]:.1f} hPa backcast "
          f"{bc:.1f} km {hist['backcast_hpa'][-1]:.1f} hPa (world {world})", flush=True)
if distributed:
    dist.destroy_process_group()
