"""Train the acceleration and score networks on the window [t_start, t_end] after release, for one patch (or all
parcels), on one GPU or many.

One GPU: python 05_train.py ...; several: torchrun --nproc-per-node G [--nnodes N --rdzv-backend c10d
--rdzv-endpoint HOST:PORT] 05_train.py ... (DDP; each rank draws its minibatches from its own shard of the training
parcels). Rank 0 writes <out>/<name>.pt (best by the validation 24-h gap error), <out>/<name>_latest.pt,
<out>/<name>_history.json and logs/live_<name>_{loss,best,latest}.png.
"""
import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ki_tracks import KINet, batch_loss, great_circle_km, infill, load_patch, stereo_to_lonlat, to_km, training_batch  # noqa: E402
from live_panels import example_png, loss_png  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--data", default=str(ROOT / "data/tracks_lagranto_w0.npz"))
p.add_argument("--patch", type=int, default=0, help="patch index; -1 trains one model on all parcels")
p.add_argument("--t-start-h", type=float, default=24.0, help="training window start, hours after release")
p.add_argument("--t-end-h", type=float, default=120.0)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--batch", type=int, default=1024, help="per GPU")
p.add_argument("--width", type=int, default=512)
p.add_argument("--depth", type=int, default=4)
p.add_argument("--freqs", type=float, nargs="+", default=None, help="default: 1, 2, 4, ... up to a 6-h period")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--ema", type=float, default=0.99)
p.add_argument("--val-every", type=int, default=500)
p.add_argument("--val-parcels", type=int, default=512)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--name", default=None)
p.add_argument("--out", default=str(ROOT / "outputs/models"))
a = p.parse_args()

distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
if distributed:
    dist.init_process_group("nccl")
    rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
else:
    rank, world, local = 0, 1, 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
main = rank == 0

tag = "all" if a.patch < 0 else a.patch
name = a.name or f"{Path(a.data).stem}_p{tag}_s{int(a.t_start_h)}_t{int(a.t_end_h)}"
out, logs = Path(a.out), ROOT / "outputs/logs"
d = load_patch(a.data, a.patch, a.t_end_h, t_start_h=a.t_start_h)
if a.freqs is None:
    a.freqs = [2.0**k for k in range(int(np.log2(d.clock / 6.0)) + 1)]
shard = d.train[rank::world]
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed + 1000 * rank)
model = KINet(width=a.width, depth=a.depth, freqs=a.freqs).to(dev)
net = DDP(model, device_ids=[local]) if distributed else model
ema = deepcopy(model).requires_grad_(False)
opt = torch.optim.Adam(net.parameters(), lr=a.lr)

vrng = np.random.default_rng(1234)
val = np.sort(vrng.choice(d.val, min(a.val_parcels, len(d.val)), replace=False))
val_batches = [training_batch(d, vrng, val, a.batch, dev) for _ in range(4)]
gap_steps, L = 48, 24.0 / d.clock
k0 = np.searchsorted(d.t, vrng.uniform(0.0, 1.0 - L, len(val)))
k1 = np.searchsorted(d.t, d.t[k0] + L - 1e-9)
fixed = int(np.argmin(np.abs(d.t[k0] - 0.5 * (1.0 - L))))


def val_infill(m):
    tt = lambda arr, k: torch.tensor(arr[k, val], dtype=torch.float32, device=dev)
    path = infill(m, d, tt(d.x, k0), tt(d.v, k0), tt(d.x, k1), tt(d.v, k1), d.t[k0], d.t[k1], gap_steps).cpu().numpy()
    tq = d.t[k0][None] + (d.t[k1] - d.t[k0])[None] * np.linspace(0, 1, gap_steps + 1)[:, None]
    truth = np.stack([np.stack([np.interp(tq[:, j], d.t, d.x[:, q, c]) for c in (0, 1)], -1) for j, q in enumerate(val)], 1)
    lo_p, la_p = stereo_to_lonlat(to_km(d, path))
    lo_t, la_t = stereo_to_lonlat(to_km(d, truth))
    return float(np.median(great_circle_km(lo_p, la_p, lo_t, la_t).mean(0))), path.transpose(1, 0, 2)


if main:
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
hist = dict(step=[], train=[], val_step=[], val_loss=[], val_infill_km=[], world_size=world)
best = np.inf
for step in range(1, a.steps + 1):
    loss = batch_loss(net, training_batch(d, rng, shard, a.batch, dev))
    opt.zero_grad()
    loss.backward()
    opt.step()
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
    with torch.no_grad():
        vl = float(np.mean([batch_loss(ema, b).item() for b in val_batches]))
    vk, paths = val_infill(ema)
    hist["val_step"].append(step)
    hist["val_loss"].append(vl)
    hist["val_infill_km"].append(vk)
    ck = dict(state=ema.state_dict(), step=step, val_loss=vl, val_infill_km=vk, args=vars(a), name=name,
              center=d.center, scale=d.scale, clock=d.clock, t_start_h=a.t_start_h, world_size=world)
    for which in ["latest"] + (["best"] if vk < best else []):
        path = out / (f"{name}.pt" if which == "best" else f"{name}_latest.pt")
        torch.save(ck, path.with_suffix(".tmp"))
        os.replace(path.with_suffix(".tmp"), path)
        example_png(logs / f"live_{name}_{which}.png", d, fixed, k0[fixed], k1[fixed], paths[fixed],
                    f"{name} {which}: step {step}, validation 24-h gap median {vk:.1f} km", parcels=val)
    best = min(best, vk)
    loss_png(logs / f"live_{name}_loss.png", hist, name)
    (out / f"{name}_history.json").write_text(json.dumps(hist))
    print(f"{name} step {step} train {loss.item():.4g} val {vl:.4g} val-gap {vk:.1f} km (world {world})", flush=True)
if distributed:
    dist.destroy_process_group()
