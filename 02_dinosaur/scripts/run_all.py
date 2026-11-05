"""Runs every flow of configs/flows.json: basic state (CPU), normal mode (shared by flows with the same basic state and
wavenumber), life cycle with hourly output (time step halved, then quartered, if it blows up), omega + regridding,
then deletes the raw Gaussian output. `--per-gpu` jobs run at once on each of `--gpus`; logs/done/<flow>.done markers
make the driver re-runnable; step times go to logs/timings.csv and flow status to configs/flows.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, LOGS = ROOT / "data", ROOT / "logs"
SCRATCH = Path(os.environ.get("DINO_SCRATCH", ROOT / "data/scratch_runs"))
ENV = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", TMPDIR=os.environ.get("TMPDIR", str(SCRATCH)),
           OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMPY_MADVISE_HUGEPAGE="0")
LOCK = threading.Lock()
MODE_LOCKS: dict[str, threading.Lock] = {}
DTS = (600.0, 300.0, 150.0)


def run(step: str, name: str, args: list, gpu: int | None) -> int:
    """Runs one script of this folder with the environment's python; appends to logs/<name>.log and timings.csv."""
    env = dict(ENV, CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
               JAX_PLATFORMS="cpu" if gpu is None else "cuda")
    t0 = time.time()
    with open(LOGS / f"{name}.log", "a") as fh:
        fh.write(f"$ {' '.join(map(str, args))}  (GPU {gpu})\n")
        fh.flush()
        rc = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT / "scripts", env=env, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
        fh.write(f"exit {rc} after {time.time() - t0:.0f} s\n")
    with LOCK, open(LOGS / "timings.csv", "a") as fh:
        fh.write(f"{name},{step},{gpu},{time.time() - t0:.1f},{rc}\n")
    return rc


def update_flows() -> None:
    """Rewrites configs/flows.json with the status of every flow from the mode and check files."""
    spec = importlib.util.spec_from_file_location("flows", ROOT / "scripts" / "00_flows.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    flows = mod.draw_flows()
    mod.update(flows)
    tmp = ROOT / "configs" / "flows.json.part"
    tmp.write_text(json.dumps(flows, indent=1) + "\n")
    tmp.rename(ROOT / "configs" / "flows.json")


def finish(f: dict, status: str) -> None:
    if status != "ok":
        (DATA / f"check_{f['name']}.json").write_text(json.dumps(dict(status=status, flow=f["name"])) + "\n")
    (LOGS / "done" / f"{f['name']}.done").write_text(status + "\n")
    with LOCK:
        update_flows()
    print(f"{time.strftime('%H:%M:%S')} {f['name']}: {status}", flush=True)


def flow_job(f: dict, gpus: queue.Queue) -> None:
    gpu = gpus.get()
    try:
        mode = DATA / f"mode_{f['mode']}"
        with MODE_LOCKS[f["mode"]]:
            if not mode.with_suffix(".npz").exists():
                if run("mode", f"mode_{f['mode']}", ["02_normal_mode.py", "--basic", DATA / f"basic_{f['basic']}.nc",
                                                      "--m", f["m"], "--out", mode], gpu):
                    return print(f"{f['name']}: normal mode failed", flush=True)
        if pd.read_csv(mode.with_suffix(".csv")).growth_per_day.iloc[-1] <= 0:
            return finish(f, "no growing mode")
        work = SCRATCH / f["name"]
        work.mkdir(parents=True, exist_ok=True)
        state = work / "state.nc"
        for dt in DTS:
            rc = run("life_cycle", f["name"], ["03_life_cycle.py", "--basic", DATA / f"basic_{f['basic']}.nc", "--mode",
                                               mode.with_suffix(".npz"), "--amp", f["amp"], "--days", 15,
                                               "--out-hours", 1, "--dt", dt, "--out", state], gpu)
            if rc != 3:
                break
        if rc == 3:
            hit = re.findall(r"blew up: non-finite state at hour (\d+)", (LOGS / f"{f['name']}.log").read_text())
            return finish(f, f"blew up day {int(hit[-1]) / 24:.1f} (dt {DTS[-1]:g} s)" if hit else "blew up")
        if rc:
            return print(f"{f['name']}: life cycle failed (exit {rc})", flush=True)
        out, check = DATA / f"dino_{f['name']}.nc", DATA / f"check_{f['name']}.json"
        if run("omega_regrid", f["name"], ["04_omega_regrid.py", "--run", state, "--flow", f["name"], "--out", out,
                                           "--check", check], gpu):
            return print(f"{f['name']}: omega/regrid failed, raw output kept in {work}", flush=True)
        status = json.loads(check.read_text())["status"]
        if status == "ok" and out.exists():
            shutil.rmtree(work)
        finish(f, status)
    finally:
        gpus.put(gpu)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--per-gpu", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    assert 0 not in a.gpus, "GPU 0 is reserved"
    (LOGS / "done").mkdir(parents=True, exist_ok=True)
    flows = json.loads((ROOT / "configs" / "flows.json").read_text())
    if a.only:
        flows = [f for f in flows if f["name"] in a.only]
    for f in {f["basic"]: f for f in flows}.values():
        if not (DATA / f"basic_{f['basic']}.nc").exists():
            z2 = ["--z2", f["z2"]] if f["case"] == "lc2" else []
            run("basic", f"basic_{f['basic']}", ["01_basic_state.py", "--case", f["case"], *z2, "--ujet", f["ujet"],
                                                  "--out", DATA / f"basic_{f['basic']}.nc"], None)
    MODE_LOCKS.update({f["mode"]: threading.Lock() for f in flows})
    update_flows()
    todo = [f for f in flows if not (LOGS / "done" / f"{f['name']}.done").exists()]
    gpus: queue.Queue = queue.Queue()
    for _ in range(a.per_gpu):
        for g in a.gpus:
            gpus.put(g)
    print(f"{time.strftime('%H:%M:%S')} {len(todo)} flows to run on GPUs {a.gpus} x {a.per_gpu}", flush=True)
    with ThreadPoolExecutor(len(a.gpus) * a.per_gpu) as pool:
        for fut in [pool.submit(flow_job, f, gpus) for f in todo]:
            fut.result()
    update_flows()
    print(f"{time.strftime('%H:%M:%S')} finished", flush=True)


if __name__ == "__main__":
    main()
