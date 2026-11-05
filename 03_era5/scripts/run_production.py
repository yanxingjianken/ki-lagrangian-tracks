"""Stream ERA5 into LAGRANTO tracks: download -> P files -> caltra -> dataset -> delete, under a disk budget.

Hours are downloaded in time order by a pool of worker processes (scripts/era5_pfiles.py), at most --budget-gb of P
files ahead of the earliest unfinished release. A release is tracked (11_tracks.py, then 12_dataset.py) as soon as
the 157 P files of its window exist, with at most --max-releases at a time; P files older than the earliest
unfinished release are deleted. Finished releases (data/tracks/tracks_<release>.npz) are skipped, so a stopped run
resumes where it was. Progress: outputs/logs/production/status.json and one log per release.

Run detached, with the environment's python (parallel `micromamba run` calls block each other), e.g.

    setsid nohup ~/.conda/envs/era5_gcs/bin/python scripts/run_production.py --split train \
        > outputs/logs/production/run.log 2>&1 &
"""
import argparse
import json
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import era5_pfiles as ep  # noqa: E402

P_GB = 0.4655  # one P file

p = argparse.ArgumentParser()
p.add_argument("--split", choices=["train", "test", "all"], default="all")
p.add_argument("--since", help="first release time to include (e.g. 2017-01-01)")
p.add_argument("--until", help="releases before this time only")
p.add_argument("--workers", type=int, default=6)
p.add_argument("--max-releases", type=int, default=3)
p.add_argument("--jobs", type=int, default=8, help="caltra runs per release")
p.add_argument("--budget-gb", type=float, default=400.0)
a = p.parse_args()

H = np.timedelta64(1, "h")
cfg = json.loads((ROOT / "configs/releases.json").read_text())
rels = [r for r in cfg["releases"] if a.split in ("all", r["split"])]
if a.since:
    rels = [r for r in rels if np.datetime64(r["time"]) >= np.datetime64(a.since)]
if a.until:
    rels = [r for r in rels if np.datetime64(r["time"]) < np.datetime64(a.until)]
done_file = lambda r: ROOT / f"data/tracks/tracks_{r['name']}.npz"  # noqa: E731
start = {r["name"]: np.datetime64(r["time"], "h") for r in rels}
window = lambda r: start[r["name"]] + np.arange(cfg["window_h"] + 1) * H  # noqa: E731
logs = ROOT / "outputs/logs/production"
logs.mkdir(parents=True, exist_ok=True)
budget_h = int(a.budget_gb / P_GB)


def pending():
    return [r for r in rels if not done_file(r).exists()]


def hours_needed(todo):
    return np.unique(np.concatenate([window(r) for r in todo])) if todo else np.array([], "datetime64[h]")


def clean(todo):
    """Delete P files no unfinished release of this run needs (older than the earliest one, or in gaps)."""
    need = set(hours_needed(todo).astype(str))
    for f in ep.PDIR.glob("P*_*"):
        if f.name.endswith(".part"):
            continue
        t = np.datetime64(f"{f.name[1:5]}-{f.name[5:7]}-{f.name[7:9]}T{f.name[10:12]}", "h")
        if str(t) not in need:
            f.unlink()


def status(**kw):
    (logs / "status.json").write_text(json.dumps(kw | dict(updated=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))


if __name__ == "__main__":
    ep.PDIR.mkdir(parents=True, exist_ok=True)
    todo = pending()
    clean(todo)
    queue = list(hours_needed(todo))
    print(f"{len(rels)} releases selected, {len(todo)} to do, {len(queue)} hours to have", flush=True)
    pool = ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), initializer=ep.open_store)
    futures, running, failed = {}, {}, []
    mb_total, t_start = 0.0, time.time()
    while todo:
        earliest = min(start[r["name"]] for r in todo)
        while queue and len(futures) < 2 * a.workers and queue[0] < earliest + budget_h * H:
            t = queue.pop(0)
            futures[t] = pool.submit(ep.write_hour, str(t))
        for t, fu in list(futures.items()):
            if fu.done():
                mb_total += fu.result()[1]  # raises if the hour failed after its retries
                del futures[t]
        have = {f.name for f in ep.PDIR.glob("P*_*") if not f.name.endswith(".part")}
        for r in todo:
            name = r["name"]
            if name in running or len(running) >= a.max_releases:
                continue
            if all(ep.pfile_path(t).name in have for t in window(r)) and not any(t in futures for t in window(r)):
                log = open(logs / f"{name}.log", "w")
                py = sys.executable
                cmd = (f"{py} {SCRIPTS / '11_tracks.py'} --release {name} --jobs {a.jobs} && "
                       f"{py} {SCRIPTS / '12_dataset.py'} --release {name}")
                running[name] = (subprocess.Popen(["bash", "-c", cmd], stdout=log, stderr=subprocess.STDOUT), log,
                                 time.time())
        for name, (proc, log, t0) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[name]
            if proc.returncode:
                failed.append(name)
                print(f"{name} FAILED (see {logs / (name + '.log')})", flush=True)
                shutil.rmtree(ROOT / f"data/lagranto/{name}", ignore_errors=True)
            else:
                print(f"{name} done in {(time.time() - t0) / 60:.1f} min", flush=True)
        todo = [r for r in pending() if r["name"] not in failed]
        clean(todo + [r for r in rels if r["name"] in running])
        n_p = len(list(ep.PDIR.glob("P*_*")))
        elapsed = time.time() - t_start
        status(releases_selected=len(rels), releases_done=len(rels) - len(pending()), running=sorted(running),
               failed=failed, pfiles_on_disk=n_p, pfiles_gb=round(n_p * P_GB, 1), hours_queued=len(queue),
               downloaded_gb=round(mb_total / 1e3, 1), download_mb_s=round(mb_total / max(elapsed, 1), 1),
               elapsed_h=round(elapsed / 3600, 2))
        if failed and not running and not futures and all(r["name"] in failed for r in todo):
            break
        time.sleep(5)
    pool.shutdown()
    clean([])
    print(f"finished: {len(rels) - len(pending())} of {len(rels)} releases; failed {failed}", flush=True)
    (logs / ("FAILED" if failed else "DONE")).write_text(json.dumps(dict(failed=failed)))
