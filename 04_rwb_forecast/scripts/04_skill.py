"""Skill of breaking calls from forecast patch deformation, against ERA5 onsets on theta at 2 PVU.

A case is a breaking when an onset (rwb/events.py) lies within 1000 km of the patch's true centroid at the onset
time, within 24 h after the anchor time; its type is that onset's (nearest one if several). Predictors (same cases):
  truth, straight, model, ensemble mean: call when the patch elongation grows by at least G over 24 h
    (E(24 h) / E(0) >= G); type AWB if the long axis at 24 h runs SW-NE (alpha > 0), else CWB;
  ensemble: call when at least a share p_c of the members call (each with the same G); type by the members'
    majority among those that call;
  wave amplitude: call when the page's wave amplitude at the centroid at the anchor time reaches A (no type).
Thresholds (G, p_c, A) maximize the Peirce skill score on the fit year; scores are reported on the test year:
hit rate H, false alarm rate F, false alarm ratio, Peirce skill score H - F, type accuracy of the hits.
Writes outputs/skill_<fit>_<test>.json and prints the table.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "rwb"))
ROOT = Path(__file__).resolve().parents[1]
DET = ["truth", "straight", "model", "ensemble mean"]


def gc_km(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon2 - lon1)
    return 6371.0 * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1))


def labels(c, onsets, km=1000.0, window_h=24):
    """(breaking bool, type str) per case."""
    ot = np.array([np.datetime64(o["time"], "h") for o in onsets])
    brk, typ = np.zeros(len(c["release"]), bool), np.array([""] * len(c["release"]), dtype=object)
    for i, t in enumerate(c["anchor_time"]):
        t0 = np.datetime64(t, "h")
        best = np.inf
        for k in np.flatnonzero((ot > t0) & (ot <= t0 + np.timedelta64(window_h, "h"))):
            h = int((ot[k] - t0).astype(int))
            d = gc_km(c["track_lat"][i, h], c["track_lon"][i, h], onsets[k]["lat"], onsets[k]["lon"])
            if d < km and d < best:
                best, brk[i], typ[i] = d, True, onsets[k]["type"]
    return brk, typ


def scores(call, brk, ptype=None, ttype=None):
    hits, misses = (call & brk).sum(), (~call & brk).sum()
    fa, cn = (call & ~brk).sum(), (~call & ~brk).sum()
    H, F = hits / max(hits + misses, 1), fa / max(fa + cn, 1)
    out = dict(hit_rate=H, false_alarm_rate=F, false_alarm_ratio=fa / max(hits + fa, 1), peirce=H - F,
               alarms=int(call.sum()), hits=int(hits))
    if ptype is not None:
        h = call & brk
        out["type_accuracy"] = float((ptype[h] == ttype[h]).mean()) if h.any() else float("nan")
    return out


def calls(c, name, G, pc=0.5):
    names = list(c["predictors"])
    E, AL = c["E"], c["alpha"]                                    # (n, 3 horizons, predictors)
    if name in DET:
        j = names.index(name)
        r = E[:, 2, j] / E[:, 0, 0]
        return r >= G, np.where(AL[:, 2, j] > 0, "AWB", "CWB")
    mem = [k for k, s in enumerate(names) if s.startswith("member")]
    r = E[:, 2, mem] / E[:, 0, :1]
    vote = r >= G
    p = vote.mean(1)
    awb = ((AL[:, 2, mem] > 0) & vote).sum(1) >= 0.5 * np.maximum(vote.sum(1), 1)
    return p >= pc, np.where(awb, "AWB", "CWB")


def fit(c, brk, name):
    if name == "wave amplitude":
        grid = np.arange(2.0, 15.01, 0.25)
        pss = [scores(c["amp"] >= A, brk)["peirce"] for A in grid]
        return dict(A=float(grid[int(np.nanargmax(pss))]))
    Gs = np.round(np.arange(1.0, 4.01, 0.05), 2)
    pcs = [0.5] if name in DET else np.round(np.arange(0.1, 0.91, 0.1), 1)
    best = max(((scores(calls(c, name, G, pc)[0], brk)["peirce"], G, pc) for G in Gs for pc in pcs))
    return dict(G=float(best[1]), pc=float(best[2]))


def amplitude(c, year):
    """The page's wave amplitude on theta at 2 PVU at each case's centroid and anchor time (ERA5 truth fields)."""
    import predictors as PR
    q = {}
    for f in sorted((ROOT / "data/truth").glob(f"th2pvu_{year}_*.npz")):
        with np.load(f) as d:
            q.update(zip(d["times"], d["q"]))
    amp, cache = np.full(len(c["release"]), np.nan), {}
    for i, t in enumerate(c["anchor_time"]):
        if t not in q:
            continue
        if t not in cache:
            cache[t] = PR.amp_field(q[t])
        amp[i] = cache[t][int(np.clip(round(float(c["clat"][i])), 0, 90)), int(round(float(c["clon"][i]))) % 360]
    return amp


def load(year):
    c = dict(np.load(ROOT / f"outputs/cases_{year}.npz", allow_pickle=True))
    c["amp"] = amplitude(c, year)
    ev = sorted((ROOT / "data/truth").glob(f"events_*{year}*.json")) + sorted((ROOT / "data/truth").glob(f"events_{year - 1}-12*.json"))
    onsets = json.loads(ev[0].read_text())["onsets"]
    return c, onsets


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fit", type=int, default=2017)
    p.add_argument("--test", type=int, default=2021)
    p.add_argument("--min-amp", type=float, default=0.0, help="score only cases with wave amplitude >= this")
    a = p.parse_args()
    (cf, of), (ct, ot) = load(a.fit), load(a.test)
    bf, tf = labels(cf, of)
    bt, tt = labels(ct, ot)
    sel_f, sel_t = cf["amp"] >= a.min_amp, ct["amp"] >= a.min_amp
    cf = {k: (v[sel_f] if getattr(v, "shape", ())[:1] == sel_f.shape else v) for k, v in cf.items()}
    ct = {k: (v[sel_t] if getattr(v, "shape", ())[:1] == sel_t.shape else v) for k, v in ct.items()}
    bf, tf, bt, tt = bf[sel_f], tf[sel_f], bt[sel_t], tt[sel_t]
    out = dict(fit_year=a.fit, test_year=a.test, min_amp=a.min_amp, n_fit=int(len(bf)), n_test=int(len(bt)),
               base_rate_fit=float(bf.mean()), base_rate_test=float(bt.mean()),
               test_awb_share=float((tt[bt] == "AWB").mean()), predictors={})
    # reference without any field: each method's calls against the calls from the true tracks (same G as truth)
    gt = out_truth_G = fit(cf, bf, "truth")["G"]
    tcall, ttyp = calls(ct, "truth", gt)
    out["against_true_tracks"] = {}
    for name in ["straight", "model", "ensemble mean", "ensemble"]:
        th = fit(cf, calls(cf, "truth", gt)[0], name)
        call, ptype = calls(ct, name, th["G"], th["pc"])
        s2 = scores(call, tcall, ptype, ttyp)
        out["against_true_tracks"][name] = dict(thresholds=th, **{k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in s2.items()})
    for name in DET + ["ensemble", "wave amplitude"]:
        th = fit(cf, bf, name)
        if name == "wave amplitude":
            s = scores(ct["amp"] >= th["A"], bt)
        else:
            call, ptype = calls(ct, name, th["G"], th["pc"])
            s = scores(call, bt, ptype, tt)
        out["predictors"][name] = dict(thresholds=th, **{k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in s.items()})
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / f"outputs/skill_{a.fit}_{a.test}_amp{a.min_amp:g}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"fit {a.fit}: {out['n_fit']} cases, base rate {out['base_rate_fit']:.2f}; test {a.test}: {out['n_test']} cases, "
          f"base rate {out['base_rate_test']:.2f}, AWB share of breakings {out['test_awb_share']:.2f}")
    print(f"  against the true-track calls (G {gt}; {int(tcall.sum())} of {len(tcall)} cases called):")
    for name, s in out["against_true_tracks"].items():
        print(f"    {name:15s} H {s['hit_rate']:.2f}  F {s['false_alarm_rate']:.2f}  PSS {s['peirce']:.2f}  type agreement {s.get('type_accuracy', float('nan')):.2f}")
    print("  against ERA5 breaking onsets (theta on 2 PVU):")
    for name, s in out["predictors"].items():
        print(f"  {name:15s} {s['thresholds']}  H {s['hit_rate']:.2f}  F {s['false_alarm_rate']:.2f}  FAR {s['false_alarm_ratio']:.2f}"
              f"  PSS {s['peirce']:.2f}  type acc {s.get('type_accuracy', float('nan')):.2f}  alarms {s['alarms']}")
