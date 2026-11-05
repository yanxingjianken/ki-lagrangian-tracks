"""Flow set of the life-cycle runs, written to configs/flows.json.

Factorial: jet speed {35..60 m/s} x basic-state type {lc1, lc2 lobes, lc2 polar cap} x normal-mode wavenumber {4..8} x
mode amplitude {0.5, 1, 2} (1 = 1 hPa surface pressure). With numpy default_rng(2026): 4 (jet, type, m) triples are held
out with all their amplitudes (unseen flows inside the training ranges, amplitude 1); 48 training flows are drawn from
the remaining 258 combinations until every level of every factor appears at least 5 times. Four unseen flows lie
outside the ranges. `--update` fills `status`, growth rate, phase speed and the eddy-kinetic-energy peak from the
mode and check files.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UJETS, TYPES, MS, AMPS = (35, 40, 45, 50, 55, 60), ("lc1", "lc2lobes", "lc2polarcap"), (4, 5, 6, 7, 8), (0.5, 1.0, 2.0)
CASE = {"lc1": ("lc1", "none"), "lc2lobes": ("lc2", "lobes"), "lc2polarcap": ("lc2", "polarcap")}
OUTSIDE = [("lc1", 30, 6, "jet 30 m/s"), ("lc2lobes", 65, 5, "jet 65 m/s"), ("lc2polarcap", 45, 9, "wavenumber 9"),
           ("lc1", 65, 3, "jet 65 m/s and wavenumber 3")]
N_TRAIN, MIN_COUNT, N_HELD = 48, 5, 4


def amp_tag(amp: float) -> str:
    return {0.5: "05", 1.0: "1", 2.0: "2"}[amp]


def entry(name: str, typ: str, ujet: int, m: int, amp: float, split: str, label: str) -> dict:
    case, z2 = CASE[typ]
    return dict(name=name, case=case, z2=z2, ujet=float(ujet), m=int(m), amp=float(amp), split=split, label=label,
                basic=f"{typ}_u{ujet}", mode=f"{typ}_u{ujet}_m{m}", status="pending")


def draw_flows(seed: int = 2026) -> list[dict]:
    """The 56 flows: 48 training, 4 held-out factorial triples, 4 outside the ranges."""
    rng = np.random.default_rng(seed)
    triples = list(itertools.product(UJETS, TYPES, MS))
    held = sorted(int(i) for i in rng.choice(len(triples), N_HELD, replace=False))
    pool = [(u, t, m, a) for i, (u, t, m) in enumerate(triples) if i not in held for a in AMPS]
    levels = (UJETS, TYPES, MS, AMPS)
    while True:
        pick = [pool[int(i)] for i in rng.choice(len(pool), N_TRAIN, replace=False)]
        counts = [Counter(p[k] for p in pick) for k in range(4)]
        if all(counts[k][lv] >= MIN_COUNT for k in range(4) for lv in levels[k]):
            break
    pick.sort(key=lambda p: (TYPES.index(p[1]), p[0], p[2], p[3]))
    flows = [entry(f"t{k + 1:02d}_{t}_u{u}_m{m}_a{amp_tag(a)}", t, u, m, a, "train", "training")
             for k, (u, t, m, a) in enumerate(pick)]
    for k, i in enumerate(held):
        u, t, m = triples[i]
        flows.append(entry(f"o{k + 1}_{t}_u{u}_m{m}_a1", t, u, m, 1.0, "test",
                           "new combination inside the training ranges"))
    for k, (t, u, m, what) in enumerate(OUTSIDE):
        flows.append(entry(f"o{k + 5}_{t}_u{u}_m{m}_a1", t, u, m, 1.0, "test", f"outside the training ranges: {what}"))
    return flows


def update(flows: list[dict]) -> None:
    """Fills status and summary numbers from data/mode_<mode>.csv and data/check_<name>.json."""
    for f in flows:
        mode_csv, check = ROOT / "data" / f"mode_{f['mode']}.csv", ROOT / "data" / f"check_{f['name']}.json"
        if mode_csv.exists():
            df = pd.read_csv(mode_csv)
            f["growth_per_day"] = round(float(df.growth_per_day.iloc[-1]), 4)
            f["phase_speed_deg_per_day"] = round(float(df.phase_speed_deg_per_day.iloc[-1]), 3)
            f["mode_converged"] = bool(df.converged.iloc[-1])
            if f["growth_per_day"] <= 0:
                f["status"] = "no growing mode"
        if check.exists() and f["status"] == "pending":
            c = json.loads(check.read_text())
            f["status"] = c["status"]
            f["dt_s"] = c.get("dt_s", 600.0)
            f["eke_peak_day"] = c.get("eke_peak_day")
            f["eke_peak_j_m2"] = c.get("eke_peak_j_m2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "configs" / "flows.json")
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    flows = draw_flows()
    if a.update:
        update(flows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".json.part")
    tmp.write_text(json.dumps(flows, indent=1) + "\n")
    tmp.rename(a.out)


if __name__ == "__main__":
    main()
