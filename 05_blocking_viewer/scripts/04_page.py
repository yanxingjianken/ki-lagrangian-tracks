"""Build the page from page/rwb-prediction.template.html: every number comes from result files.

Numbers: 2021 medians of the 24-h forecast/backcast errors (model and straight motion, case "2021, unseen year") from
03_era5/outputs/eval/context_ki_era5.csv.gz; ensemble-mean and deterministic 24-h medians on the 2015-2017
validation parcels from 04_rwb_forecast/outputs/calibration.csv (the chosen eps_rel of configs/ensemble.json);
training tracks counted from the 03_era5 datasets (training patch ids); patches, parcels and launch-time ranges
from configs/releases.json and pkg/<event>/index.json; the in-browser settings (parcels launched, members, steps per
hour) from hf_model/onnx/config.json, the download size from the ONNX files and the time per launch from
outputs/wasm_timing.json (onnxruntime-web, one thread). The script page/viewer.js is inlined. Writes the page to
--out (default: _pages/rwb-prediction.html in the site repository named by the environment variable SITE_REPO).
"""
import argparse
import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R03, R04 = ROOT.parent / "03_era5", ROOT.parent / "04_rwb_forecast"
SITE = Path(os.environ.get("SITE_REPO", ".")) / "_pages" / "rwb-prediction.html"


def training_tracks():
    cfg = json.loads((R03 / "configs/releases.json").read_text())
    ids = cfg["split_by_patch"]["train"]
    n = 0
    for r in cfg["releases"]:
        if r["split"] == "train":
            with np.load(R03 / f"data/tracks/tracks_{r['name']}.npz") as f:
                n += int(np.isin(f["patch"], ids).sum())
    return n


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(SITE))
    a = p.parse_args()
    ev = json.loads((ROOT / "configs/events.json").read_text())
    rel = json.loads((ROOT / "configs/releases.json").read_text())
    ens = json.loads((R04 / "configs/ensemble.json").read_text())
    df = pd.read_csv(R03 / "outputs/eval/context_ki_era5.csv.gz", usecols=["case", "task", "horizon_h", "model", "error_km"])
    d24 = df[(df.case == "2021, unseen year") & (df.horizon_h == 24)].groupby(["task", "model"])["error_km"].median()
    cal = pd.read_csv(R04 / "outputs/calibration.csv")
    c24 = cal[cal.horizon_h == 24].set_index("eps_rel")["median_err_km"]
    events, rows, n_patches = [], [], 0
    ready = [n for n in ev if (ROOT / f"pkg/{n}/centroids.bin").exists()]     # events packaged so far
    for name, e in ((n, ev[n]) for n in ready):
        rs = [r for r in rel["releases"] if r["event"] == name]
        idx = json.loads((ROOT / f"pkg/{name}/index.json").read_text())
        n_patches += len(idx["patches"])
        tmin = str(np.datetime64(rs[0]["time"], "h") + np.timedelta64(48, "h"))
        tmax = str(np.datetime64(rs[-1]["time"], "h") + np.timedelta64(108, "h"))
        events.append(dict(id=name, label=e["label"], region_lat=e["region_lat"], region_lon=e["region_lon"],
                           sigma=rel["sigma"], tmin=tmin, tmax=tmax))
        rows.append(f"<tr><td>{html.escape(e['label'])}</td><td>{tmin[:10]} {tmin[11:13]} to {tmax[:10]} {tmax[11:13]}</td>"
                    f"<td>{html.escape(e['basis'])}</td></tr>")
    rep = {
        "N_PER": str(rel["n_per"]),
        "SIGMA_TEXT": ", ".join(f"{s:g}" for s in rel["sigma"][:-1]) + f" and {rel['sigma'][-1]:g}",
        "N_PATCHES": f"{n_patches:,}", "N_PARCELS": f"{n_patches * rel['n_per'] / 1e6:.1f} million",
        "N_TRAIN": f"{training_tracks() / 1e6:.1f} million",
        "FC24": f"{d24[('forecast', 'conditional KI')]:.0f}", "BC24": f"{d24[('backcast', 'conditional KI')]:.0f}",
        "FC24S": f"{d24[('forecast', 'no learning (straight motion)')]:.0f}",
        "BC24S": f"{d24[('backcast', 'no learning (straight motion)')]:.0f}",
        "ENS24": f"{c24[ens['eps_rel']]:.0f}", "DET24": f"{c24[0.0]:.0f}", "CAL_MEMBERS": str(ens["members"]),
        "EVENTS_JSON": json.dumps(events), "EVENT_ROWS": "\n      ".join(rows),
    }
    ox = json.loads((ROOT / "hf_model/onnx/config.json").read_text())
    wt = json.loads((ROOT / "outputs/wasm_timing.json").read_text())
    q = ox["members"] * ox["launch_parcels"]
    ms = wt["decoder_ms"][str(q)]
    launch_s = 2 * (24 * ox["steps_per_h"] * ms + wt["encoder_ms"]) / 1000
    mb = sum((ROOT / "hf_model/onnx" / ox["graphs"][g]).stat().st_size for g in ("encoder", "decoder")) / 1e6
    rep["N_LAUNCH"] = str(ox["launch_parcels"])
    rep["MEMBERS"] = str(ox["members"])
    rep["MODEL_MB"] = f"{mb:.0f}"
    rep["LAUNCH_S"] = f"about {launch_s:.0f}&nbsp;s"
    js = (ROOT / "page/viewer.js").read_text().replace("@@EVENTS_JSON@@", rep["EVENTS_JSON"])
    rep["SCRIPT"] = '{% raw %}<script type="module">\n' + js + "</script>{% endraw %}"
    page = (ROOT / "page/rwb-prediction.template.html").read_text()
    for k, v in rep.items():
        page = page.replace(f"@@{k}@@", v)
    left = [s for s in page.split("@@")[1::2] if s.isupper()]
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    Path(a.out).write_text(page)
    print("events:", ready)
    print("wrote", a.out, {k: v for k, v in rep.items() if k not in ("EVENTS_JSON", "EVENT_ROWS", "SCRIPT")})
