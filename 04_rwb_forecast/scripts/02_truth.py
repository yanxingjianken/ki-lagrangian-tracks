"""ERA5 truth for the breaking forecasts: theta on 2 PVU every 6 h, breaking objects and onsets (RWB page code).

Stage `surfaces` (env with gcsfs): u, v, T on the page's 10 levels (70-500 hPa) from the ARCO ERA5 1-degree hourly
store (1959-2021) every 6 h, Ertel PV and theta on 2 PVU with rwb/surfaces.py -> data/truth/th2pvu_<year>_<month>.npz
(times, q = -theta, u, v on 0-90 N x 0-359 E). Stage `events` (env with scikit-image): rwb/events.py objects at every
time and their onsets over the whole period -> data/truth/events_<period>.json.

    python scripts/02_truth.py surfaces 2017-01-01 2018-01-10
    python scripts/02_truth.py events 2017-01-01 2018-01-10
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "rwb"))
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/truth"
STORE = "gcp-public-data-arco-era5/ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr"
VARS = {"u": "u_component_of_wind", "v": "v_component_of_wind", "t": "temperature"}


def month_surfaces(month: str):
    """month 'YYYY-MM' (clipped to the requested period by the caller) -> npz of theta on 2 PVU every 6 h."""
    import gcsfs
    import xarray as xr
    from config import LEVELS_HPA
    from surfaces import surfaces
    m0, m1, a0, a1 = month
    path = OUT / f"th2pvu_{m0[:7].replace('-', '_')}.npz"
    if path.exists():
        return path.name
    ds = xr.open_zarr(gcsfs.GCSFileSystem(token="anon").get_mapper(STORE), chunks=None)
    times = np.arange(np.datetime64(max(m0, a0), "h"), np.datetime64(min(m1, a1), "h"), np.timedelta64(6, "h"))
    sub = {k: ds[v].sel(time=times, level=list(LEVELS_HPA)).sel(latitude=slice(0, 90)) for k, v in VARS.items()}
    arr = {k: s.transpose("time", "level", "latitude", "longitude").values.astype(np.float32) for k, s in sub.items()}
    lat = sub["u"].latitude.values
    assert np.allclose(lat, np.arange(0, 90.01, 1.0)) and np.allclose(sub["u"].longitude.values, np.arange(360.0))
    q, u, v = [], [], []
    for i in range(len(times)):
        s = surfaces(arr["u"][i], arr["v"][i], arr["t"][i])["th2pvu"]
        q.append(s[0]); u.append(s[1]); v.append(s[2])
    tmp = path.with_suffix(".part.npz")
    np.savez(tmp, times=times.astype("datetime64[h]").astype(str), q=np.stack(q), u=np.stack(u), v=np.stack(v))
    tmp.rename(path)
    return path.name


def months(a0, a1):
    m = np.datetime64(a0[:7], "M")
    while m < np.datetime64(a1, "D"):
        yield (str(m.astype("datetime64[D]")), str((m + 1).astype("datetime64[D]")), a0, a1)
        m += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["surfaces", "events"])
    p.add_argument("start")
    p.add_argument("end")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "surfaces":
        with ProcessPoolExecutor(a.workers) as pool:
            for name in pool.map(month_surfaces, list(months(a.start, a.end))):
                print(name, flush=True)
    else:
        import events as E
        seq, fields = [], sorted(OUT.glob("th2pvu_*.npz"))
        for f in fields:
            with np.load(f) as d:
                for k, t in enumerate(d["times"]):
                    if a.start <= t[:10] < a.end:
                        objs = E.objects(d["q"][k], d["u"][k], d["v"][k], "th2pvu")
                        seq.append((t, [{k2: o[k2] for k2 in ("lon", "lat", "type", "area", "flank")} for o in objs]))
        seq.sort(key=lambda s: s[0])
        on = [dict(time=t, **o) for t, o in E.onsets(seq)]
        name = OUT / f"events_{a.start}_{a.end}.json"
        name.write_text(json.dumps(dict(objects=[dict(time=t, objects=o) for t, o in seq], onsets=on)))
        print(f"{len(seq)} times, {sum(len(o) for _, o in seq)} objects, {len(on)} onsets "
              f"({sum(o['type'] == 'AWB' for o in on)} AWB, {sum(o['type'] == 'CWB' for o in on)} CWB) -> {name.name}")
