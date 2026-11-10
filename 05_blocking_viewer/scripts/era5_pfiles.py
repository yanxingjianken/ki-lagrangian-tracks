# Copy of roger/03_era5/scripts/era5_pfiles.py.
"""One hour of ERA5 from Google's ARCO store -> one LAGRANTO P file on the 37 pressure levels.

Store: gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3, read anonymously. Each variable-hour is
one chunk covering the globe and all 37 levels (154 MB as float32), so the whole field is read. Used by
run_production.py (one store per worker process) and as a script:

    python scripts/era5_pfiles.py 2017-01-05T00 2017-01-05T03 [outdir]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "lagranto/python"))
import lagranto_io as lio  # noqa: E402

STORE = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VARS = {"U": "u_component_of_wind", "V": "v_component_of_wind", "OMEGA": "vertical_velocity",
        "PS": "surface_pressure"}
PDIR = ROOT / "data/pfiles"
_DS = None


def open_store():
    """Open the store once per process (anonymous access, no credentials)."""
    global _DS
    if _DS is None:
        import gcsfs
        import xarray as xr
        fs = gcsfs.GCSFileSystem(token="anon")
        _DS = xr.open_zarr(fs.get_mapper(STORE), chunks=None)
    return _DS


def pfile_path(t: np.datetime64, outdir: Path = PDIR) -> Path:
    return Path(outdir) / lio._pname(np.datetime64(t, "h"))


def write_hour(t: str | np.datetime64, outdir: str | Path = PDIR, tries: int = 4) -> tuple[str, float, float]:
    """Download hour t and write its P file unless it exists. Returns (file name, MB read, seconds)."""
    t = np.datetime64(t, "h")
    path = pfile_path(t, outdir)
    if path.exists():
        return path.name, 0.0, 0.0
    ds = open_store()
    for attempt in range(tries):
        try:
            t0 = time.time()
            f = {k: ds[v].sel(time=t).values for k, v in VARS.items()}
            if not all(np.isfinite(a).all() for a in f.values()):
                raise ValueError(f"non-finite ERA5 values at {t}")
            lio.write_pfile_plev(outdir, t, ds.longitude.values, ds.latitude.values, ds.level.values,
                                 f["U"], f["V"], f["OMEGA"], f["PS"])
            return path.name, sum(a.nbytes for a in f.values()) / 1e6, time.time() - t0
        except (OSError, TimeoutError, RuntimeError) as err:  # network hiccups: back off and retry
            if attempt == tries - 1:
                raise
            print(f"{t}: {type(err).__name__} {err}; retry {attempt + 1}", flush=True)
            time.sleep(10 * 2**attempt)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    t0, t1 = np.datetime64(sys.argv[1], "h"), np.datetime64(sys.argv[2], "h")
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else PDIR
    for t in np.arange(t0, t1 + np.timedelta64(1, "h"), np.timedelta64(1, "h")):
        name, mb, sec = write_hour(t, out)
        print(name, f"{mb:.0f} MB in {sec:.1f} s", flush=True)
