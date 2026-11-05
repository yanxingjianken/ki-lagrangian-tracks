"""LAGRANTO input/output for 2-D (model-level, caltra -2d) trajectories on a regular lon-lat grid.

P files follow the IVE netCDF layout read by lib/ioinp_ive.f (the reader built by build.sh).
"""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np

MDV = -999.0
LEVELS_HPA = (600.0, 400.0)
PS_HPA = 1000.0


def _regular(x: np.ndarray, name: str) -> float:
    d = np.diff(x)
    if x.size < 2 or not np.allclose(d, d[0], rtol=0, atol=1e-4 * abs(d[0])):
        raise ValueError(f"{name} must be equally spaced (LAGRANTO needs a regular grid)")
    return float(d[0])


def _ive_var(nc: netCDF4.Dataset, name: str, dims: tuple[str, ...], box: tuple[float, ...]) -> netCDF4.Variable:
    var = nc.createVariable(name, "f4", dims)
    for key, val in zip(("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"), box):
        var.setncattr(key, np.float32(val))
    for key in ("xstag", "ystag", "zstag"):
        var.setncattr(key, np.float32(0.0))
    var.setncattr("missing_data", np.float32(MDV))
    return var


def write_pfiles(outdir: str | Path, times: np.ndarray, lon: np.ndarray, lat: np.ndarray, u: np.ndarray,
                 v: np.ndarray, levels: tuple[float, float] = LEVELS_HPA, ps: float = PS_HPA) -> list[Path]:
    """Write one LAGRANTO P file per time for 2-D trajectories.

    The 2-D wind is copied to two pressure levels (``levels``, bottom first) so that both ``caltra -2d`` and
    ``trace`` accept the files. A global longitude axis (n * dx = 360) is closed by appending the first column at
    lon[0] + 360, which is what makes caltra treat the domain as periodic. Latitudes may be given N->S or S->N;
    files are written S->N. Include both poles (lat -90..90) for a global run: parcels leave the domain beyond
    the first and last latitude row.

    Args:
        outdir: Directory for the files (created if missing); files are named ``PYYYYMMDD_HH``.
        times: Valid times, numpy datetime64, shape (nt,); must fall on full hours.
        lon: Longitudes in degrees, equally spaced and increasing, shape (nx,).
        lat: Latitudes in degrees, equally spaced, shape (ny,).
        u: Zonal wind in m/s, shape (nt, ny, nx).
        v: Meridional wind in m/s, shape (nt, ny, nx).
        levels: Pressures in hPa of the two identical model levels, bottom (larger) first.
        ps: Surface pressure in hPa; must exceed both levels.

    Returns:
        Paths of the written files.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    times = np.asarray(times).astype("datetime64[m]")
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    nt, ny, nx = times.size, lat.size, lon.size
    if u.shape != (nt, ny, nx) or v.shape != (nt, ny, nx):
        raise ValueError(f"u, v must have shape (nt, ny, nx) = {(nt, ny, nx)}, got {u.shape}, {v.shape}")
    if np.any(times.astype("datetime64[h]") != times):
        raise ValueError("times must fall on full hours (P file names carry YYYYMMDD_HH)")
    if not (levels[0] > levels[1] and ps > levels[0]):
        raise ValueError("need ps > levels[0] > levels[1] (hPa)")
    dx = _regular(lon, "lon")
    dy = _regular(lat, "lat")
    if dx <= 0:
        raise ValueError("lon must increase")
    if dy < 0:
        lat, u, v, dy = lat[::-1], u[:, ::-1], v[:, ::-1], -dy
    if abs(nx * dx - 360.0) < 1e-4 * dx:
        lon = np.append(lon, lon[0] + 360.0)
        u = np.concatenate([u, u[:, :, :1]], axis=2)
        v = np.concatenate([v, v[:, :, :1]], axis=2)
    nxc = lon.size
    box = (lon[0], lon[-1], lat[0], lat[-1], levels[0], levels[1])
    box_ps = (lon[0], lon[-1], lat[0], lat[-1], 0.0, 0.0)
    plev = np.broadcast_to(np.asarray(levels, np.float32)[:, None, None], (2, ny, nxc))
    hours = ((times - times[0]) / np.timedelta64(1, "m") / 60.0).astype(np.float32)
    paths = []
    for it in range(nt):
        name = "P" + times[it].astype(object).strftime("%Y%m%d_%H")
        path = outdir / name
        with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as nc:
            nc.createDimension("time", 1)
            nc.createDimension("nz", 2)
            nc.createDimension("nz1", 1)
            nc.createDimension("ny", ny)
            nc.createDimension("nx", nxc)
            nc.setncattr("constants_file_name", name)
            nc.createVariable("time", "f4", ("time",))[:] = hours[it]
            for key in ("aklev", "aklay"):
                nc.createVariable(key, "f4", ("nz",))[:] = np.asarray(levels, np.float32)
            for key in ("bklev", "bklay"):
                nc.createVariable(key, "f4", ("nz",))[:] = 0.0
            nc.createVariable("pollon", "f4")[...] = 0.0
            nc.createVariable("pollat", "f4")[...] = 90.0
            dims = ("time", "nz", "ny", "nx")
            _ive_var(nc, "U", dims, box)[0] = np.broadcast_to(u[it], (2, ny, nxc))
            _ive_var(nc, "V", dims, box)[0] = np.broadcast_to(v[it], (2, ny, nxc))
            _ive_var(nc, "P", dims, box)[0] = plev
            _ive_var(nc, "PS", ("time", "nz1", "ny", "nx"), box_ps)[0, 0] = np.full((ny, nxc), ps, np.float32)
        paths.append(path)
    return paths


def startfile(path: str | Path, lon: np.ndarray, lat: np.ndarray, p: float | np.ndarray = 500.0,
              lon_min: float | None = None) -> Path:
    """Write a caltra start file (one ``lon lat p`` line per parcel).

    Args:
        path: Output file. Its name must not contain a dot followed by a digit (caltra.sh reads that as a
            trajectory-format suffix) and should be short: caltra keeps 80 characters of the path.
        lon: Start longitudes in degrees, shape (n,).
        lat: Start latitudes in degrees, shape (n,).
        p: Start pressure in hPa; for 2-D runs any value between the two P-file levels.
        lon_min: If given, longitudes are wrapped into [lon_min, lon_min + 360), e.g. the first P-file longitude.

    Returns:
        The path written.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    p = np.broadcast_to(np.asarray(p, dtype=np.float64), lon.shape)
    if lon_min is not None:
        lon = lon_min + np.mod(lon - lon_min, 360.0)
    path = Path(path)
    np.savetxt(path, np.column_stack([lon, lat, p]), fmt="%.6f %.6f %.3f")
    return path


def _hhmm_to_hours(t: np.ndarray) -> np.ndarray:
    a = np.abs(t.astype(np.float64))
    hh = np.floor(a + 1e-5)
    return np.sign(t) * (hh + np.round(100.0 * (a - hh)) / 60.0)


def _fortran_records(raw: bytes) -> list[bytes]:
    recs, pos = [], 0
    while pos < len(raw):
        chunks = []
        while True:
            n = int(np.frombuffer(raw, "<i4", 1, pos)[0])
            chunks.append(raw[pos + 4:pos + 4 + abs(n)])
            pos += abs(n) + 8
            if n >= 0:
                break
        recs.append(b"".join(chunks))
    return recs


def read_traj(path: str | Path) -> dict[str, np.ndarray]:
    """Read a LAGRANTO trajectory file in ASCII format 1 (``.1``/``.ls``/no suffix) or binary format 3 (``.3``).

    Format 3 keeps full single precision; format 1 rounds lon/lat to 0.01 deg and p to 1 hPa.

    Args:
        path: Trajectory file written by caltra or trace.

    Returns:
        Dict with ``time`` (hours since the reference date, shape (ntime,)), ``lon``, ``lat``, ``p`` and one entry
        per traced column, each shape (ntime, ntraj); ``refdate`` (numpy datetime64) and ``columns`` (names).
        Positions of parcels that left the domain are NaN (LAGRANTO writes -999).
    """
    path = Path(path)
    if path.suffix == ".3" or path.suffix == ".du":
        recs = _fortran_records(path.read_bytes())
        ntra, ntim, ncol = np.frombuffer(recs[0], "<i4")
        ref = np.frombuffer(recs[1], "<i4")
        names = [recs[2][80 * i:80 * (i + 1)].decode().strip() for i in range(ncol)]
        tra = np.frombuffer(recs[3], "<f4").reshape((ncol, ntim, ntra)).transpose(1, 2, 0)
    else:
        lines = path.read_text().splitlines()
        head = lines[0].split()
        stamp = head[2]
        ref = np.array([int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]), int(stamp[9:11]), int(stamp[11:13])])
        names = lines[2].split()
        blocks, cur = [], []
        for line in lines[4:]:
            if line.strip():
                cur.append([float(x) for x in line.split()])
            elif cur:
                blocks.append(cur)
                cur = []
        if cur:
            blocks.append(cur)
        tra = np.asarray(blocks, dtype=np.float64).transpose(1, 0, 2)
    tra = np.asarray(tra, dtype=np.float64)
    refdate = np.datetime64(f"{ref[0]:04d}-{ref[1]:02d}-{ref[2]:02d}T{ref[3]:02d}:{ref[4]:02d}")
    out: dict[str, np.ndarray] = {"time": _hhmm_to_hours(tra[:, 0, 0]), "refdate": refdate,
                                  "columns": np.array(names)}
    for j, name in enumerate(names[1:], start=1):
        col = tra[:, :, j].copy()
        col[np.abs(col - MDV) < 1e-3] = np.nan
        out[name] = col
    return out
