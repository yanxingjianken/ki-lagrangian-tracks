"""LAGRANTO input/output on a regular lon-lat grid: P files for 2-D (``caltra -2d``) and 3-D trajectories, start
files, trajectory files and the caltra log.

P files follow the IVE netCDF layout read by lib/ioinp_ive.f (the reader built by build.sh).
"""

from __future__ import annotations

import re
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
    atts = dict(zip(("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"), box)) | dict(xstag=0.0, ystag=0.0, zstag=0.0)
    var.setncatts({key: np.float32(val) for key, val in (atts | {"missing_data": MDV}).items()})
    return var


def _times(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Times as datetime64[m] and as hours since the first time (float32)."""
    times = np.asarray(times).astype("datetime64[m]")
    if np.any(times.astype("datetime64[h]") != times):
        raise ValueError("times must fall on full hours (P file names carry YYYYMMDD_HH)")
    return times, ((times - times[0]) / np.timedelta64(1, "m") / 60.0).astype(np.float32)


def _horizontal(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """Grid as written: (lon, lat, flip_lat, close_ring); lat S->N, a global lon axis closed at lon[0] + 360."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    dx, dy = _regular(lon, "lon"), _regular(lat, "lat")
    if dx <= 0:
        raise ValueError("lon must increase")
    flip = dy < 0
    close = abs(lon.size * dx - 360.0) < 1e-4 * dx
    return (np.append(lon, lon[0] + 360.0) if close else lon), (lat[::-1] if flip else lat), flip, close


def _orient(a: np.ndarray, flip: bool, close: bool) -> np.ndarray:
    """Field (..., ny, nx) as written: float32, rows S->N, closed longitude ring."""
    a = np.asarray(a, dtype=np.float32)
    if flip:
        a = a[..., ::-1, :]
    if close:
        a = np.concatenate([a, a[..., :1]], axis=-1)
    return np.ascontiguousarray(a)


def _write_pfile(path: Path, hour: float, lon: np.ndarray, lat: np.ndarray, ak: np.ndarray, bk: np.ndarray,
                 fields: dict[str, np.ndarray], ps: np.ndarray, zbox: tuple[float, float]) -> None:
    """Write one IVE P file: fields (nz, ny, nx), PS (ny, nx) in hPa, and the constants block (ak in hPa, bk)."""
    nz, ny, nx = next(iter(fields.values())).shape
    for name, f in [*fields.items(), ("PS", ps)]:
        if not np.isfinite(f).all():
            raise ValueError(f"{name} has non-finite values in {path.name}")
    box = (lon[0], lon[-1], lat[0], lat[-1])
    # Built in memory and written in one go: each NETCDF3 define step on disk rewrites the file (1 s per 8-MB file
    # on NFS instead of 0.04 s). memory= rather than diskless=True: netCDF4 1.7.4 drops diskless files on close.
    nc = netCDF4.Dataset(path.name, "w", format="NETCDF3_CLASSIC", memory=1)
    try:
        nc.set_fill_off()
        for dim, size in (("time", 1), ("nz", nz), ("nz1", 1), ("ny", ny), ("nx", nx)):
            nc.createDimension(dim, size)
        nc.setncattr("constants_file_name", path.name)
        nc.createVariable("time", "f4", ("time",))[:] = hour
        for key in ("aklev", "aklay"):
            nc.createVariable(key, "f4", ("nz",))[:] = ak
        for key in ("bklev", "bklay"):
            nc.createVariable(key, "f4", ("nz",))[:] = bk
        nc.createVariable("pollon", "f4")[...] = 0.0
        nc.createVariable("pollat", "f4")[...] = 90.0
        for name, f in fields.items():
            _ive_var(nc, name, ("time", "nz", "ny", "nx"), box + zbox)[0] = f
        _ive_var(nc, "PS", ("time", "nz1", "ny", "nx"), box + (0.0, 0.0))[0, 0] = ps
    finally:
        buf = nc.close()
    path.write_bytes(buf)


def _pname(time: np.datetime64) -> str:
    return "P" + time.astype(object).strftime("%Y%m%d_%H")


def write_pfiles(outdir: str | Path, times: np.ndarray, lon: np.ndarray, lat: np.ndarray, u: np.ndarray,
                 v: np.ndarray, levels: tuple[float, float] = LEVELS_HPA, ps: float = PS_HPA) -> list[Path]:
    """Write one LAGRANTO P file per time for 2-D trajectories (``caltra -2d``).

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
    times, hours = _times(times)
    nt, ny, nx = times.size, np.size(lat), np.size(lon)
    if np.shape(u) != (nt, ny, nx) or np.shape(v) != (nt, ny, nx):
        raise ValueError(f"u, v must have shape (nt, ny, nx) = {(nt, ny, nx)}, got {np.shape(u)}, {np.shape(v)}")
    if not (levels[0] > levels[1] and ps > levels[0]):
        raise ValueError("need ps > levels[0] > levels[1] (hPa)")
    lon, lat, flip, close = _horizontal(lon, lat)
    ak = np.asarray(levels, np.float32)
    plev = np.broadcast_to(ak[:, None, None], (2, ny, lon.size))
    ps2 = np.full((ny, lon.size), ps, np.float32)
    paths = []
    for it in range(nt):
        path = outdir / _pname(times[it])
        fields = {"U": np.broadcast_to(_orient(u[it], flip, close), plev.shape),
                  "V": np.broadcast_to(_orient(v[it], flip, close), plev.shape), "P": plev}
        _write_pfile(path, hours[it], lon, lat, ak, np.zeros(2, np.float32), fields, ps2, levels)
        paths.append(path)
    return paths


def write_pfiles_3d(outdir: str | Path, times: np.ndarray, lon: np.ndarray, lat: np.ndarray, sigma: np.ndarray,
                    u: np.ndarray, v: np.ndarray, omega: np.ndarray, ps: np.ndarray,
                    t: np.ndarray | None = None) -> list[Path]:
    """Write one LAGRANTO P file per time for 3-D trajectories from a sigma-coordinate model.

    caltra (3-D mode) reads U, V, OMEGA and PS and builds the pressure of level k as p = ak(k) + bk(k) * PS
    (hPa); for pure sigma levels ak = 0 and bk = sigma, written to both the level (aklev, bklev) and layer (aklay,
    bklay) arrays because caltra uses one or the other depending on the call. Levels are written bottom first
    (pressure decreasing with index, which caltra's level search assumes), PS in hPa, latitudes S->N, and a
    global longitude axis (n * dx = 360) closed at lon[0] + 360. OMEGA stays in Pa/s: run caltra with
    wfactor = 1 (the default of the patched caltra.sh) and confirm it with ``caltra_wfactor``.

    Arrays are converted one time at a time, so numpy memmaps or lazily indexed arrays keep memory small.

    Args:
        outdir: Directory for the files (created if missing); files are named ``PYYYYMMDD_HH``.
        times: Valid times, numpy datetime64, shape (nt,); must fall on full hours.
        lon: Longitudes in degrees, equally spaced and increasing, shape (nx,).
        lat: Latitudes in degrees, equally spaced, either order, shape (ny,).
        sigma: Sigma of the model levels, 0 < sigma < 1, strictly monotonic in either order (top to bottom or
            bottom to top), shape (nlev,). It gives the level order of ``u``, ``v``, ``omega`` and ``t``.
        u: Zonal wind in m/s, shape (nt, nlev, ny, nx).
        v: Meridional wind in m/s, shape (nt, nlev, ny, nx).
        omega: Pressure vertical velocity Dp/Dt in Pa/s, shape (nt, nlev, ny, nx).
        ps: Surface pressure in Pa, shape (nt, ny, nx).
        t: Optional temperature in K, shape (nt, nlev, ny, nx), written as T for ``trace``.

    Returns:
        Paths of the written files.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    times, hours = _times(times)
    sigma = np.asarray(sigma, dtype=np.float64)
    nt, nlev, ny, nx = times.size, sigma.size, np.size(lat), np.size(lon)
    ds = np.diff(sigma)
    if sigma.ndim != 1 or nlev < 2 or not (np.all(ds > 0) or np.all(ds < 0)):
        raise ValueError("sigma must be 1-D, with at least 2 strictly monotonic values")
    if not (sigma.min() > 0.0 and sigma.max() < 1.0):
        raise ValueError("sigma must lie in (0, 1): a level at the surface breaks caltra's surface-layer search")
    fields = {"U": u, "V": v, "OMEGA": omega} | ({"T": t} if t is not None else {})
    for name, a in fields.items():
        if np.shape(a) != (nt, nlev, ny, nx):
            raise ValueError(f"{name} must have shape (nt, nlev, ny, nx) = {(nt, nlev, ny, nx)}, got {np.shape(a)}")
    if np.shape(ps) != (nt, ny, nx):
        raise ValueError(f"ps must have shape (nt, ny, nx) = {(nt, ny, nx)}, got {np.shape(ps)}")
    lev = slice(None) if ds[0] < 0 else slice(None, None, -1)  # bottom (largest sigma) first
    bk = sigma[lev].astype(np.float32)
    lon, lat, flip, close = _horizontal(lon, lat)
    paths = []
    for it in range(nt):
        ps_hpa = _orient(ps[it], flip, close) / np.float32(100.0)
        if not (100.0 < ps_hpa.min() and ps_hpa.max() < 1200.0):
            raise ValueError(f"ps at time {it} spans {ps_hpa.min() * 100:.4g} to {ps_hpa.max() * 100:.4g}; give Pa")
        path = outdir / _pname(times[it])
        _write_pfile(path, hours[it], lon, lat, np.zeros(nlev, np.float32), bk,
                     {name: _orient(np.asarray(a[it])[lev], flip, close) for name, a in fields.items()}, ps_hpa,
                     (float(bk[0]) * PS_HPA, float(bk[-1]) * PS_HPA))
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
        p: Start pressure in hPa; for 2-D runs any value between the two P-file levels, for 3-D runs from sigma
            levels sigma * ps.
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


_WFACTOR_LINES = {"caltra.sh": re.compile(r"Vertical scaling factor \(wfactor\)\s*:\s*(\S+)"),
                  "caltra": re.compile(r"Vertical wind \(scale\)\s*:\s*(\S+)")}


def caltra_wfactor(log: str | Path, expect: float | None = 1.0) -> float:
    """Vertical-velocity scaling factor of a 3-D caltra run, read from its standard output.

    caltra advances pressure as p1 = p0 + wfactor * OMEGA * dt / 100 (p in hPa, dt in s), so OMEGA in Pa/s needs
    wfactor = 1. The factor is read from caltra.sh's "Vertical scaling factor (wfactor)" line (printed in 3-D mode
    only) and from caltra's "Vertical wind (scale)" line; both must be present and agree.

    Args:
        log: caltra's standard output as text, or a Path to a file holding it.
        expect: Raise if the factor differs from this value; None skips the check.

    Returns:
        The factor.

    Raises:
        ValueError: A line is missing, the two lines disagree, or the factor differs from ``expect``.
    """
    text = log.read_text() if isinstance(log, Path) else log
    found = {}
    for source, pattern in _WFACTOR_LINES.items():
        values = {float(m) for m in pattern.findall(text)}
        if len(values) != 1:
            raise ValueError(f"expected one wfactor value from {source} in the caltra log, found {sorted(values)}")
        found[source] = values.pop()
    if found["caltra.sh"] != found["caltra"]:
        raise ValueError(f"caltra.sh and caltra report different wfactors: {found}")
    if expect is not None and found["caltra"] != expect:
        raise ValueError(f"caltra used wfactor = {found['caltra']}, expected {expect} (OMEGA in Pa/s needs 1)")
    return found["caltra"]


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
