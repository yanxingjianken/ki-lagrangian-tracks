"""Omega, continuity check, eddy kinetic energy and 1-degree regridding of one life cycle; writes data/dino_<flow>.nc.

omega(sigma) = sigma V.grad(ps) - int_0^sigma div(ps V) dsigma' on the 15 model levels, with div(ps V) and grad(ps)
taken spectrally on the sphere (Dinosaur transforms of the northern-hemisphere fields mirrored about the equator: u, T,
ps even, v odd) and the integral over the model's layers. Check: -int_0^1 div(ps V) dsigma against centred hourly
differences of ps. Regridding: Fourier interpolation in longitude, cubic spline in latitude (three mirrored rows below
the equator); the 90N row is the zonal mean (scalars) or the wavenumber-1 part (u, v) of the last Gaussian row (89.05N).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from dinosaur import spherical_harmonic
from scipy.interpolate import CubicSpline

from common import A_SI, COORDS, G_SI, GRID, SIGMA

ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEYS = ("name", "case", "z2", "ujet", "m", "amp", "split", "label", "basic", "mode")
EDGES = np.asarray(COORDS.vertical.boundaries)
DSIG = np.diff(EDGES)
LAT_OUT, LON_OUT = np.arange(0.0, 91.0), np.arange(0.0, 360.0)
PARITY = dict(u=1.0, v=-1.0, t=1.0, omega=1.0, ps=1.0)
UNITS = dict(u="m s-1", v="m s-1", omega="Pa s-1", t="K", ps="Pa")
NAMES = dict(u="eastward wind", v="northward wind", omega="pressure vertical velocity Dp/Dt", t="temperature",
             ps="surface pressure")


def integral_weights(edges: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """W[k, j]: weight of layer j in the integral from sigma = 0 to the centre of layer k."""
    w = np.tril(np.broadcast_to(np.diff(edges), (len(centres),) * 2), -1).copy()
    w[np.diag_indices(len(centres))] = centres - edges[:-1]
    return w


def to_global(nh: np.ndarray, parity: float) -> np.ndarray:
    """(..., rows lat >= 0, lon) -> Dinosaur nodal order (..., lon, all Gaussian latitudes ascending)."""
    return np.swapaxes(np.concatenate([parity * nh[..., :0:-1, :], nh], axis=-2), -1, -2)


@jax.jit
def mass_flux_terms(u, v, ps):
    """div(ps V) and V.grad(ps) in Pa/s on (level, lon, lat) from nodal u, v (m/s) and ps (Pa)."""
    _, div = spherical_harmonic.uv_nodal_to_vor_div_modal(GRID, ps * u, ps * v)
    gx, gy = GRID.cos_lat_grad(GRID.to_modal(jnp.log(ps)))
    adv = ps * (u * GRID.to_nodal(gx) + v * GRID.to_nodal(gy)) / (A_SI * GRID.cos_lat)
    return GRID.to_nodal(div) / A_SI, adv


def omega_sigma(u, v, ps):
    """omega (level, lat, lon) and int_0^1 div(ps V) dsigma (lat, lon) from northern-hemisphere u, v, ps."""
    div, adv = (np.asarray(x) for x in mass_flux_terms(to_global(u, 1.0), to_global(v, -1.0), to_global(ps, 1.0)))
    om = SIGMA[:, None, None] * adv - np.tensordot(integral_weights(EDGES, SIGMA), div, axes=(1, 0))
    nh = slice(GRID.nodal_shape[1] // 2, None)
    return np.swapaxes(om[..., nh], -1, -2), np.swapaxes(np.tensordot(DSIG, div, axes=(0, 0))[..., nh], -1, -2)


def lat_weights(lat_in: np.ndarray, lat_out: np.ndarray) -> np.ndarray:
    """Matrix of the not-a-knot cubic spline through rows at lat_in, evaluated at lat_out."""
    return CubicSpline(lat_in, np.eye(len(lat_in)), axis=0)(lat_out)


def regrid(x: np.ndarray, lat_g: np.ndarray, parity: float, vector: bool, n_mirror: int = 3) -> np.ndarray:
    """(..., Gaussian rows lat >= 0, lon) -> (..., 91 rows 0..90N, 360 columns 0..359E)."""
    xl = np.fft.irfft(np.fft.rfft(x, axis=-1), n=LON_OUT.size, axis=-1) * (LON_OUT.size / x.shape[-1])
    ext = np.concatenate([parity * xl[..., n_mirror:0:-1, :], xl], axis=-2)
    body = np.matmul(lat_weights(np.r_[-lat_g[n_mirror:0:-1], lat_g], LAT_OUT[:-1]), ext)
    last = xl[..., -1, :]
    if vector:
        c = np.fft.rfft(last, axis=-1)
        c[..., 0], c[..., 2:] = 0.0, 0.0
        pole = np.fft.irfft(c, n=LON_OUT.size, axis=-1)
    else:
        pole = np.broadcast_to(last.mean(-1, keepdims=True), last.shape)
    return np.concatenate([body, pole[..., None, :]], axis=-2)


def eddy_kinetic_energy(u, v, ps, lat) -> float:
    """(1/g) int (u'^2 + v'^2)/2 dp averaged over 0-90N (J m-2); eddies = departures from the zonal mean."""
    ke = 0.5 * ((u - u.mean(-1, keepdims=True)) ** 2 + (v - v.mean(-1, keepdims=True)) ** 2)
    col = (ke * DSIG[:, None, None]).sum(0) * ps / G_SI
    w = np.cos(np.deg2rad(lat)) * (lat > 0)
    return float((col.mean(-1) * w).sum() / w.sum())


def continuity(tend: np.ndarray, dpdt: np.ndarray) -> dict:
    tend, dpdt = tend.ravel(), dpdt.ravel()
    return dict(corr=float(np.corrcoef(tend, dpdt)[0, 1]),
                rel_rms=float(np.sqrt(np.mean((tend - dpdt) ** 2) / np.mean(dpdt**2))))


def write(out: Path, fields: dict, hours: np.ndarray, attrs: dict) -> None:
    """netCDF4 file, float32, one time per chunk; time in hours plus a datetime coordinate from 2000-01-01T00."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".nc.part")
    with netCDF4.Dataset(tmp, "w") as nc:
        for d, n in (("time", len(hours)), ("level", len(SIGMA)), ("lat", LAT_OUT.size), ("lon", LON_OUT.size)):
            nc.createDimension(d, n)
        for name, dims, val, at in (
                ("time", ("time",), hours.astype("i4"), dict(units="hours", long_name="hours from the start")),
                ("datetime", ("time",), hours.astype("i4"), dict(units="hours since 2000-01-01 00:00:00",
                                                                  calendar="proleptic_gregorian")),
                ("level", ("level",), SIGMA, dict(units="1", long_name="sigma = p/ps at layer centres, top first",
                                                  positive="down", edges=EDGES)),
                ("lat", ("lat",), LAT_OUT, dict(units="degrees_north")),
                ("lon", ("lon",), LON_OUT, dict(units="degrees_east"))):
            var = nc.createVariable(name, val.dtype, dims)
            var[:] = val
            var.setncatts(at)
        for k, arr in fields.items():
            dims = ("time", "lat", "lon") if k == "ps" else ("time", "level", "lat", "lon")
            var = nc.createVariable(k, "f4", dims, chunksizes=(1,) + arr.shape[1:])
            var.setncatts(dict(units=UNITS[k], long_name=NAMES[k]))
            var[:] = arr
        nc.setncatts({k: (json.dumps(v) if isinstance(v, (dict, list, bool)) or v is None else v)
                      for k, v in attrs.items()})
    tmp.rename(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--flow", required=True)
    ap.add_argument("--flows", type=Path, default=ROOT / "configs" / "flows.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", type=Path, required=True)
    a = ap.parse_args()

    cfg = next(f for f in json.loads(a.flows.read_text()) if f["name"] == a.flow)
    ds = xr.open_dataset(a.run)
    lat_g = ds.lat.values
    hours = np.round(ds.time.values * 24.0).astype(int)
    raw = {k: ds[k].values for k in ("u", "v", "t", "ps")}
    nt = len(hours)
    omega = np.empty_like(raw["u"])
    col = np.empty_like(raw["ps"])
    for i in range(nt):
        om, c = omega_sigma(*(raw[k][i].astype(float) for k in ("u", "v", "ps")))
        omega[i], col[i] = om, c
    eke = np.array([eddy_kinetic_energy(raw["u"][i], raw["v"][i], raw["ps"][i], lat_g) for i in range(nt)])

    dt = np.diff(hours * 3600.0)
    dpdt = (raw["ps"][2:] - raw["ps"][:-2]) / (dt[1:] + dt[:-1])[:, None, None]
    tend = -col[1:-1]
    jet = (lat_g >= 30) & (lat_g <= 60)
    late = hours[1:-1] >= 48
    check = dict(
        status="ok", flow=a.flow, dt_s=float(ds.attrs.get("dt_s", 600.0)),
        continuity_30_60N=continuity(tend[:, jet], dpdt[:, jet]),
        continuity_30_60N_day2_on=continuity(tend[late][:, jet], dpdt[late][:, jet]),
        continuity_nh=continuity(tend, dpdt),
        equator_v_over_max_v=float(np.abs(raw["v"][:, :, 0]).max() / np.abs(raw["v"]).max()),
        eke_peak_day=float(hours[np.argmax(eke)] / 24.0), eke_peak_j_m2=float(eke.max()),
        omega_rms_sigma0887_day7=float(np.sqrt(np.mean(omega[np.argmin(np.abs(hours - 168)), 13, jet] ** 2))),
        eke=[round(float(e), 3) for e in eke], hours=hours.tolist())

    fields = {}
    for k, arr in (("u", raw["u"]), ("v", raw["v"]), ("omega", omega), ("t", raw["t"]), ("ps", raw["ps"])):
        out = np.empty(arr.shape[:-2] + (LAT_OUT.size, LON_OUT.size), np.float32)
        for s in range(0, nt, 24):
            out[s:s + 24] = regrid(arr[s:s + 24].astype(float), lat_g, PARITY[k], k in ("u", "v"))
        fields[k] = out
    finite = all(np.isfinite(f).all() for f in fields.values())
    check["status"] = "ok" if finite else "non-finite regridded fields"
    mode = pd.read_csv(ROOT / "data" / f"mode_{cfg['mode']}.csv")
    attrs = dict({k: cfg[k] for k in CONFIG_KEYS}, mode_growth_per_day=float(mode.growth_per_day.iloc[-1]),
                 mode_phase_speed_deg_per_day=float(mode.phase_speed_deg_per_day.iloc[-1]), amp_hpa=cfg["amp"],
                 dt_s=check["dt_s"],
                 model="Dinosaur primitive equations, THM93 set-up, T95, 15 sigma levels, hourly output",
                 omega_method="sigma-coordinate continuity, spectral div(ps V) and grad(ps)",
                 grid="1-degree, lat 0..90N, lon 0..359E; 90N row from the 89.05N Gaussian row",
                 continuity_30_60N=check["continuity_30_60N"])
    write(a.out, fields, hours, attrs)
    a.check.parent.mkdir(parents=True, exist_ok=True)
    a.check.write_text(json.dumps(check, indent=1) + "\n")
    c = check["continuity_30_60N"]
    print(f"{a.flow}: continuity 30-60N corr {c['corr']:.5f} rel rms {c['rel_rms']:.4f}; "
          f"EKE peak {check['eke_peak_j_m2']:.3e} J/m2 on day {check['eke_peak_day']:.2f}; status {check['status']}")


if __name__ == "__main__":
    main()
