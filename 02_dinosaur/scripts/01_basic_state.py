"""Balanced zonal basic states of THM93 on the model's sigma levels and Gaussian latitudes; writes data/basic_<name>.nc.
Adapted from literature_review/rwb/thorncroft93_baroclinic/thm93_dinosaur/01_basic_state.py.

Z1 (LC1): u = U_jet M(mu) F(sigma) with uniform surface pressure; T from the thermal-wind relation
dT/dphi = (a/R) (f + 2 u tan(phi)/a) du/dln(sigma), solved on a fine sigma grid, with the area mean of T at each level
equal to the January-climatology profile (SH77). Z2 (LC2): theta(p, lat) identical to Z1 (THM93 section 4); the
surface pressure carries the barotropic flow u_b (+10 m/s at 20N, -10 m/s at 50N) and the wind on each pressure surface
is the gradient wind of the Z1 forcing plus that of u_b.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from common import A_SI, LAT, OMEGA_SI, P0, R_SI, SIGMA, barotropic_z2, jet_meridional, mean_temperature, u0_profile

PHI_D = np.linspace(-90.0, 90.0, 7201)
PHI = np.deg2rad(PHI_D)
F_COR = 2 * OMEGA_SI * np.sin(PHI)
TAN = np.tan(np.clip(PHI, -np.deg2rad(89.99), np.deg2rad(89.99)))
W = np.cos(PHI)
LNSIG = np.linspace(np.log(0.005), np.log(1.06), 800)


def cumtrap(y: np.ndarray) -> np.ndarray:
    d = np.diff(PHI)
    return np.concatenate([np.zeros(y.shape[:-1] + (1,)), np.cumsum((y[..., 1:] + y[..., :-1]) / 2 * d, -1)], -1)


def area_mean(y: np.ndarray) -> np.ndarray:
    return (y * W).sum(-1) / W.sum()


def gradient_wind(forcing: np.ndarray) -> np.ndarray:
    """Root of (tan/a) u^2 + f u - Q = 0 that tends to Q/f."""
    disc = np.sqrt(np.maximum(F_COR**2 + 4 * TAN / A_SI * forcing, 0.0))
    den = F_COR + np.sign(F_COR) * disc
    return np.where(np.abs(den) > 1e-12, 2 * forcing / np.where(np.abs(den) > 1e-12, den, 1.0), 0.0)


def on_levels(field: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Interpolates a (fine ln sigma, lat) field to sigma values given per (level, lat)."""
    return np.stack([[np.interp(np.log(s), LNSIG, field[:, j]) for j, s in enumerate(row)] for row in sig])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=("lc1", "lc2"), required=True)
    ap.add_argument("--z2", choices=("lobes", "polarcap"), default="lobes")
    ap.add_argument("--u0", choices=("sh77", "analytic"), default="sh77")
    ap.add_argument("--tbar", choices=("era5jan", "stdatm"), default="era5jan")
    ap.add_argument("--ujet", type=float, default=47.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    sig = np.exp(LNSIG)
    prof = u0_profile(np.minimum(sig, 1.0), a.u0)
    slope = (u0_profile(1.0, a.u0) - u0_profile(0.99, a.u0)) / -np.log(0.99)
    prof = np.where(sig > 1.0, u0_profile(1.0, a.u0) + slope * LNSIG, prof)
    u1 = a.ujet * jet_meridional(PHI_D)[None] * prof[:, None]
    du = np.gradient(u1, LNSIG, axis=0)
    t1 = cumtrap(A_SI / R_SI * (F_COR + 2 * u1 * TAN / A_SI) * du)
    t1 += (mean_temperature(sig * P0, a.tbar) - area_mean(t1))[:, None]
    f1 = (F_COR + u1 * TAN / A_SI) * u1

    lnps = np.full_like(PHI, np.log(P0))
    sig_lev = np.broadcast_to(SIGMA[:, None], (len(SIGMA), len(PHI)))
    if a.case == "lc2":
        ub = barotropic_z2(PHI_D, a.z2)
        t_sfc = np.array([np.interp(0.0, LNSIG, t1[:, j]) for j in range(len(PHI))])
        lnps = cumtrap(-A_SI * (F_COR + ub * TAN / A_SI) * ub) / (R_SI * t_sfc)
        lnps = lnps - np.log(area_mean(np.exp(lnps))) + np.log(P0)
        sig_lev = SIGMA[:, None] * np.exp(lnps)[None] / P0
        u = gradient_wind(on_levels(f1, sig_lev) + (F_COR + ub * TAN / A_SI) * ub)
    else:
        u = on_levels(u1, sig_lev)
    t = on_levels(t1, sig_lev)

    interp = lambda y: np.stack([np.interp(LAT, PHI_D, row) for row in np.atleast_2d(y)])
    t_ref = mean_temperature(SIGMA * P0, a.tbar)
    ds = xr.Dataset(
        dict(u=(("level", "lat"), interp(u), {"units": "m s-1"}), t=(("level", "lat"), interp(t), {"units": "K"}),
             lnps=(("lat",), interp(lnps)[0], {"units": "ln(Pa)"}), t_ref=(("level",), t_ref, {"units": "K"})),
        coords=dict(level=SIGMA, lat=LAT),
        attrs=dict(case=a.case, z2=a.z2 if a.case == "lc2" else "none", u0=a.u0, tbar=a.tbar, ujet=a.ujet))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(a.out)


if __name__ == "__main__":
    main()
