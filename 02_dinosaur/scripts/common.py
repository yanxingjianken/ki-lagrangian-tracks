"""Shared set-up of the THM93 life cycles with Dinosaur: T95 grid, the 15 THM93 sigma levels, del^6 diffusion with a 1 h
decay at n = 95, the zonal-flow profiles of Z1/Z2 and conversions between the model state and SI fields.
Adapted from literature_review/rwb/thorncroft93_baroclinic/thm93_dinosaur/common.py (wavenumber made an argument,
unclipped winds)."""
from __future__ import annotations

import dataclasses

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from dinosaur import coordinate_systems, filtering, primitive_equations as pe, scales  # noqa: E402
from dinosaur import sigma_coordinates, spherical_harmonic, time_integration  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

U = scales.units
SPECS = pe.PrimitiveEquationsSpecs.from_si()
A_SI, OMEGA_SI, G_SI, R_SI, KAPPA = 6.37122e6, 7.292e-5, 9.80616, 287.0, 2.0 / 7.0
P0 = 1.0e5
SIGMA_FULL = np.array([0.018, 0.060, 0.106, 0.152, 0.197, 0.241, 0.287, 0.338, 0.400, 0.477, 0.569, 0.674, 0.784,
                       0.887, 0.967])


def sigma_boundaries(full: np.ndarray) -> np.ndarray:
    """Layer edges whose midpoints are the THM93 full levels (the recursion closes at 0.998, set to 1)."""
    b = [0.0]
    for s in full:
        b.append(2 * s - b[-1])
    b[-1] = 1.0
    return np.array(b)


GRID = spherical_harmonic.Grid.with_wavenumbers(96)
COORDS = coordinate_systems.CoordinateSystem(GRID, sigma_coordinates.SigmaCoordinates(sigma_boundaries(SIGMA_FULL)))
SIGMA = np.asarray(COORDS.vertical.centers)
LON = np.rad2deg(GRID.nodal_axes[0])
LAT = np.rad2deg(np.arcsin(GRID.nodal_axes[1]))
M_AXIS, L_AXIS = (np.asarray(x) for x in GRID.modal_axes)

# U0(sigma)/U0max on the 45-degree jet axis, read off the isotachs of Simmons & Hoskins (1977) Fig. 1c (5 m/s contours,
# peak 39.8 m/s near 200 mb, 5 m/s above 30 mb); monotone cubic in ln sigma.
U0_SIG = np.array([0.010, 0.030, 0.056, 0.070, 0.084, 0.098, 0.116, 0.139, 0.202, 0.286, 0.347, 0.415, 0.494, 0.591,
                   0.705, 0.841, 1.0])
U0_VAL = np.array([5.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 39.8, 35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0,
                   0.0]) / 39.8
_U0 = PchipInterpolator(np.log(U0_SIG), U0_VAL)
# ERA5 1990-2020 January, northern-hemisphere area-mean temperature (SH77: horizontal mean from January climatology).
TB_P = np.array([100.0, 200.0, 250.0, 300.0, 400.0, 500.0, 700.0, 850.0, 1000.0]) * 100.0
TB_T = np.array([203.77, 217.92, 224.02, 230.94, 244.42, 255.25, 270.40, 277.28, 283.61])


def u0_profile(sigma, kind: str = "sh77"):
    """Vertical profile F(sigma) of the jet (max 1); 'analytic' is s exp((1 - s^2)/2), s = ln sigma / ln 0.2."""
    sigma = np.asarray(sigma, float)
    if kind == "analytic":
        s = np.log(sigma) / np.log(0.2)
        return s * np.exp((1 - s**2) / 2)
    return _U0(np.log(np.clip(sigma, U0_SIG[0], 1.0)))


def mean_temperature(p, kind: str = "era5jan"):
    """Horizontal-mean temperature (K) at pressure p (Pa)."""
    p = np.asarray(p, float)
    if kind == "stdatm":
        return np.maximum(288.0 * (p / P0) ** (R_SI * 6.5e-3 / G_SI), 216.65)
    return np.interp(np.log(np.clip(p, TB_P[0], TB_P[-1])), np.log(TB_P), TB_T)


def jet_meridional(lat_deg):
    """M(mu) = sin^3(pi mu^2), mu = sin(latitude): the 45-degree jet of SH77 (their profile c)."""
    return np.sin(np.pi * np.sin(np.deg2rad(lat_deg)) ** 2) ** 3


def barotropic_z2(lat_deg, kind: str):
    """THM93 Z2 barotropic flow: +10 m/s at 20N, -10 m/s at 50N; the shape away from those latitudes is not given."""
    x = np.abs(np.asarray(lat_deg, float))
    lobe = lambda c: np.where(np.abs(x - c) < 12.5, np.cos(np.pi * (x - c) / 25.0) ** 2, 0.0)
    if kind == "lobes":
        return 10.0 * (lobe(20.0) - lobe(50.0))
    cap = -10.0 * np.tan(np.deg2rad(45 - np.minimum(x, 90) / 2)) / np.tan(np.deg2rad(20.0))
    mid = 10.0 * np.cos(np.pi * (x - 20.0) / 30.0)
    low = np.where(x > 7.5, 10.0 * np.cos(np.pi * (20.0 - x) / 25.0) ** 2, 0.0)
    return np.where(x >= 50, cap, np.where(x >= 20, mid, low))


def nd(value, unit):
    return SPECS.nondimensionalize(value * unit)


def state_from_basic(ds: xr.Dataset) -> tuple[pe.State, np.ndarray]:
    """Model state of a zonal basic state (u, T on sigma x lat; ln ps on lat, SI) and its reference temperatures."""
    u = np.broadcast_to(ds.u.values[:, None, :], (len(SIGMA), len(LON), len(LAT)))
    vor, _ = spherical_harmonic.uv_nodal_to_vor_div_modal(GRID, nd(u, U.m / U.s), np.zeros_like(u))
    t_ref = ds.t_ref.values
    temp = np.broadcast_to((ds.t.values - t_ref[:, None])[:, None, :], u.shape)
    lnps = np.broadcast_to(ds.lnps.values[None, None, :], (1, len(LON), len(LAT))) + np.log(nd(1.0, U.Pa))
    state = pe.State(vorticity=jnp.asarray(vor), divergence=jnp.zeros_like(vor),
                     temperature_variation=GRID.to_modal(jnp.asarray(temp)),
                     log_surface_pressure=GRID.to_modal(jnp.asarray(lnps)))
    return state, t_ref


def make_step(t_ref: np.ndarray, dt_s: float = 600.0, tau_s: float = 3600.0):
    """One IMEX-RK3 step of the dry, frictionless primitive equations followed by THM93's del^6 diffusion on
    vorticity, divergence and temperature (decay rate 1/tau at n = 95)."""
    eq = pe.PrimitiveEquations(t_ref, np.zeros(GRID.modal_shape), COORDS, SPECS)
    dt = nd(dt_s, U.s)
    scale = dt / (nd(tau_s, U.s) * (95.0 * 96.0) ** 3)
    diffuse = filtering.horizontal_diffusion_filter(GRID, scale, order=3)

    def thm93_diffusion(_, x):
        return dataclasses.replace(x, vorticity=diffuse(x.vorticity), divergence=diffuse(x.divergence),
                                   temperature_variation=diffuse(x.temperature_variation))

    return time_integration.step_with_filters(time_integration.imex_rk_sil3(eq, dt), [thm93_diffusion])


def propagator(step, n_steps: int):
    return jax.jit(lambda x: jax.lax.fori_loop(0, n_steps, lambda _, s: step(s), x))


def project_wave(x: pe.State, m: int) -> pe.State:
    """Zonal wavenumber m, symmetric about the equator: ps, T, divergence even in (n - m); vorticity odd."""
    keep = np.abs(M_AXIS)[:, None] == m
    even = ((L_AXIS[None, :] - m) % 2) == 0
    sym, anti = jnp.asarray(keep & even), jnp.asarray(keep & ~even)
    return dataclasses.replace(x, vorticity=x.vorticity * anti, divergence=x.divergence * sym,
                               temperature_variation=x.temperature_variation * sym,
                               log_surface_pressure=x.log_surface_pressure * sym)


def nodal_fields(x: pe.State, t_ref: np.ndarray) -> dict[str, np.ndarray]:
    """u, v (m/s), vorticity (1/s), T (K) on (level, lon, lat) and ps (Pa) on (lon, lat); u, v unclipped, i.e. the wind
    the model itself advects with (clipping the top wavenumber changes the divergence by ~2 %)."""
    u, v = spherical_harmonic.vor_div_to_uv_nodal(GRID, x.vorticity, x.divergence, clip=False)
    to_ms = 1.0 / nd(1.0, U.m / U.s)
    vor = np.asarray(GRID.to_nodal(x.vorticity)) / nd(1.0, 1 / U.s)
    temp = np.asarray(GRID.to_nodal(x.temperature_variation)) + t_ref[:, None, None]
    ps = np.exp(np.asarray(GRID.to_nodal(x.log_surface_pressure))[0] - np.log(nd(1.0, U.Pa)))
    return dict(u=np.asarray(u) * to_ms, v=np.asarray(v) * to_ms, vor=vor, t=temp, ps=ps)
