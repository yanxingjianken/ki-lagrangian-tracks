"""Most unstable zonal-wavenumber-m normal mode of a basic state, symmetric about the equator, by power iteration of the
model's tangent-linear one-day propagator (forward-mode derivative of the full nonlinear model at the basic state).
Writes data/mode_<name>.npz (the mode scaled to 1 hPa maximum surface-pressure amplitude) and data/mode_<name>.csv
(growth rate per day and eastward phase speed in degrees longitude per day, per iteration; `converged` marks whether
the iteration met the stopping rule). Adapted from literature_review/rwb/thorncroft93_baroclinic/thm93_dinosaur.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import xarray as xr

from common import GRID, LAT, LON, P0, make_step, project_wave, propagator, state_from_basic

STEPS_PER_DAY = 144


def scaled(x, c):
    return jax.tree_util.tree_map(lambda a: a * c, x)


def ps_wave(dx, m: int) -> tuple[float, float]:
    """Maximum surface-pressure amplitude (hPa) and longitude of the wave crest at the latitude of that maximum."""
    lnps = np.asarray(GRID.to_nodal(dx.log_surface_pressure))[0]
    j = int(np.argmax(np.abs(lnps).max(0) * (LAT > 0)))
    coef = np.fft.rfft(lnps[:, j])[m]
    return float(np.abs(lnps).max() * P0 / 100), float(-np.rad2deg(np.angle(coef)) / m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basic", type=Path, required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    base, t_ref = state_from_basic(xr.open_dataset(a.basic))
    one_day = propagator(make_step(t_ref), STEPS_PER_DAY)
    tangent = jax.jit(lambda dx: jax.jvp(one_day, (base,), (dx,))[1])

    lon, lat = np.meshgrid(np.deg2rad(LON), np.deg2rad(LAT), indexing="ij")
    seed = 1e-5 * np.cos(a.m * lon) * np.exp(-((np.abs(np.rad2deg(lat)) - 45.0) / 15.0) ** 2)
    zero = scaled(base, 0.0)
    dx = project_wave(dataclasses.replace(zero, log_surface_pressure=GRID.to_modal(seed[None])), a.m)
    amp, crest = ps_wave(dx, a.m)
    dx = scaled(dx, 1.0 / amp)
    rows, converged = [], False
    for it in range(1, a.iterations + 1):
        dx = project_wave(tangent(dx), a.m)
        amp_new, crest_new = ps_wave(dx, a.m)
        shift = (crest_new - crest + 180.0 / a.m) % (360.0 / a.m) - 180.0 / a.m
        rows.append(dict(iteration=it, growth_per_day=np.log(amp_new), phase_speed_deg_per_day=shift))
        print(f"{a.basic.stem} iteration {it:2d}: growth {np.log(amp_new):.4f} /day, phase speed {shift:+.3f} deg/day",
              flush=True)
        dx, crest = scaled(dx, 1.0 / amp_new), crest_new
        if it >= 8 and np.ptp([r["growth_per_day"] for r in rows[-4:]]) < 2e-4 and \
                np.ptp([r["phase_speed_deg_per_day"] for r in rows[-4:]]) < 2e-3:
            converged = True
            break

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out.with_suffix(".npz"), **{k: np.asarray(getattr(dx, k)) for k in
                                           ("vorticity", "divergence", "temperature_variation", "log_surface_pressure")})
    pd.DataFrame(rows).assign(m=a.m, converged=converged).to_csv(a.out.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
