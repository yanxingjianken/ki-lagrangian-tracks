"""Stirred-jet BVE runs: spin-up, then hourly wind and vorticity on a regular 1-degree grid.

One flow (--u0 --lat0 --width --seed, --out) or a set integrated as one batch (--flows configs/flows.json, one
data/bve_<name>.nc each). Each file: u, v, zeta on time x lat x lon; ke_spinup daily kinetic energy.
"""
import json
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stirred_jet import RegularGrid, StirredJet  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--u0", type=float, default=30.0)
p.add_argument("--lat0", type=float, default=45.0)
p.add_argument("--width", type=float, default=10.0)
p.add_argument("--amp", type=float, default=4e-10)
p.add_argument("--spinup-days", type=float, default=60.0)
p.add_argument("--record-days", type=float, default=30.0)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--start", default="2000-01-01T00")
p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data/bve_stirred_jet.nc"))
p.add_argument("--flows", default=None, help="JSON flow set; overrides the single-flow options")
a = p.parse_args()

if a.flows:
    flows = json.loads(Path(a.flows).read_text())["flows"]
    outs = [Path(a.out).parent / f"bve_{f['name']}.nc" for f in flows]
    sj = StirredJet([a.amp] * len(flows), u0=[f["u0"] for f in flows], lat0=[f["lat0"] for f in flows],
                    width=[f["width"] for f in flows], stir_m=[f["m"] for f in flows],
                    stir_lat=[f["lat0"] for f in flows], seed=a.seed)
else:
    flows = [dict(u0=a.u0, lat0=a.lat0, width=a.width, m=None)]
    outs = [Path(a.out)]
    sj = StirredJet([a.amp], u0=a.u0, lat0=a.lat0, width=a.width, seed=a.seed)
grid = RegularGrid(sj.m)
spd = int(round(86400 / sj.m.dt))
area = np.cos(np.deg2rad(grid.lat))[:, None]


def kinetic_energy(uv):
    return float((0.5 * (uv**2).sum(0) * area).sum() / (area.sum() * uv.shape[-1]))


ke = []
for day in range(int(a.spinup_days)):
    sj.run(spd)
    ke.append([kinetic_energy(uv) for uv in grid.uv(sj.zeta).cpu().numpy()])
ke = np.asarray(ke)

n_rec = int(a.record_days * 24) + 1
B = len(flows)
U = np.empty((B, n_rec, len(grid.lat), len(grid.lon)), np.float32)
V, Z = np.empty_like(U), np.empty_like(U)
per_h = spd // 24
for n in range(n_rec):
    if n:
        sj.run(per_h)
    uv = grid.uv(sj.zeta).cpu().numpy()
    U[:, n], V[:, n] = uv[:, 0], uv[:, 1]
    Z[:, n] = grid.zeta(sj.zeta).cpu().numpy()

time = np.datetime64(a.start) + np.arange(n_rec).astype("timedelta64[h]")
enc = {k: dict(chunksizes=(1, len(grid.lat), len(grid.lon))) for k in ("u", "v", "zeta")}
for b, (f, out) in enumerate(zip(flows, outs)):
    stir = "l 8-14" + (f", zonal wavenumber {f['m']}" if f.get("m") else "")
    ds = xr.Dataset(
        dict(u=(("time", "lat", "lon"), U[b], dict(units="m s-1", long_name="eastward wind")),
             v=(("time", "lat", "lon"), V[b], dict(units="m s-1", long_name="northward wind")),
             zeta=(("time", "lat", "lon"), Z[b], dict(units="s-1", long_name="relative vorticity")),
             ke_spinup=(("spinup_day",), ke[:, b], dict(units="m2 s-2", long_name="area-mean kinetic energy"))),
        coords=dict(time=time, lat=("lat", grid.lat, dict(units="degrees_north")),
                    lon=("lon", grid.lon, dict(units="degrees_east")), spinup_day=np.arange(1, len(ke) + 1)),
        attrs=dict(model="spherical barotropic vorticity equation, torch-harmonics, T85 Gaussian grid 128x256, dt 600 s",
                   forcing=f"relaxation to Gaussian jet u0={f['u0']} m/s at {f['lat0']}N, width {f['width']} deg, "
                           f"tau 10 d; AR(1) stirring amp={a.amp} s-2, tau 1 d, {stir}, Gaussian band "
                           f"{f['lat0'] if a.flows else 45}N +- 12 deg; del^8 hyperdiffusion 6 h",
                   spinup_days=a.spinup_days, seed=a.seed, flow=json.dumps(f)))
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out, encoding=enc)
    print(out, ds.sizes)
