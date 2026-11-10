"""Write ki_era5.ipynb: a self-contained walk from ERA5 winds to conditional forecasts and backcasts of parcel tracks.

Ocean-notebook layout (the template is ki/notebooks/oceans_currents_interpolant.ipynb): download three days of ERA5
from Google's public ARCO store, write LAGRANTO input files, release patches of air parcels and compute their 3-D
tracks, turn them into hourly positions, velocities and accelerations, load the trained conditional Kinematic
Interpolant (public Hugging Face model) and forecast and backcast parcels that the model has not seen, against their
true tracks and straight motion. The model code is copied from scripts/ctx_model.py.
"""
import ast
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def source(file, *names):
    src = (ROOT / "scripts" / file).read_text()
    nodes = [n for n in ast.parse(src).body if getattr(n, "name", None) in names]
    return "\n\n\n".join(ast.get_source_segment(src, n) for n in nodes)


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# ERA5: conditional forecasts and backcasts of air-parcel tracks with a Kinematic Interpolant\n\n"
       "A self-contained run from reanalysis winds to forecasts, in the layout of the ocean notebook of the Kinematic "
       "Interpolant paper (Jiang, Marzouk, Bose & Pooladian, *Learning smooth dynamics via Kinematic Interpolants*, "
       "arXiv:2610.22519; `ki/notebooks/oceans_currents_interpolant.ipynb`). Steps:\n\n"
       "1. download three days of hourly ERA5 winds (0.25°, 37 pressure levels) from Google's public ARCO store;\n"
       "2. write them as LAGRANTO input files;\n"
       "3. release patches of air parcels and compute their 3-D tracks with LAGRANTO (the *true tracks*);\n"
       "4. turn the tracks into hourly positions, velocities and accelerations;\n"
       "5. load the conditional Kinematic Interpolant trained on 2015–2017 (public Hugging Face model);\n"
       "6. **forecast**: from the tracks of 192 parcels of a patch over the 48 h before an anchor time, integrate other "
       "parcels of the patch 24 h forward; **backcast**: from the 48 h after the anchor, integrate them 24 h backward; "
       "deterministically and as a 16-member ensemble;\n"
       "7. compare with the true tracks and with straight motion.\n\n"
       "The model never sees a wind field. Case: the Pacific Northwest heat dome of late June 2021 (2021 was not used in "
       "training). Running it downloads about 34 GB (a few minutes at 150 MB/s) into `data/notebook/`, which is "
       "deleted at the end; LAGRANTO must be built (`bash ../lagranto/build.sh`)."),
    md("## 1. Imports and settings"),
    code("from pathlib import Path\nimport json, math, os, shutil, subprocess, sys, time\nimport numpy as np\nimport torch\nfrom torch import nn\n"
         "import matplotlib.pyplot as plt\nimport cartopy.crs as ccrs\nfrom scipy.interpolate import CubicSpline, RegularGridInterpolator\n"
         "ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'scripts/ctx_model.py').is_file() and (p / 'PLAN.md').is_file())\n"
         "LAGRANTO = ROOT.parent / 'lagranto'\nsys.path.insert(0, str(LAGRANTO / 'python'))\nimport lagranto_io as lio\n\n"
         "RELEASE = np.datetime64('2021-06-24T00', 'h')   # parcels released here; the window is 72 h\n"
         "WINDOW_H = 72\n"
         "CENTRES = [(52.0, 232.0), (56.0, 240.0), (48.0, 226.0), (60.0, 248.0)]   # patch centres (deg N, deg E) around the ridge\n"
         "SIGMAS = [0.477, 0.285]                      # start levels: pressure / surface pressure\n"
         "N_PER, RADIUS_KM = 480, 300.0                # parcels per patch, disk radius (as in training)\n"
         "WORK = ROOT / 'data/notebook'                # downloaded winds and LAGRANTO files (deleted at the end)\n"
         "MODEL_REPO = 'yanxingjianken/ki-era5-tracks'\n"
         "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
         "H_KM, A_KM = 7.0, 6371.0\nWORK.mkdir(parents=True, exist_ok=True)\nprint('device', DEVICE)"),
    md("## 2. ERA5 winds as LAGRANTO input files\n"
       "One file per hour with u, v, the pressure vertical velocity omega (Pa/s) and the surface pressure on the 37 "
       "pressure levels. On pure pressure levels LAGRANTO's reader ignores the surface pressure and puts the ground at "
       "1000 hPa everywhere; `write_pfile_plev` therefore gives the bottom level a tiny surface-pressure weight "
       "(bk = 0.002), which makes LAGRANTO keep parcels above the terrain. Each variable-hour of the store is one "
       "global chunk of 154 MB, so the whole globe is downloaded."),
    code("import gcsfs, xarray as xr\n"
         "store = xr.open_zarr(gcsfs.GCSFileSystem(token='anon').get_mapper('gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'), chunks=None)\n"
         "VARS = {'U': 'u_component_of_wind', 'V': 'v_component_of_wind', 'OMEGA': 'vertical_velocity', 'PS': 'surface_pressure'}\n"
         "HOURS = RELEASE + np.arange(WINDOW_H + 1) * np.timedelta64(1, 'h')\n"
         "t0 = time.time()\n"
         "for t in HOURS:\n"
         "    if (WORK / lio._pname(t)).exists():\n"
         "        continue\n"
         "    f = {k: store[v].sel(time=t).values for k, v in VARS.items()}\n"
         "    lio.write_pfile_plev(WORK, t, store.longitude.values, store.latitude.values, store.level.values, f['U'], f['V'], f['OMEGA'], f['PS'])\n"
         "z500 = store['geopotential'].sel(time=[RELEASE + np.timedelta64(h, 'h') for h in (24, 48, 72)], level=500).isel(latitude=slice(None, None, 4), longitude=slice(None, None, 4)) / 9.80665\n"
         "z500 = z500.load()\n"
         "print(f'{len(HOURS)} hourly files from {HOURS[0]} to {HOURS[-1]} in {time.time() - t0:.0f} s')"),
    md("## 3. True tracks with LAGRANTO\n"
       "Each patch is 480 parcels uniform in a disk of 300 km radius at one start level, the start pressure being sigma "
       "times the local surface pressure (as in training). caltra integrates the 3-D tracks with a 1-minute step and "
       "writes positions every 10 minutes; `-j` lifts parcels that touch the ground back into the air."),
    code("def destination(lat, lon, bearing, dist_km):\n"
         "    p1, l1, d = np.deg2rad(lat), np.deg2rad(lon), dist_km / A_KM\n"
         "    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(bearing))\n"
         "    l2 = l1 + np.arctan2(np.sin(bearing) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))\n"
         "    return np.rad2deg(p2), np.rad2deg(l2) % 360.0\n\n"
         "import netCDF4\n"
         "with netCDF4.Dataset(WORK / lio._pname(RELEASE)) as nc:\n"
         "    ps0 = np.asarray(nc['PS'][0, 0], float); box = (nc['PS'].xmin, nc['PS'].xmax, nc['PS'].ymin, nc['PS'].ymax)\n"
         "ps_at = RegularGridInterpolator((np.linspace(box[2], box[3], ps0.shape[0]), np.linspace(box[0], box[1], ps0.shape[1])), ps0)\n"
         "rng = np.random.default_rng(0)\n"
         "lat, lon, pres, lev, pid = [], [], [], [], []\n"
         "for k, sigma in enumerate(SIGMAS):\n"
         "    for q, (clat, clon) in enumerate(CENTRES):\n"
         "        la, lo = destination(clat, clon, rng.uniform(0, 2 * np.pi, N_PER), RADIUS_KM * np.sqrt(rng.uniform(0, 1, N_PER)))\n"
         "        lat.append(la); lon.append(lo); pres.append(sigma * ps_at(np.c_[la, lo])); lev.append(np.full(N_PER, k)); pid.append(np.full(N_PER, q))\n"
         "lat, lon, pres, lev, pid = map(np.concatenate, (lat, lon, pres, lev, pid))\n"
         "lio.startfile(WORK / 'startf', lon, lat, pres, lon_min=0.0)\n"
         "env = dict(os.environ, LAGRANTO=str(LAGRANTO), PATH=f\"{LAGRANTO / 'bin'}:{os.environ['PATH']}\")\n"
         "t0 = time.time()\n"
         "out = subprocess.run(['caltra', lio._pname(HOURS[0])[1:], lio._pname(HOURS[-1])[1:], 'startf', 'traj.3', '-i', '1', '-t', '1', '-o', '10', '-j'],\n"
         "                     cwd=WORK, env=env, capture_output=True, text=True)\n"
         "assert out.returncode == 0, out.stdout[-2000:]\n"
         "tr = lio.read_traj(WORK / 'traj.3')\n"
         "print(f\"{tr['lon'].shape[1]} tracks, {len(tr['time'])} times in {time.time() - t0:.0f} s; incomplete: {int((~np.isfinite(tr['lon'])).any(0).sum())}\")"),
    md("## 4. Hourly positions, velocities and accelerations\n"
       "Positions in north-polar stereographic x, y (km) and log-pressure height z = −7 km ln(p / 1000 hPa); a cubic "
       "spline through the 10-minute positions gives hourly velocity and acceleration (as in training)."),
    code("def lonlat_to_xy(lon, lat):\n"
         "    r = 2 * A_KM * np.tan(np.deg2rad(90.0 - lat) / 2); lam = np.deg2rad(lon)\n"
         "    return np.stack([r * np.sin(lam), -r * np.cos(lam)], -1)\n\n"
         "def xy_to_lonlat(x):\n"
         "    r = np.hypot(x[..., 0], x[..., 1])\n"
         "    return np.rad2deg(np.arctan2(x[..., 0], -x[..., 1])) % 360.0, 90.0 - np.rad2deg(2 * np.arctan(r / (2 * A_KM)))\n\n"
         "t = tr['time'] - tr['time'][0]\n"
         "lam = np.rad2deg(np.unwrap(np.deg2rad(tr['lon']), axis=0))\n"
         "xyz = np.concatenate([lonlat_to_xy(lam, tr['lat']), (-H_KM * np.log(tr['p'] / 1000.0))[..., None]], -1)\n"
         "spl = CubicSpline(t, xyz, axis=0)\nth = np.arange(WINDOW_H + 1.0)\n"
         "X, V, A = spl(th).astype(np.float32), spl(th, 1).astype(np.float32), spl(th, 2).astype(np.float32)   # (73 h, parcels, 3)\n"
         "P_HPA = 1000.0 * np.exp(-X[..., 2] / H_KM)\n"
         "print(X.shape, f'median horizontal speed {np.median(np.hypot(V[..., 0], V[..., 1])):.0f} km/h')"),
    code("fig = plt.figure(figsize=(12, 5), layout='constrained')\n"
         "ax = fig.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo(central_longitude=235))\n"
         "ax.set_extent([190, 280, 30, 80], ccrs.PlateCarree()); ax.coastlines(lw=0.4, color='0.5')\n"
         "ax.contour(z500.longitude, z500.latitude, z500.isel(time=1), levels=np.arange(5200, 6000, 60), colors='k', linewidths=0.5, transform=ccrs.PlateCarree())\n"
         "lo, la = xy_to_lonlat(X)\n"
         "for i in range(0, X.shape[1], 25):\n"
         "    ax.plot(np.rad2deg(np.unwrap(np.deg2rad(lo[:, i]))), la[:, i], lw=0.5, color='tab:red' if lev[i] == 1 else 'tab:blue', transform=ccrs.Geodetic())\n"
         "ax.set_title('true tracks (every 25th parcel; red: sigma 0.285, blue: 0.477); 500-hPa height at 48 h', fontsize=9)\n"
         "b = fig.add_subplot(1, 2, 2)\n"
         "for i in range(0, X.shape[1], 25):\n"
         "    b.plot(th, P_HPA[:, i], lw=0.5, color='tab:red' if lev[i] == 1 else 'tab:blue')\n"
         "b.invert_yaxis(); b.set(xlabel='hours after release', ylabel='pressure (hPa)', title='pressure along the tracks')\nplt.show()"),
    md("## 5. The conditional Kinematic Interpolant\n"
       "A parcel's state is its position (x, y, z) and velocity. Context tokens are the observed parcels of the patch "
       "every 3 h over 48 h: time from the anchor, position relative to the patch centroid at the anchor (absolute "
       "height), sin(latitude) and velocity, each axis divided by its scale. A Perceiver encoder compresses them into "
       "96 latent vectors; a query state reads the latents by cross-attention and returns the acceleration and the "
       "scaled velocity score of the paper (code copied from `scripts/ctx_model.py`). The weights and the per-axis "
       "scales are downloaded from the public Hugging Face model."),
    code("N_FEAT = 8   # token and query features: time, x, y, z, sin(latitude), u, v, w\n\n\n" + source("ctx_model.py", "Fourier", "Block", "embed", "ContextKI")),
    code("import warnings\nwarnings.filterwarnings('ignore', message='IProgress not found')   # progress bars without ipywidgets\n"
         "from huggingface_hub import hf_hub_download\n"
         "ck = torch.load(hf_hub_download(MODEL_REPO, 'context_ki_era5.pt'), map_location=DEVICE, weights_only=False)\n"
         "ens = json.loads(Path(hf_hub_download(MODEL_REPO, 'ensemble.json')).read_text())\n"
         "model = ContextKI(d=ck['args']['d_model']).to(DEVICE).eval(); model.load_state_dict(ck['state'])\n"
         "SC = ck['scales']; tt = lambda v: torch.tensor(v, dtype=torch.float32, device=DEVICE)\n"
         "POS, VEL, ACC, TIME_H = tt(SC['pos_km']), tt(SC['vel_kmh']), tt(SC['acc_kmh2']), float(SC['time_h'])\n"
         "SIGMA_V = tt([5.0, 5.0, 0.01]) * float(np.pi * np.sqrt(3) / 6.0)   # training noise in velocity (km/h)\n"
         "print(f\"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters; scales {SC['vel_kmh']} km/h, {SC['acc_kmh2']} km/h2; ensemble eps_rel {ens['eps_rel']}\")"),
    md("## 6. Forecasts and backcasts\n"
       "**Forecast** at anchor 48 h: the context is 192 parcels of the patch observed every 3 h over 0–48 h; 64 other "
       "parcels start from their true state at 48 h and are integrated to 72 h. **Backcast** at anchor 24 h: the "
       "context is observed over 24–72 h; the parcels are integrated backward to 0 h. Deterministic tracks follow "
       "x' = v, v' = a (Heun, 10-minute steps); ensemble members add the score term and random kicks, "
       "dv = [a + (eps²/2) s / sigma_v] dt + eps dW with eps = eps_rel sigma_v, so that the spread matches the error "
       "on validation data. Straight motion keeps the anchor velocity."),
    code("def sin_lat(x):\n    r = torch.hypot(x[..., 0], x[..., 1]); return torch.cos(2 * torch.atan(r / (2 * A_KM)))\n\n"
         "def tokens(Xp, Vp, t0, d, ctx):\n"
         "    rows = t0 - d * 3 * np.arange(17); Xc, Vc = tt(Xp[rows][:, ctx]), tt(Vp[rows][:, ctx])\n"
         "    cen = tt(Xp[t0, :, :2].mean(0))\n"
         "    tau = tt(-d * 3.0 * np.arange(17) / TIME_H)[:, None, None].expand(17, len(ctx), 1)\n"
         "    rel = torch.cat([Xc[..., :2] - cen, Xc[..., 2:]], -1)\n"
         "    return torch.cat([tau, rel / POS, sin_lat(Xc)[..., None], Vc / VEL], -1).reshape(1, -1, 8), cen\n\n"
         "def heads(z, cen, tau, x, v):\n"
         "    rel = torch.cat([x[..., :2] - cen, x[..., 2:]], -1)\n"
         "    a, s = model.decode(z, torch.cat([tau / TIME_H, rel / POS, sin_lat(x)[..., None], v / VEL], -1))\n"
         "    return a * ACC, s\n\n"
         "@torch.no_grad()\n"
         "def integrate(tok, cen, x0, v0, d, members=0, hours=24, sph=6, seed=0):\n"
         "    \"\"\"members = 0: deterministic (Heun); else Euler-Maruyama ensemble. Returns hourly positions (hours + 1, M, Q, 3).\"\"\"\n"
         "    z = model.encode(tok); g = torch.Generator(device=DEVICE).manual_seed(seed)\n"
         "    m = max(members, 1); x = tt(x0)[None].repeat(m, 1, 1).reshape(1, -1, 3); v = tt(v0)[None].repeat(m, 1, 1).reshape(1, -1, 3)\n"
         "    dt = d / sph; eps = ens['eps_rel'] * SIGMA_V; out = [x.clone()]\n"
         "    for k in range(hours * sph):\n"
         "        tau = torch.full_like(x[..., :1], k * dt); a, s = heads(z, cen, tau, x, v)\n"
         "        if members == 0:\n"
         "            xp, vp = x + dt * v, v + dt * a; a1, _ = heads(z, cen, tau + dt, xp, vp)\n"
         "            x, v = x + 0.5 * dt * (v + vp), v + 0.5 * dt * (a + a1)\n"
         "        else:\n"
         "            x, v = x + dt * v, v + dt * a + abs(dt) * 0.5 * eps ** 2 * s / SIGMA_V + eps * np.sqrt(abs(dt)) * torch.randn(x.shape, device=DEVICE, generator=g)\n"
         "        if (k + 1) % sph == 0: out.append(x.clone())\n"
         "    return torch.stack(out).reshape(hours + 1, m, -1, 3).cpu().numpy()\n\n"
         "def gc_km(xa, xb):\n"
         "    (l1, p1), (l2, p2) = xy_to_lonlat(xa), xy_to_lonlat(xb); p1, p2 = np.deg2rad(p1), np.deg2rad(p2)\n"
         "    return A_KM * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(np.deg2rad(l2 - l1)), -1, 1))\n\n"
         "rows, keep = [], {}\n"
         "for k in range(len(SIGMAS)):\n"
         "    for q in range(len(CENTRES)):\n"
         "        idx = np.flatnonzero((lev == k) & (pid == q) & np.isfinite(X).all((0, 2)))\n"
         "        perm = np.random.default_rng(10 * k + q).permutation(idx); ctx, tgt = perm[:192], perm[192:256]\n"
         "        for task, d, t0 in (('forecast', 1, 48), ('backcast', -1, 24)):\n"
         "            tok, cen = tokens(X, V, t0, d, ctx)\n"
         "            det = integrate(tok, cen, X[t0, tgt], V[t0, tgt], d)[:, 0]\n"
         "            ensm = integrate(tok, cen, X[t0, tgt], V[t0, tgt], d, members=ens['members'], seed=k * 10 + q)\n"
         "            for h in (6, 12, 24):\n"
         "                truth = X[t0 + d * h, tgt]\n"
         "                straight = X[t0, tgt] + d * h * V[t0, tgt]\n"
         "                for name, pred in (('model', det[h]), ('ensemble mean', ensm[h].mean(0)), ('straight motion', straight)):\n"
         "                    rows.append(dict(level=SIGMAS[k], patch=q, task=task, horizon_h=h, method=name,\n"
         "                                     error_km=float(np.median(gc_km(pred, truth))),\n"
         "                                     error_hpa=float(np.median(np.abs(1000 * np.exp(-pred[:, 2] / H_KM) - 1000 * np.exp(-truth[:, 2] / H_KM))))))\n"
         "            keep[(k, q, task)] = (tgt, det, ensm)\n"
         "import pandas as pd\nres = pd.DataFrame(rows)\n"
         "res.pivot_table(index=['task', 'horizon_h'], columns='method', values='error_km', aggfunc='median').round(0)"),
    code("res.pivot_table(index=['task', 'horizon_h'], columns='method', values='error_hpa', aggfunc='median').round(1)"),
    md("## 7. Tracks against the truth\nOne patch at sigma 0.285: true tracks (black dashed), the deterministic model "
       "(red), the ensemble members (light red) and straight motion (grey); left the forecast from 48 to 72 h, right the "
       "backcast from 24 back to 0 h."),
    code("k, q = 1, 0\nfig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained', subplot_kw=dict(projection=ccrs.NorthPolarStereo(central_longitude=235)))\n"
         "for ax, (task, d, t0) in zip(axes, (('forecast', 1, 48), ('backcast', -1, 24))):\n"
         "    tgt, det, ensm = keep[(k, q, task)]\n"
         "    ax.set_extent([195, 285, 30, 78], ccrs.PlateCarree()); ax.coastlines(lw=0.4, color='0.5')\n"
         "    ax.contour(z500.longitude, z500.latitude, z500.isel(time=1 if d > 0 else 0), levels=np.arange(5200, 6000, 60), colors='0.6', linewidths=0.5, transform=ccrs.PlateCarree())\n"
         "    hours = t0 + d * np.arange(25)\n"
         "    for j in range(0, len(tgt), 4):\n"
         "        for m in range(ensm.shape[1]):\n"
         "            lo, la = xy_to_lonlat(ensm[:, m, j]); ax.plot(lo, la, color='tab:red', alpha=0.12, lw=0.6, transform=ccrs.Geodetic())\n"
         "        for path, sty in ((X[hours, tgt[j]], dict(color='k', ls='--', lw=1.2)), (det[:, j], dict(color='tab:red', lw=1.4)),\n"
         "                          (X[t0, tgt[j]] + d * np.arange(25)[:, None] * V[t0, tgt[j]], dict(color='0.5', lw=1))):\n"
         "            lo, la = xy_to_lonlat(path); ax.plot(lo, la, transform=ccrs.Geodetic(), **sty)\n"
         "    ax.set_title(f'{task} ({\"48 to 72\" if d > 0 else \"24 back to 0\"} h after release), every 4th of 64 parcels', fontsize=9)\n"
         "plt.show()"),
    md("## 8. Clean up\nThe downloaded winds (about 34 GB) are deleted; the tracks stay in memory."),
    code("shutil.rmtree(WORK)\nprint('removed', WORK.relative_to(ROOT))"),
]
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "kinematic_interpolants", "display_name": "kinematic_interpolants", "language": "python"}
nbf.write(nb, ROOT / "ki_era5.ipynb")
print("wrote", ROOT / "ki_era5.ipynb", len(cells), "cells")
