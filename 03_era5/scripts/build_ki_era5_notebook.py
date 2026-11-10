"""Write ki_era5.ipynb: conditional forecasts and backcasts of ERA5 parcel tracks with the trained model.

Ocean-notebook layout (the template is ki/notebooks/oceans_currents_interpolant.ipynb). Nothing large is downloaded:
the trained conditional Kinematic Interpolant comes from the public Hugging Face model, and a sample of true ERA5 /
LAGRANTO tracks (eight patches of the 2010 Russian block) is read by byte range from the public Hugging Face dataset
of the blocking-tracks viewer (05_blocking_viewer). The model code is copied from scripts/ctx_model.py.
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
       "The trained model on true ERA5 parcel tracks, in the layout of the ocean notebook of the Kinematic Interpolant "
       "paper (Jiang, Marzouk, Bose & Pooladian, *Learning smooth dynamics via Kinematic Interpolants*, arXiv:2610.22519; "
       "`ki/notebooks/oceans_currents_interpolant.ipynb`). Steps:\n\n"
       "1. read eight patches of true tracks (480 air parcels each, computed by LAGRANTO from hourly ERA5 winds) of the "
       "2010 Russian block from the public Hugging Face dataset `yanxingjianken/blocking-tracks` (about 2.5 MB, by byte "
       "range; how the tracks are made: `scripts/era5_pfiles.py`, `scripts/11_tracks.py`, `scripts/12_dataset.py`);\n"
       "2. load the conditional Kinematic Interpolant trained on 2015–2017 from the public Hugging Face model "
       "`yanxingjianken/ki-era5-tracks`;\n"
       "3. **forecast**: from the tracks of 192 parcels of a patch over the 48 h before an anchor time, integrate 64 other "
       "parcels of the patch 24 h forward; **backcast**: from the 48 h after the anchor, integrate them 24 h backward; "
       "deterministically and as a 16-member ensemble;\n"
       "4. compare with the true tracks and with straight motion.\n\n"
       "The model never sees a wind field, and 2010 was not used in training. A GPU is used if there is one; on a CPU "
       "the whole notebook takes a few minutes."),
    md("## 1. Imports and settings"),
    code("from pathlib import Path\nimport json, math, warnings\nimport numpy as np\nimport pandas as pd\nimport torch\nfrom torch import nn\n"
         "import matplotlib.pyplot as plt\nimport cartopy.crs as ccrs\n"
         "warnings.filterwarnings('ignore', message='IProgress not found')   # progress bars without ipywidgets\n"
         "from huggingface_hub import HfFileSystem, hf_hub_download\n\n"
         "DATA_REPO, MODEL_REPO = 'yanxingjianken/blocking-tracks', 'yanxingjianken/ki-era5-tracks'\n"
         "EVENT = 'russia_2010'\n"
         "RELEASE = '2010-07-30T00'                    # patches released here; anchors 24 h and 48 h later\n"
         "SIGMAS = [0.38, 0.285]                       # start levels: pressure / surface pressure\n"
         "NEAR = (57.0, 40.0)                          # the four patches whose centres are nearest this point (deg N, deg E)\n"
         "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
         "H_KM, A_KM = 7.0, 6371.0\nprint('device', DEVICE)"),
    md("## 2. True tracks from the public dataset\n"
       "`index.json` lists every patch (release time, start level, centre); `tracks.npy` holds, per patch, the "
       "position (x, y in north-polar stereographic km, z = −7 km ln(p / 1000 hPa)) and velocity (km/h) of its 480 "
       "parcels every 3 h from release to 156 h, as float16 (positions to within a few km). Only the eight patch rows "
       "are read."),
    code("fs = HfFileSystem(); EV = f'datasets/{DATA_REPO}/{EVENT}'\n"
         "idx = json.loads(fs.read_text(f'{EV}/index.json'))\n"
         "NT, NP, NF = idx['row_shape']; STEP_H = idx['hours'][1] - idx['hours'][0]\n\n"
         "def gc_deg(lat1, lon1, lat2, lon2):\n"
         "    p1, p2, dl = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon2 - lon1)\n"
         "    return np.rad2deg(np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1)))\n\n"
         "chosen = []\n"
         "for s in SIGMAS:\n"
         "    k = int(np.argmin(np.abs(np.array(idx['sigma']) - s)))\n"
         "    rows = [p for p in idx['patches'] if p['release_time'] == RELEASE and p['level'] == k]\n"
         "    chosen += sorted(rows, key=lambda p: gc_deg(*p['centre'], *NEAR))[:4]\n"
         "with fs.open(f'{EV}/tracks.npy', 'rb') as f:\n"
         "    version = np.lib.format.read_magic(f)\n"
         "    shape, _, dtype = (np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0)(f)\n"
         "    off, nbytes = f.tell(), NT * NP * NF * dtype.itemsize\n"
         "    data = []\n"
         "    for p in chosen:\n"
         "        f.seek(off + p['row'] * nbytes); data.append(np.frombuffer(f.read(nbytes), dtype).reshape(NT, NP, NF))\n"
         "data = np.stack(data).astype(np.float32)               # (patch, time every 3 h, parcel, x y z u v w)\n"
         "X, V = data[..., :3], data[..., 3:]\n"
         "LEVEL = np.array([idx['sigma'][p['level']] for p in chosen])\n"
         "print(f\"{len(chosen)} patches released {RELEASE}: {X.shape[2]} parcels x {X.shape[1]} times every {STEP_H} h; \"\n"
         "      f\"centres {[tuple(p['centre']) for p in chosen[:4]]}\")"),
    code("zm = json.loads(fs.read_text(f'{EV}/z500.json')); _, NZT, NY, NX = zm['shape']\n"
         "def z500(t, field=0):\n"
         "    \"\"\"500-hPa height (field 0) or its departure from the zonal mean (field 1), m, at time t.\"\"\"\n"
         "    k = zm['times'].index(t)\n"
         "    with fs.open(f'{EV}/z500.bin', 'rb') as f:\n"
         "        f.seek(((field * NZT + k) * NY * NX) * 2); return np.frombuffer(f.read(NY * NX * 2), np.int16).reshape(NY, NX).astype(float)\n"
         "ZLON, ZLAT = np.array(zm['lon'], float), np.array(zm['lat'])\n"
         "ZLON = ZLON[0] + (ZLON - ZLON[0]) % 360        # increasing across 0 deg (e.g. 310 ... 460)\n\n"
         "def xy_to_lonlat(x):\n"
         "    r = np.hypot(x[..., 0], x[..., 1])\n"
         "    return np.rad2deg(np.arctan2(x[..., 0], -x[..., 1])) % 360.0, 90.0 - np.rad2deg(2 * np.arctan(r / (2 * A_KM)))\n"
         "hpa = lambda z: 1000.0 * np.exp(-z / H_KM)\n"
         "t_of = lambda h: str(np.datetime64(RELEASE, 'h') + np.timedelta64(int(h), 'h'))"),
    code("def base_map(ax, t):\n"
         "    ax.set_extent([5, 85, 38, 78], ccrs.PlateCarree()); ax.coastlines(lw=0.4, color='0.5')\n"
         "    ax.contourf(ZLON, ZLAT, z500(t, 1), levels=np.arange(-400, 401, 50), cmap='RdBu_r', alpha=0.45, transform=ccrs.PlateCarree())\n"
         "    ax.contour(ZLON, ZLAT, z500(t), levels=np.arange(4800, 6000, 80), colors='k', linewidths=0.5, transform=ccrs.PlateCarree())\n\n"
         "fig = plt.figure(figsize=(12, 5), layout='constrained')\n"
         "ax = fig.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo(central_longitude=45)); base_map(ax, t_of(48))\n"
         "b = fig.add_subplot(1, 2, 2); th = STEP_H * np.arange(NT)\n"
         "for i in range(len(chosen)):\n"
         "    col = 'tab:red' if LEVEL[i] == SIGMAS[1] else 'tab:blue'\n"
         "    for j in range(0, NP, 40):\n"
         "        lo, la = xy_to_lonlat(X[i, :25, j]); ax.plot(lo, la, lw=0.6, color=col, transform=ccrs.Geodetic())\n"
         "        b.plot(th, hpa(X[i, :, j, 2]), lw=0.5, color=col)\n"
         "ax.set_title(f'true tracks 0-72 h (every 40th parcel; red sigma {SIGMAS[1]}, blue {SIGMAS[0]}); 500-hPa height {t_of(48)}', fontsize=9)\n"
         "b.invert_yaxis(); b.set(xlabel='hours after release', ylabel='pressure (hPa)', title='pressure along the tracks')\nplt.show()"),
    md("## 3. The conditional Kinematic Interpolant\n"
       "A parcel's state is its position (x, y, z) and velocity. Context tokens are the observed parcels of the patch "
       "every 3 h over 48 h: time from the anchor, position relative to the patch centroid at the anchor (absolute "
       "height), sin(latitude) and velocity, each axis divided by its scale. A Perceiver encoder compresses them into "
       "96 latent vectors; a query state reads the latents by cross-attention and returns the acceleration and the "
       "scaled velocity score of the paper (code copied from `scripts/ctx_model.py`). The weights, the per-axis scales "
       "and the ensemble setting come from the public Hugging Face model."),
    code("N_FEAT = 8   # token and query features: time, x, y, z, sin(latitude), u, v, w\n\n\n" + source("ctx_model.py", "Fourier", "Block", "embed", "ContextKI")),
    code("ck = torch.load(hf_hub_download(MODEL_REPO, 'context_ki_era5.pt'), map_location=DEVICE, weights_only=False)\n"
         "ens = json.loads(Path(hf_hub_download(MODEL_REPO, 'ensemble.json')).read_text())\n"
         "model = ContextKI(d=ck['args']['d_model']).to(DEVICE).eval(); model.load_state_dict(ck['state'])\n"
         "SC = ck['scales']; tt = lambda v: torch.tensor(np.asarray(v), dtype=torch.float32, device=DEVICE)\n"
         "POS, VEL, ACC, TIME_H = tt(SC['pos_km']), tt(SC['vel_kmh']), tt(SC['acc_kmh2']), float(SC['time_h'])\n"
         "SIGMA_V = tt([5.0, 5.0, 0.01]) * float(np.pi * np.sqrt(3) / 6.0)   # training noise in velocity (km/h)\n"
         "print(f\"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters, trained {ck['args']['steps']:,} steps \"\n"
         "      f\"(weights of step {ck['step']:,}); scales {SC['vel_kmh']} km/h, {SC['acc_kmh2']} km/h2; ensemble eps_rel {ens['eps_rel']}\")"),
    md("## 4. Forecasts and backcasts\n"
       "**Forecast** at anchor 48 h after release: the context is 192 parcels of the patch observed every 3 h over "
       "0–48 h; 64 other parcels start from their true state at 48 h and are integrated to 72 h. **Backcast** at anchor "
       "24 h: the context is observed over 24–72 h; the parcels are integrated backward to 0 h. Deterministic tracks "
       "follow x' = v, v' = a (Heun, 10-minute steps); ensemble members add the score term and random kicks, "
       "dv = [a + (eps²/2) s / sigma_v] dt + eps dW with eps = eps_rel sigma_v, so that the spread matches the error "
       "on validation data. Straight motion keeps the anchor velocity."),
    code("def sin_lat(x):\n    r = torch.hypot(x[..., 0], x[..., 1]); return torch.cos(2 * torch.atan(r / (2 * A_KM)))\n\n"
         "def tokens(Xp, Vp, r0, d, ctx):\n"
         "    \"\"\"Context of one patch: 17 rows every 3 h from the anchor row r0 backward (d = 1) or forward (d = -1).\"\"\"\n"
         "    rows = r0 - d * np.arange(17); Xc, Vc = tt(Xp[rows][:, ctx]), tt(Vp[rows][:, ctx])\n"
         "    cen = tt(Xp[r0, :, :2].mean(0))\n"
         "    tau = tt(-d * STEP_H * np.arange(17) / TIME_H)[:, None, None].expand(17, len(ctx), 1)\n"
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
         "    (l1, p1), (l2, p2) = xy_to_lonlat(xa), xy_to_lonlat(xb)\n"
         "    return A_KM * np.deg2rad(gc_deg(p1, l1, p2, l2))\n\n"
         "rows, keep = [], {}\n"
         "for i in range(len(chosen)):\n"
         "    perm = np.random.default_rng(i).permutation(NP); ctx, tgt = perm[:192], perm[192:256]\n"
         "    for task, d, t0 in (('forecast', 1, 48), ('backcast', -1, 24)):\n"
         "        r0 = t0 // STEP_H; tok, cen = tokens(X[i], V[i], r0, d, ctx)\n"
         "        det = integrate(tok, cen, X[i, r0, tgt], V[i, r0, tgt], d)[:, 0]\n"
         "        ensm = integrate(tok, cen, X[i, r0, tgt], V[i, r0, tgt], d, members=ens['members'], seed=i)\n"
         "        for h in (6, 12, 24):\n"
         "            truth = X[i, r0 + d * h // STEP_H, tgt]\n"
         "            straight = X[i, r0, tgt] + d * h * V[i, r0, tgt]\n"
         "            for name, pred in (('model', det[h]), ('ensemble mean', ensm[h].mean(0)), ('straight motion', straight)):\n"
         "                rows.append(dict(level=LEVEL[i], patch=i, task=task, horizon_h=h, method=name,\n"
         "                                 error_km=float(np.median(gc_km(pred, truth))),\n"
         "                                 error_hpa=float(np.median(np.abs(hpa(pred[:, 2]) - hpa(truth[:, 2]))))))\n"
         "        keep[(i, task)] = (tgt, det, ensm)\n"
         "res = pd.DataFrame(rows)\n"
         "res.pivot_table(index=['task', 'horizon_h'], columns='method', values='error_km', aggfunc='median').round(0)"),
    code("res.pivot_table(index=['task', 'horizon_h'], columns='method', values='error_hpa', aggfunc='median').round(1)"),
    md("## 5. Tracks against the truth\nOne patch at sigma 0.285: true tracks every 3 h (black dashed), the "
       "deterministic model (red), the ensemble members (light red) and straight motion (grey) over the 500-hPa height "
       "(contours) and its departure from the zonal mean (shading) at the anchor time; left the forecast from 48 to "
       "72 h, right the backcast from 24 back to 0 h."),
    code("i = int(np.flatnonzero(LEVEL == SIGMAS[1])[0])\n"
         "fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained', subplot_kw=dict(projection=ccrs.NorthPolarStereo(central_longitude=45)))\n"
         "for ax, (task, d, t0) in zip(axes, (('forecast', 1, 48), ('backcast', -1, 24))):\n"
         "    tgt, det, ensm = keep[(i, task)]; r0 = t0 // STEP_H; base_map(ax, t_of(t0))\n"
         "    for j in range(0, len(tgt), 4):\n"
         "        for m in range(ensm.shape[1]):\n"
         "            lo, la = xy_to_lonlat(ensm[:, m, j]); ax.plot(lo, la, color='tab:red', alpha=0.12, lw=0.6, transform=ccrs.Geodetic())\n"
         "        for path, sty in ((X[i, r0 + d * np.arange(9), tgt[j]], dict(color='k', ls='--', lw=1.2)), (det[:, j], dict(color='tab:red', lw=1.4)),\n"
         "                          (X[i, r0, tgt[j]] + d * np.arange(25)[:, None] * V[i, r0, tgt[j]], dict(color='0.5', lw=1))):\n"
         "            lo, la = xy_to_lonlat(path); ax.plot(lo, la, transform=ccrs.Geodetic(), **sty)\n"
         "    ax.set_title(f'{task} ({\"48 to 72\" if d > 0 else \"24 back to 0\"} h after release), every 4th of 64 parcels', fontsize=9)\n"
         "plt.show()"),
    code("fig, ax = plt.subplots(figsize=(8, 3.5), layout='constrained')\n"
         "for task, d, t0, col in (('forecast', 1, 48, 'tab:red'), ('backcast', -1, 24, 'tab:blue')):\n"
         "    tgt, det, ensm = keep[(i, task)]; r0 = t0 // STEP_H\n"
         "    for j in range(0, len(tgt), 8):\n"
         "        ax.plot(t0 + d * np.arange(25), hpa(det[:, j, 2]), color=col, lw=1.2)\n"
         "        ax.plot(t0 + d * STEP_H * np.arange(9), hpa(X[i, r0 + d * np.arange(9), tgt[j], 2]), color='k', ls='--', lw=0.9)\n"
         "ax.invert_yaxis(); ax.set(xlabel='hours after release', ylabel='pressure (hPa)',\n"
         "                          title='pressure: model (red forecast, blue backcast) and truth (black dashed), every 8th parcel')\nplt.show()"),
]
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "kinematic_interpolants", "display_name": "kinematic_interpolants", "language": "python"}
nbf.write(nb, ROOT / "ki_era5.ipynb")
print("wrote", ROOT / "ki_era5.ipynb", len(cells), "cells")
