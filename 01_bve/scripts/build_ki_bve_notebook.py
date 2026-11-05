"""Write ki_bve.ipynb: the ocean notebook's cell layout, with the model code copied verbatim from scripts/ki_tracks.py."""
import ast
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "scripts/ki_tracks.py").read_text()
TREE = ast.parse(SRC)


def source(*names):
    nodes = [n for n in TREE.body if getattr(n, "name", None) in names]
    return "\n\n\n".join(ast.get_source_segment(SRC, n) for n in nodes)


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# BVE: Kinematic Interpolants on LAGRANTO parcel tracks\n\n"
       "Explicit version of the ocean notebook (`ki/notebooks/oceans_currents_interpolant.ipynb`) for air parcels of our "
       "stochastically stirred barotropic jet. Same two networks (acceleration and velocity score: ReLU MLPs with polynomial "
       "and Fourier features of time), same rotating-noise construction, Adam and EMA. What changes: the parcels have "
       "identities (LAGRANTO tracks), so the mean path is each parcel's own track and the target is its true acceleration; "
       "the split is by parcel (70 % training, 10 % validation, 20 % test) inside one compact patch released together, as "
       "in the ocean data. The model sees the window 24-120 h after release; the tests cut a random segment out of every "
       "test parcel's track and fill it, forecast after the window and backcast before it, and repeat all three in flows "
       "the model never saw. Run in order in the `kinematic_interpolants` kernel; one patch takes about 3 min on "
       "an A100."),
    md("## 1. Imports"),
    code("from pathlib import Path\nfrom types import SimpleNamespace\nfrom copy import deepcopy\nimport json, time\n"
         "import numpy as np\nimport torch\nfrom torch import nn\nimport xarray as xr\nimport matplotlib.pyplot as plt\n"
         "ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'scripts/ki_tracks.py').is_file())"),
    md("## 2. Editable hyperparameters\nOne patch of window 0 by default; set `PATCHES = list(range(8))` for all eight."),
    code("BVE_RUN = ROOT / 'data/bve_stirred_jet.nc'\nTRACKS = ROOT / 'data/tracks_lagranto_w0.npz'\nPATCHES = [0]\n"
         "T_START_H, T_END_H = 24.0, 120.0   # training window, hours after release\n"
         "TRAIN_STEPS = 20000\nBATCH_SIZE = 1024\nWIDTH = 512\nDEPTH = 4\n"
         "FREQS = [2.0**k for k in range(int(np.log2((T_END_H - T_START_H) / 6)) + 1)]  # Fourier periods down to 6 h\n"
         "LEARNING_RATE = 1e-3\nEMA_DECAY = 0.99\nTRAIN_NOISE = 1.0\nDELTA_H = 6.0   # time scale of the rotating noise\n"
         "SCORE_LOSS_WEIGHT = 1.0\nGAPS_H = [6, 12, 24, 48]\nHORIZONS_H = [6, 12, 24]   # forecast and backcast\nSTEPS_PER_HOUR = 6\n"
         "UNSEEN = {'trained flow, other patch': (TRACKS, 1),\n"
         "          'new perturbation': (ROOT / 'data/tracks_lagranto_seed2_w0.npz', 0),\n"
         "          'new jet and perturbation': (ROOT / 'data/tracks_lagranto_jet40n_w0.npz', 0)}\n"
         "RUNS = {'trained flow': 'bve_stirred_jet.nc', 'new perturbation': 'bve_seed2.nc', 'new jet and perturbation': 'bve_jet40n.nc'}\n"
         "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\nSEED = 0"),
    md("## 3. The BVE run and the LAGRANTO tracks\nRelative vorticity at the release with the eight patch centres; "
       "each patch is 1000 parcels in a 300-km disk, advected by LAGRANTO (2-D mode) with the hourly BVE wind."),
    code(source("lonlat_to_stereo", "stereo_to_lonlat", "great_circle_km") + "\n\n\n"
         "bve = xr.open_dataset(BVE_RUN)\ntracks = np.load(TRACKS)\n"
         "first = bve.time.values[0] + np.timedelta64(int(round(tracks['start_hour'] + tracks['time_h_abs'][0])), 'h')\n"
         "z = bve.zeta.sel(time=first, lat=slice(90, 15))\nLON, LAT = np.meshgrid(z.lon, z.lat)\n"
         "XY = lonlat_to_stereo(LON, LAT) / 1e3\nfig, ax = plt.subplots(figsize=(6.4, 6), layout='constrained')\n"
         "pc = ax.pcolormesh(XY[..., 0], XY[..., 1], z * 1e4, cmap='RdBu_r', vmin=-1.2, vmax=1.2, shading='gouraud')\n"
         "for k in np.unique(tracks['patch']):\n    s = tracks['patch'] == k\n"
         "    xy = tracks['x_km'][0, s] / 1e3\n    ax.plot(*xy.T, '.', ms=1, color='k')\n"
         "    ax.text(*xy.mean(0), f' {k}', fontsize=11, weight='bold')\n"
         "ax.set_aspect('equal'); ax.set(xlabel='x (1000 km)', ylabel='y (1000 km)', "
         "title=f'relative vorticity at release ({str(first)[:13]}), patches 0-7')\n"
         "fig.colorbar(pc, ax=ax, label='$10^{-4}$ s$^{-1}$', shrink=0.8)\nplt.show()\n"
         "print({k: tracks[k].shape for k in ('x_km', 'v_kmh', 'a_kmh2', 'patch', 'split')}, "
         "'split counts', np.bincount(tracks['split']))"),
    md("## 4. Acceleration and score networks\nThe paper's `Field`, with width, depth and time frequencies as arguments "
       "(the ocean runs used width 256, depth 2, frequencies 1, 2, 4 per window); `KINet` holds both networks and returns both "
       "from one call, which is what DistributedDataParallel needs for multi-GPU training (`scripts/05_train.py`)."),
    code("OMEGA = np.pi * np.sqrt(3)\n\n\n" + source("Field", "KINet")),
    md("## 5. Training minibatches\nPositions are centred and scaled by the training parcels, time by the window, "
       "exactly as in the ocean notebook. Each sample is a training parcel at a random time, moved off its track by the "
       "rotating Gaussian noise; the target acceleration is the true one plus the noise's restoring term."),
    code(source("load_patch", "training_batch", "batch_loss")),
    md("## 6. Explicit training loop\nAdam on acceleration MSE plus score MSE; EMA weights are evaluated on the "
       "validation parcels (median error of 24-h gaps) and the best ones are kept."),
    code(source("integrate", "infill", "to_km") + "\n\n\n"
         "def gap_errors(net, d, parcels, gap_h, rng, reps=1):\n"
         "    L, n, out = gap_h / d.clock, int(gap_h * STEPS_PER_HOUR), []\n"
         "    tt = lambda arr, k: torch.tensor(arr[k, parcels], dtype=torch.float32, device=DEVICE)\n"
         "    for _ in range(reps):\n"
         "        k0 = np.searchsorted(d.t, rng.uniform(0, 1 - L, len(parcels)))\n"
         "        k1 = np.searchsorted(d.t, d.t[k0] + L - 1e-9)\n"
         "        path = infill(net, d, tt(d.x, k0), tt(d.v, k0), tt(d.x, k1), tt(d.v, k1), d.t[k0], d.t[k1], n).cpu().numpy()\n"
         "        tq = d.t[k0][None] + (d.t[k1] - d.t[k0])[None] * np.linspace(0, 1, n + 1)[:, None]\n"
         "        truth = np.stack([np.stack([np.interp(tq[:, j], d.t, d.x[:, q, c]) for c in (0, 1)], -1) for j, q in enumerate(parcels)], 1)\n"
         "        (lp, ap), (lt, at) = stereo_to_lonlat(to_km(d, path)), stereo_to_lonlat(to_km(d, truth))\n"
         "        out.append(dict(err=great_circle_km(lp, ap, lt, at), k0=k0, k1=k1, path=path, truth=truth))\n"
         "    return out\n\n\n"
         "def train_one_fit(d, seed):\n"
         "    torch.manual_seed(seed); rng = np.random.default_rng(seed)\n"
         "    model = KINet(width=WIDTH, depth=DEPTH, freqs=FREQS).to(DEVICE)\n"
         "    ema = deepcopy(model).requires_grad_(False)\n"
         "    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)\n"
         "    best, best_state, losses, val = np.inf, None, [], []\n"
         "    for step in range(1, TRAIN_STEPS + 1):\n"
         "        loss = batch_loss(model, training_batch(d, rng, d.train, BATCH_SIZE, DEVICE), SCORE_LOSS_WEIGHT)\n"
         "        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())\n"
         "        with torch.no_grad():\n"
         "            for e, w in zip(ema.parameters(), model.parameters()):\n"
         "                e.lerp_(w, 1 - EMA_DECAY)\n"
         "        if step % 1000 == 0:\n"
         "            v = float(np.median(gap_errors(ema, d, d.val, 24, np.random.default_rng(1))[0]['err'].mean(0)))\n"
         "            val.append((step, v))\n"
         "            if v < best:\n"
         "                best, best_state = v, deepcopy(ema.state_dict())\n"
         "            print(f'  {step}/{TRAIN_STEPS}: loss {loss.item():.4g}, validation 24-h gaps median {v:.1f} km', flush=True)\n"
         "    ema.load_state_dict(best_state)\n"
         "    return ema.eval(), losses, val\n\n\n"
         "datasets, models, histories = {}, {}, {}\n"
         "for patch in PATCHES:\n"
         "    datasets[patch] = load_patch(TRACKS, patch, T_END_H, TRAIN_NOISE, DELTA_H, T_START_H)\n"
         "    print('Training patch', patch)\n"
         "    models[patch], *histories[patch] = train_one_fit(datasets[patch], SEED)"),
    md("## 7. Three abilities: fill, forecast, backcast\nEvery test parcel loses a random segment of 6, 12, 24 or 48 h inside "
       "the training window; the model integrates forward from the segment's start and backward from its end, and the two "
       "are blended. The forecast starts every test parcel at the end of the window (120 h after release) and integrates "
       "forward; the backcast starts it at the beginning of the window (24 h) and integrates backward, toward the release. "
       "`zero acceleration` (straight motion at the known velocities) is the control that shows what training adds."),
    code("class Zero(nn.Module):\n    def forward(self, t, x, v):\n        return torch.zeros_like(x)\n\n\n"
         "zero = nn.ModuleList([Zero(), Zero()])\n\n\n"
         "def abilities(net, d, rng, base):\n"
         "    rows, ex, pj = [], {}, d.test\n"
         "    tt = lambda arr, k: torch.tensor(arr[k, pj], dtype=torch.float32, device=DEVICE)\n"
         "    for gap in GAPS_H:\n"
         "        res = gap_errors(net, d, pj, gap, rng, reps=3)\n"
         "        ex[gap] = res[0]\n"
         "        e = np.concatenate([r['err'].mean(0) for r in res])\n"
         "        rows.append(dict(base, ability='fill', horizon_h=gap, median_km=np.median(e), p90_km=np.percentile(e, 90)))\n"
         "    for ability, t0, sign in (('forecast', 1.0, 1), ('backcast', 0.0, -1)):\n"
         "        k0 = int(np.argmin(np.abs(d.t_all - t0)))\n"
         "        for h in HORIZONS_H:\n"
         "            k1 = int(np.argmin(np.abs(d.t_all - (d.t_all[k0] + sign * h / d.clock))))\n"
         "            end = integrate(net, d, tt(d.x_all, k0), tt(d.v_all, k0), d.t_all[k0], d.t_all[k1], int(h * STEPS_PER_HOUR))[-1].cpu().numpy()\n"
         "            (lp, ap), (lt, at) = stereo_to_lonlat(to_km(d, end)), stereo_to_lonlat(to_km(d, d.x_all[k1, pj]))\n"
         "            e = great_circle_km(lp, ap, lt, at)\n"
         "            rows.append(dict(base, ability=ability, horizon_h=h, median_km=np.median(e), p90_km=np.percentile(e, 90)))\n"
         "    return rows, ex\n\n\n"
         "rows, examples = [], {}\n"
         "for patch, d in datasets.items():\n"
         "    for label, net in (('trained', models[patch]), ('zero acceleration', zero)):\n"
         "        r, examples[(patch, label)] = abilities(net, d, np.random.default_rng(100 + patch), dict(patch=patch, data='trained flow, same patch', model=label))\n"
         "        rows += r"),
    md("## 8. Median errors\nGreat-circle distance to the true position (averaged along the gap for `fill`; at the end "
       "point for `forecast` and `backcast`), median and 90th percentile over the test parcels."),
    code("import pandas as pd\nsummary = pd.DataFrame(rows)\n"
         "summary.pivot_table(index=['ability', 'horizon_h'], columns='model', values=['median_km', 'p90_km'], aggfunc='median', sort=False).round(1)"),
    md("## 9. Visualize\nLeft: training parcels of the patch (grey) and eight test parcels whose 24-h gap was filled "
       "(red dashed = model, blue = truth, black dots = the known ends). Right: error against horizon for the three "
       "abilities (solid = trained, dashed = no learning)."),
    code("patch = PATCHES[0]; d = datasets[patch]; ex = examples[(patch, 'trained')][24]\n"
         "fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')\n"
         "trk = to_km(d, d.x) / 1e3\nax[0].plot(trk[:, d.train[::10], 0], trk[:, d.train[::10], 1], color='0.85', lw=0.4)\n"
         "pick = np.argsort(ex['err'].mean(0))[np.linspace(0, len(d.test) - 1, 8).astype(int)]\n"
         "for j in pick:\n"
         "    tr, pa = to_km(d, ex['truth'][:, j]) / 1e3, to_km(d, ex['path'][:, j]) / 1e3\n"
         "    ax[0].plot(*tr.T, color='tab:blue', lw=2); ax[0].plot(*pa.T, color='tab:red', lw=1.2, ls='--')\n"
         "    ax[0].plot(*tr[[0, -1]].T, 'ko', ms=3)\n"
         "ax[0].set_aspect('equal'); ax[0].set(xlabel='x (1000 km)', ylabel='y (1000 km)', title=f'patch {patch}: filled 24-h gaps (red) vs truth (blue)')\n"
         "s = summary[summary.patch == patch]\n"
         "for ability, color in (('fill', 'tab:red'), ('forecast', 'tab:purple'), ('backcast', 'tab:green')):\n"
         "    for model, ls in (('trained', '-'), ('zero acceleration', '--')):\n"
         "        g = s[(s.ability == ability) & (s.model == model) & (s.horizon_h <= 24)]\n"
         "        ax[1].plot(g.horizon_h, g.median_km, marker='o', ls=ls, color=color, label=f'{ability}, {model}')\n"
         "ax[1].set(xscale='log', yscale='log', xlabel='horizon (h)', ylabel='median error (km)')\n"
         "ax[1].xaxis.set_minor_locator(plt.NullLocator()); ax[1].set_xticks(HORIZONS_H, [str(h) for h in HORIZONS_H])\n"
         "ax[1].legend(frameon=False, fontsize=8)\nplt.show()"),
    md("## 10. Flows the model never saw\nThe trained model of each patch is applied, with its own centre and scale, to: "
       "the next patch of the same flow (another region at the same time); and parcels released at the same positions and "
       "time in two other BVE runs, one with the same jet and a new random stirring (new perturbation), one with a 40 m/s "
       "jet at 40°N, 8° wide, and a new stirring (new jet and perturbation). Left: the jets (time-mean zonal-mean wind). "
       "Right: 24-h errors of the three abilities in each case."),
    code("fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained', gridspec_kw=dict(width_ratios=[1, 2.4]))\n"
         "for label, nc in RUNS.items():\n"
         "    u = xr.open_dataset(ROOT / 'data' / nc).u.mean(('time', 'lon'))\n"
         "    ax[0].plot(u, u.lat, label=label)\n"
         "ax[0].set(xlabel='zonal-mean u (m/s)', ylabel='latitude', ylim=(0, 90), title='jets of the three runs')\n"
         "ax[0].legend(frameon=False, fontsize=8)\n"
         "for patch in PATCHES:\n"
         "    ck = dict(center=datasets[patch].center, scale=datasets[patch].scale)\n"
         "    for label, (path, shift) in UNSEEN.items():\n"
         "        d_u = load_patch(path, (patch + shift) % 8, T_END_H, TRAIN_NOISE, DELTA_H, T_START_H, **ck)\n"
         "        for mlabel, net in (('trained', models[patch]), ('zero acceleration', zero)):\n"
         "            r, _ = abilities(net, d_u, np.random.default_rng(100 + patch), dict(patch=patch, data=label, model=mlabel))\n"
         "            rows += r\n"
         "summary = pd.DataFrame(rows)\n"
         "cases = ['trained flow, same patch', *UNSEEN]\n"
         "g = summary[(summary.horizon_h == 24)].groupby(['ability', 'data', 'model']).median_km.median()\n"
         "w = 0.13\n"
         "for i, ability in enumerate(('fill', 'forecast', 'backcast')):\n"
         "    for j, case in enumerate(cases):\n"
         "        ax[1].bar(i + (j - 1.5) * w * 1.6, g.get((ability, case, 'trained'), np.nan), w, color=plt.cm.Reds(0.9 - 0.18 * j), label=case if i == 0 else None)\n"
         "        ax[1].plot(i + (j - 1.5) * w * 1.6, g.get((ability, case, 'zero acceleration'), np.nan), 'k_', ms=12, label='no learning' if i == 0 and j == 0 else None)\n"
         "ax[1].set(yscale='log', ylabel='median error at 24 h (km)', title='trained model of each patch applied to unseen regions and flows')\n"
         "ax[1].set_xticks(range(3), ['fill a 24-h gap', 'forecast 24 h', 'backcast 24 h'])\n"
         "ax[1].legend(frameon=False, fontsize=8)\nplt.show()\n"
         "summary[summary.horizon_h == 24].pivot_table(index='data', columns=['ability', 'model'], values='median_km', aggfunc='median').round(1)"),
    md("## 11. Optional export\nSet `SAVE = True` to write the summary and the EMA weights."),
    code("SAVE = False\nif SAVE:\n    out = ROOT / 'outputs/notebook'; out.mkdir(parents=True, exist_ok=True)\n"
         "    summary.to_csv(out / 'summary.csv', index=False)\n"
         "    for patch, net in models.items():\n        torch.save(net.state_dict(), out / f'patch{patch}_s{int(T_START_H)}_t{int(T_END_H)}.pt')"),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = dict(name="kinematic_interpolants", display_name="kinematic_interpolants", language="python")
nbf.write(nb, ROOT / "ki_bve.ipynb")
print("wrote", ROOT / "ki_bve.ipynb")
