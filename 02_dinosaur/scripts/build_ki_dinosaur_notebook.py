"""Write ki_dinosaur.ipynb: the ocean notebook's cell layout for the 3-D conditional KI, model code copied from the scripts."""
import ast
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def source(file, *names):
    src = (ROOT / "scripts" / file).read_text()
    nodes = [n for n in ast.parse(src).body if getattr(n, "name", None) in names
             or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in ast.walk(n.targets[0])))]
    seg = lambda n: "".join(f"@{ast.get_source_segment(src, d)}\n" for d in getattr(n, "decorator_list", [])) + \
        ast.get_source_segment(src, n)
    return "\n\n\n".join(seg(n) for n in nodes)


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Dinosaur: 3-D conditional Kinematic Interpolant on LAGRANTO parcel tracks\n\n"
       "Explicit, ocean-notebook-style walk through the conditional KI (`ki/notebooks/oceans_currents_interpolant.ipynb` "
       "is the template), in three dimensions. The acceleration a(t, x, v) and velocity score of the paper are predicted from the state of a "
       "parcel *and* the observed tracks of the other parcels of its patch (a Perceiver encoder over those tracks, "
       "cross-attention from the query state). Two tasks: forecast (the patch was observed for the 48 h before the "
       "anchor time) and backcast (observed for the 48 h after it); in both, every parcel is unobserved on the other side, "
       "and the target parcels are not in the context. State: polar-stereographic x, y and log-pressure height z. Data: 3-D "
       "LAGRANTO tracks in 56 baroclinic life cycles of the Dinosaur primitive-equation core (THM93 set-up, 15 sigma levels), "
       "48 used for training and 8 unseen (4 inside, 4 outside the training ranges of jet speed, basic-state type, "
       "wavenumber and amplitude); patches start at 5 levels from the surface layer to about 240 hPa. By default the "
       "notebook loads the model trained with DDP by `scripts/20_train_context.py`; set `TRAIN = True` to train here."),
    md("## 1. Imports"),
    code("from pathlib import Path\nfrom copy import deepcopy\nimport json, math, sys\nimport numpy as np\nimport pandas as pd\n"
         "import torch\nfrom torch import nn\nimport xarray as xr\nimport matplotlib.pyplot as plt\n"
         "ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'scripts/ctx_model.py').is_file() and (p / 'PLAN.md').is_file())"),
    md("## 2. Editable hyperparameters"),
    code("CONFIG = json.loads((ROOT / 'configs/flows.json').read_text())\nFLOWS = [f for f in (CONFIG['flows'] if isinstance(CONFIG, dict) else CONFIG) if f.get('status', 'ok') == 'ok']\nWINDOWS = [0, 1]\nLEVELS = [0, 1, 2, 3, 4]\nH_KM = 7.0\n"
         "MODEL = ROOT / 'outputs/models/context_ki_3d.pt'\nTRAIN = False      # True: train here (single GPU)\n"
         "TRAIN_STEPS = 60000\nBATCH = 32\nCONTEXT_PARCELS = 192\nQUERIES = 256\nSPAN_H = 48\nHORIZON_H = 24\n"
         "LR = 3e-4\nEMA_DECAY = 0.999\nEVAL_WINDOWS = [0]\nEVAL_ANCHORS_H = [72]\nHORIZONS_H = [6, 12, 24]\n"
         "DEVICE = 'cuda'"),
    md("## 3. The flows\nBasic-state type, jet speed, mode wavenumber and amplitude of the 56 life cycles; left: eddy "
       "kinetic energy at the lowest model level against time (training grey, unseen coloured); right: omega near "
       "850 hPa (sigma 0.887) on day 7 for a training and an unseen flow."),
    code("pd.DataFrame(FLOWS)[[c for c in ('name', 'case', 'z2', 'ujet', 'm', 'amp', 'split', 'label') if c in FLOWS[0]]]"),
    code("fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), layout='constrained', gridspec_kw=dict(width_ratios=[1.2, 1, 1]))\n"
         "for f in FLOWS:\n"
         "    ds = xr.open_dataset(ROOT / f\"data/dino_{f['name']}.nc\")\n"
         "    u = ds.u.isel(level=-1, time=slice(None, None, 6)); v = ds.v.isel(level=-1, time=slice(None, None, 6))\n"
         "    eke = 0.5 * ((u - u.mean('lon')) ** 2 + (v - v.mean('lon')) ** 2).sel(lat=slice(20, 80)).mean(('lat', 'lon'))\n"
         "    ax[0].semilogy(np.arange(eke.size) / 4, eke, lw=0.6 if f['split'] == 'train' else 2, color='0.6' if f['split'] == 'train' else None)\n"
         "ax[0].set(xlabel='day', ylabel='eddy kinetic energy, lowest level (m2 s-2)', title='56 life cycles (coloured: unseen)')\n"
         "train0 = next(f['name'] for f in FLOWS if f['split'] == 'train'); test0 = next(f['name'] for f in FLOWS if f['split'] == 'test')\n"
         "for a_, name in zip(ax[1:], [train0, test0]):\n"
         "    w = xr.open_dataset(ROOT / f'data/dino_{name}.nc').omega.sel(level=0.887, method='nearest').isel(time=168).sel(lat=slice(15, 90))\n"
         "    LON, LAT = np.meshgrid(w.lon, w.lat); r = 2 * 6371 * np.tan(np.deg2rad(90 - LAT) / 2) / 1e3\n"
         "    pc = a_.pcolormesh(r * np.sin(np.deg2rad(LON)), -r * np.cos(np.deg2rad(LON)), w, cmap='RdBu_r', vmin=-1, vmax=1, shading='gouraud')\n"
         "    a_.set_aspect('equal'); a_.set(title=f'{name}: omega near 850 hPa, day 7 (Pa/s)', xlabel='x (1000 km)')\n"
         "fig.colorbar(pc, ax=ax[2], shrink=0.8)\nplt.show()"),
    md("## 4. Context model\nFourier features of the token and query coordinates, a Perceiver encoder (64 latents) "
       "over the context tokens, and a cross-attention decoder that returns the acceleration and the scaled velocity "
       "score of a query state. Everything is in normalized units relative to the anchor time and the patch centroid."),
    code(source("ctx_model.py", "POS", "VEL", "ACC", "TIME_H", "N_FEAT", "Fourier", "Block", "embed", "ContextKI")),
    md("## 5. Minibatches and rollouts\nA sample is a patch, an anchor time and a direction. Context: 128 parcels of "
       "the patch, every 3 h over the 48 h on the observed side. Queries: parcels on and around their true tracks within "
       "24 h on the unobserved side (the paper's rotating Gaussian noise; the target acceleration includes its restoring "
       "term). Rollouts integrate x' = v, v' = a forward (forecast) or backward (backcast) from the anchor."),
    code("from types import SimpleNamespace\n\n\n" + source("ctx_data.py", "OMEGA", "A_KM", "flow_names", "read_flows", "load_patches", "context",
                                                           "_scale", "sample_batch", "sin_lat", "rollout")),
    md("## 6. Training\nAdamW, warm-up and cosine decay, EMA weights. The released model was trained with "
       "`torchrun --nproc-per-node 2 scripts/20_train_context.py` (DDP); its history is plotted below."),
    code("if TRAIN:\n"
         "    train = load_patches(ROOT, flow_names(ROOT / 'configs/flows.json', 'train'), WINDOWS, LEVELS, range(6), device=DEVICE)\n"
         "    model = ContextKI().to(DEVICE); ema = deepcopy(model).requires_grad_(False)\n"
         "    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)\n"
         "    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * s / TRAIN_STEPS)))\n"
         "    gen = torch.Generator(device=DEVICE).manual_seed(0)\n"
         "    for step in range(1, TRAIN_STEPS + 1):\n"
         "        tok, qry, acc, score = sample_batch(train, BATCH, CONTEXT_PARCELS, QUERIES, SPAN_H, HORIZON_H, gen=gen)\n"
         "        a_hat, s_hat = model(tok, None, qry)\n"
         "        loss = ((a_hat - acc) ** 2).mean() + ((s_hat - score) ** 2).mean()\n"
         "        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()\n"
         "        with torch.no_grad():\n"
         "            for e, w in zip(ema.parameters(), model.parameters()):\n"
         "                e.lerp_(w, 1 - EMA_DECAY)\n"
         "        if step % 2000 == 0:\n"
         "            print(step, loss.item(), flush=True)\n"
         "    model = ema.eval()\n"
         "else:\n"
         "    ck = torch.load(MODEL, map_location=DEVICE, weights_only=False)\n"
         "    model = ContextKI(d=ck['args']['d_model']).to(DEVICE).eval(); model.load_state_dict(ck['state'])\n"
         "    hist = json.loads((MODEL.parent / (MODEL.stem + '_history.json')).read_text())\n"
         "    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), layout='constrained')\n"
         "    tr = np.asarray(hist['train']); k = 500\n"
         "    ax[0].semilogy(hist['step'][k - 1:], np.convolve(tr, np.ones(k) / k, 'valid'), color='k'); ax[0].set(xlabel='step', ylabel=f'training loss ({k}-step mean)')\n"
         "    ax[1].plot(hist['val_step'], hist['forecast_km'], 'o-', color='tab:purple', label='forecast')\n"
         "    ax[1].plot(hist['val_step'], hist['backcast_km'], 'o-', color='tab:green', label='backcast')\n"
         "    ax[1].set(xlabel='step', ylabel='validation 24-h median error (km)'); ax[1].legend(frameon=False)\n"
         "    print('trained on', hist['world_size'], 'GPUs; best step', ck['step']); plt.show()"),
    md("## 7. Conditional forecast and backcast on unseen patches and unseen flows\nTargets are 128 parcels that are not "
       "in the context. `no learning` is straight motion at the anchor velocity."),
    code(source("geo.py", "stereo_to_lonlat", "great_circle_km") + "\n\n\n"
         "class Straight(nn.Module):\n    def encode(self, tok):\n        return None\n\n    def decode(self, z, q):\n        return torch.zeros_like(q[..., 1:4]), None\n\n\n"
         "cases = [('trained flows, unseen patches', [f['name'] for f in FLOWS if f['split'] == 'train'], [7])]\n"
         "cases += [(f'unseen, {k} the training ranges', [f['name'] for f in FLOWS if f['split'] == 'test' and k in f.get('label', '')], list(range(8))) for k in ('inside', 'outside')]\n"
         "rows, examples = [], {}\n"
         "for label, names, patches in cases:\n"
         "    D = load_patches(ROOT, names, EVAL_WINDOWS, LEVELS, patches, device=DEVICE)\n"
         "    gen = torch.Generator(device=DEVICE).manual_seed(7)\n"
         "    pp, dd, tt = (torch.tensor(z, device=DEVICE).flatten() for z in np.meshgrid(np.arange(D.X.shape[0]), [1, -1], EVAL_ANCHORS_H, indexing='ij'))\n"
         "    perm = torch.rand(len(pp), D.X.shape[2], device=DEVICE, generator=gen).argsort(1)\n"
         "    kidx, qidx = perm[:, :CONTEXT_PARCELS], perm[:, CONTEXT_PARCELS:CONTEXT_PARCELS + 128]\n"
         "    tok, cen = context(D, pp, tt, dd, kidx, SPAN_H)\n"
         "    x0, v0 = D.X[pp[:, None], tt[:, None], qidx], D.V[pp[:, None], tt[:, None], qidx]\n"
         "    steps = tt[:, None] + dd[:, None] * torch.arange(HORIZON_H + 1, device=DEVICE)\n"
         "    truth = D.X[pp[:, None, None], steps[:, :, None], qidx[:, None]].transpose(0, 1).cpu().numpy()\n"
         "    for mlabel, m in (('conditional KI', model), ('no learning', Straight())):\n"
         "        path = rollout(m, tok, cen, x0, v0, dd, HORIZON_H).cpu().numpy()\n"
         "        examples[(label, mlabel)] = (tok.cpu(), cen.cpu(), path, truth, dd.cpu().numpy())\n"
         "        for h in HORIZONS_H:\n"
         "            (lp, ap), (lt, at) = stereo_to_lonlat(path[h][..., :2]), stereo_to_lonlat(truth[h][..., :2])\n"
         "            err = great_circle_km(lp, ap, lt, at)\n"
         "            dp = np.abs(1000 * (np.exp(-path[h][..., 2] / H_KM) - np.exp(-truth[h][..., 2] / H_KM)))\n"
         "            for task, sel in (('forecast', dd.cpu().numpy() > 0), ('backcast', dd.cpu().numpy() < 0)):\n"
         "                rows.append(dict(case=label, task=task, horizon_h=h, model=mlabel, median_km=float(np.median(err[sel])), median_hpa=float(np.median(dp[sel]))))\n"
         "summary = pd.DataFrame(rows)\n"
         "summary.pivot_table(index=['task', 'horizon_h'], columns=['case', 'model'], values=['median_km', 'median_hpa']).round(0)"),
    md("## 8. Visualize\nOne forecast and one backcast from an unseen patch of a training flow (top) and of the "
       "flow furthest from training (bottom): context tracks grey, model red dashed, truth blue, black dots = anchor states."),
    code("fig, axes = plt.subplots(2, 2, figsize=(12, 11), layout='constrained')\n"
         "for row, label in zip(axes, [cases[0][0], cases[-1][0]]):\n"
         "    tok, cen, path, truth, dd = examples[(label, 'conditional KI')]\n"
         "    for ax, b in zip(row, [int(np.argmax(dd > 0)), int(np.argmax(dd < 0))]):\n"
         "        ctx = tok[b].reshape(SPAN_H // 3 + 1, -1, 8)[..., 1:3].numpy() * POS[0] + cen[b].numpy()\n"
         "        ax.plot(ctx[:, :, 0] / 1e3, ctx[:, :, 1] / 1e3, color='0.8', lw=0.5)\n"
         "        ax.plot(truth[:, b, :12, 0] / 1e3, truth[:, b, :12, 1] / 1e3, color='tab:blue', lw=2)\n"
         "        ax.plot(path[:, b, :12, 0] / 1e3, path[:, b, :12, 1] / 1e3, color='tab:red', lw=1.2, ls='--')\n"
         "        ax.plot(truth[0, b, :12, 0] / 1e3, truth[0, b, :12, 1] / 1e3, 'ko', ms=3)\n"
         "        ax.set_aspect('equal'); ax.set(title=f\"{label}: {'forecast' if dd[b] > 0 else 'backcast'} 24 h\", xlabel='x (1000 km)', ylabel='y (1000 km)')\n"
         "plt.show()"),
    md("The vertical side of the same examples: pressure of 12 target parcels against hours from the anchor (truth "
       "blue, model red dashed), for parcels started near the surface and aloft."),
    code("fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, layout='constrained')\n"
         "tok, cen, path, truth, dd = examples[(cases[0][0], 'conditional KI')]\n"
         "for ax, b in zip(axes, [int(np.argmax(dd > 0)), int(np.argmax(dd < 0))]):\n"
         "    hours = np.arange(path.shape[0]) * (1 if dd[b] > 0 else -1)\n"
         "    pr = lambda z: 1000 * np.exp(-z / H_KM)\n"
         "    ax.plot(hours, pr(truth[:, b, :12, 2]), color='tab:blue', lw=1.5)\n"
         "    ax.plot(hours, pr(path[:, b, :12, 2]), color='tab:red', lw=1, ls='--')\n"
         "    ax.set(xlabel='hours from the anchor', title=f\"{'forecast' if dd[b] > 0 else 'backcast'}: pressure of target parcels\")\n"
         "axes[0].set_ylabel('pressure (hPa)'); axes[0].invert_yaxis()\nplt.show()"),
    md("## 9. Error against horizon in every case"),
    code("fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, layout='constrained')\n"
         "colors = dict(zip([c[0] for c in cases], ['tab:red', 'tab:orange', 'tab:purple', 'tab:brown']))\n"
         "for ax, task in zip(axes, ['forecast', 'backcast']):\n"
         "    for (label, mlabel), g in summary[summary.task == task].groupby(['case', 'model']):\n"
         "        ax.plot(g.horizon_h, g.median_km, 'o-' if mlabel == 'conditional KI' else 's:', color=colors[label], label=f'{label} ({mlabel})')\n"
         "    ax.set(xscale='log', yscale='log', xlabel='horizon (h)', title=task); ax.set_xticks(HORIZONS_H, [str(h) for h in HORIZONS_H]); ax.xaxis.set_minor_locator(plt.NullLocator())\n"
         "axes[0].set_ylabel('median error of target parcels (km)'); axes[1].legend(fontsize=7, frameon=False)\nplt.show()"),
    md("## 10. Test movies\nOne movie per test case, written by `scripts/23_movies.py`: forecast (left) and backcast "
       "(right) of 12 target parcels started near 480 hPa, over relative vorticity (colour), streamfunction (contours) "
       "and wind (arrows) on that level; below, their pressure against time. Grey = the other parcels' observed tracks "
       "(the model's input), blue = truth, red dashed = model."),
    code("from IPython.display import Video, display\n"
         "for mp4 in sorted((ROOT / 'figures/movies').glob('*.mp4')):\n"
         "    print(mp4.stem.replace('_', ' '))\n"
         "    display(Video(str(mp4), embed=True, width=900))"),
    md("## 11. Optional export\nSet `SAVE = True` to write the summary table."),
    code("SAVE = False\nif SAVE:\n    out = ROOT / 'outputs/notebook'; out.mkdir(parents=True, exist_ok=True)\n    summary.to_csv(out / 'summary.csv', index=False)"),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = dict(name="kinematic_interpolants", display_name="kinematic_interpolants", language="python")
nbf.write(nb, ROOT / "ki_dinosaur.ipynb")
print("wrote", ROOT / "ki_dinosaur.ipynb")
