# 03_era5 — conditional forecasts and backcasts of 3-D parcel tracks in ERA5

The `02_dinosaur` experiment on reanalysis winds. Air parcels are released in compact patches at 20 levels, from the
surface layer to about 100 hPa, over both hemispheres, every 5 days of 2015–2017 (training) and 2021 (test).
LAGRANTO computes full 3-D tracks from hourly ERA5 at 0.25° on its 37 pressure levels. A context transformer predicts
the acceleration and velocity score of the Kinematic Interpolant paper from the state of a parcel and the observed
tracks of the other parcels of its patch; two tasks, as in `02_dinosaur`:

- **conditional forecast**: the patch was observed for the 48 h before the anchor time; target parcels (not in the
  context) are integrated forward 6–24 h;
- **conditional backcast**: observed for the 48 h after the anchor; targets are integrated backward 6–24 h.

Tests: unseen patches of the training releases, the unseen year 2021, and patches released in 2021 at latitudes
never used in training (10–25° and 70–85°). `PLAN.md` holds the design and the measured resource budget.

## Winds

ERA5 is read anonymously from Google's ARCO store (`full_37-1h-0p25deg-chunk-1.zarr-v3`): u, v, vertical velocity
and surface pressure, hourly, 0.25°, global, 37 levels. Each hour becomes one LAGRANTO P file (465 MB). On pure
pressure levels (bk = 0) LAGRANTO's reader ignores the surface pressure and puts the ground at 1000 hPa everywhere;
the writer `write_pfile_plev` (`lagranto/python/lagranto_io.py`) therefore gives the bottom level bk = 0.002, which
makes caltra read PS and keep parcels above the terrain (`lagranto/tests/test_lagranto_plev.py`). Downloading is
the slowest step (about 150 MB/s, 7.6 h per year of data), so the P files are streamed: written in time order,
used, and deleted.

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| 0 | `scripts/00_releases.py` — 220 training releases (2015–2017) and 73 test releases (2021), every 5 days, hour cycling 00/06/12/18 UTC; 20 levels, 16 patches per level (8 per hemisphere, 25–70°), 8 more at unseen latitudes in 2021, 480 parcels per patch | `configs/releases.json` |
| 1 | `scripts/run_production.py` (detached) — downloads hours in time order with `scripts/era5_pfiles.py` workers, at most 400 GB of P files ahead of the earliest unfinished release; tracks each release once its 157 P files exist; deletes P files no unfinished release needs; resumes from finished releases | `data/pfiles/` (transient), `outputs/logs/production/` |
| 2 | `scripts/11_tracks.py` — seeds the patches (start pressure σ × bilinear PS), runs 8 caltra jobs of patch-ordered parcels (full 3-D, `-j`, 1-min step, 10-min output, wfactor 1 checked) | `data/lagranto/<release>.npz` (transient) |
| 3 | `scripts/12_dataset.py` — Southern-Hemisphere patches mirrored to the north; polar-stereographic x, y and z = −7 km ln(p/1000 hPa); spline velocity and acceleration, hourly, float32 | `data/tracks/tracks_<release>.npz` |
| 4 | `scripts/13_check_release.py` — tracks of two patches on a map and their pressure; kept/dropped parcels, displacement per level, parcels below the surface | `figures/check_<release>.png` |
| 14 | `scripts/14_scales.py` — per-axis velocity and acceleration scales of the model: spread of the training tracks with the largest 0.5 % capped | `configs/scales.json` |
| 20 | `scripts/20_train_context.py` (`torchrun`, DDP) — training set sharded by release across GPUs, float32 x, v, a on the GPUs | `outputs/models/context_ki_era5.pt`, live panels in `outputs/logs/` |
| 21 | `scripts/21_eval_context.py` — unseen patches of the training releases, 2021, 2021 at unseen latitudes; anchors 48/72/96 h; horizontal (km) and vertical (hPa) errors against straight motion | `outputs/eval/context_ki_era5.csv.gz` |
| 22 | `scripts/22_figures.py` | `figures/f1_conditional_ki_era5.png`, `figures/f2_by_level.png` |
| nb | `ki_era5.ipynb` (written by `scripts/build_ki_era5_notebook.py`) — self-contained run in the ocean-notebook layout: downloads 73 h of ERA5 (Pacific Northwest heat dome, June 2021), writes LAGRANTO files, computes the true tracks of 8 patches, loads the model from the public Hugging Face repository and forecasts / backcasts 24 h (deterministic and 16-member ensemble) against the truth and straight motion; deletes the downloaded winds at the end | — |
| all | `scripts/run_all.sh` (detached) — production for every release; training once all training releases have tracks; evaluation and figures once 2021 is done | `outputs/logs/run_all.log` |

```mermaid
graph TD
    R["00_releases.py -> configs/releases.json (2015-2017 training, 2021 test)"] --> O["run_production.py (detached)"]
    O --> D["era5_pfiles.py workers: ARCO hour -> P file (U, V, OMEGA, PS; 37 levels; bottom bk = 0.002)"]
    D --> T["11_tracks.py per release: 20 levels x 16 (24) patches x 480 parcels, 8 caltra runs"]
    T --> S["12_dataset.py -> data/tracks/tracks_release.npz (x, v, a, lat, p; SH mirrored)"]
    T --> X["P files no unfinished release needs are deleted"]
    S --> TR["20_train_context.py: DDP, training set sharded across GPUs 1+2"]
    TR --> E["21_eval_context.py: unseen patches, 2021, unseen latitudes"]
    S --> E
    E --> F["22_figures.py -> figures/f1, f2"]
```

## First release

Release tr147 (2017-01-05 18 UTC): 153 600 tracks, none dropped or incomplete; 23 min for 8 caltra runs with two
other releases running. Median 24-h displacement 660–950 km for start sigma 0.62–0.967 and 1640–2260 km for
sigma 0.10–0.38; median 24-h pressure change 29–64 hPa for sigma 0.43–0.967 and 4–36 hPa for sigma 0.10–0.38; no
parcel below the surface pressure.

![first release](figures/check_tr147.png)

## The model

`scripts/ctx_model.py` is the `02_dinosaur` model (width 512 here), with per-axis scales from `configs/scales.json`:
velocities (62.0, 60.7, 0.085) km/h and accelerations (5.37, 5.41, 0.125) km/h² for (x, y, z). Hourly vertical
accelerations of parcels in 0.25° ERA5 winds are large and short-lived (median 0.02 km/h², 99th percentile
0.65 km/h²); with a scale of the size of their typical value they would make up almost all of the loss, so each
axis is scaled by its own spread (each then contributes a comparable error when predicted as zero). `scripts/ctx_data.py` keeps only x, v, a in
float32 on the GPU (sin latitude follows from x) and draws context and query parcels disjointly. Each rank loads
only its shard of releases, so two A100s hold the 2.5e7 training tracks (about 71 GB each; the measured limit next to
a training step is 1.38e7 tracks per GPU).

## Tests

```bash
micromamba run -n kinematic_interpolants python -m pytest -q tests/test_pipeline.py
```
