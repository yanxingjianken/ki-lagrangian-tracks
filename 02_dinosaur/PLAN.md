# 02_dinosaur plan — conditional forecasts and backcasts of 3-D parcel tracks in baroclinic life cycles

Same design as `01_bve` (context transformer reading the other parcels' observed tracks; two tasks: conditional
forecast and backcast, 6–24 h; in-distribution vs unseen flows), now in three dimensions with the primitive
equations.

## Model runs
- Dinosaur (JAX spectral primitive equations), THM93 set-up: T95, 15 sigma levels whose midpoints are THM93's
  (0.967 … 0.018), dry, frictionless, del^6 diffusion; basic state Z1 (SH77 jet, U0 scaled to `ujet`) + Z2 (LC1: none;
  LC2: lobes or polar cap); perturbation = the most unstable normal mode of zonal wavenumber m at 1 hPa surface
  pressure; 15 days, hourly output (northern hemisphere, the modes are symmetric).
- Scripts copied (not imported) from the THM93 replication, with the wavenumber and jet as arguments.
- 56 flows (`configs/flows.json`). Factors: ujet 35–60 m/s (step 5), basic-state type (LC1 without Z2, LC2 with
  lobes, LC2 with polar cap), mode wavenumber m 4–8, mode amplitude 0.5/1/2 hPa (270 combinations). Unseen inside the
  ranges: 4 (ujet, type, m) triples held out entirely (amp 1). Training: 48 combinations drawn from the remaining 258
  (fixed seed; every level of every factor at least 5 times). Unseen outside the ranges: LC1 30 m/s m6, LC2-lobes
  65 m/s m5, LC2-polarcap 45 m/s m9, LC1 65 m/s m3.
- Storage: only the regridded hourly fields are kept (about 2.8 GB per flow); raw output and LAGRANTO input files are
  deleted after use; roger/ stays well under 1 TB (01: 27 GB; 02: about 160 GB fields + 31 GB tracks; checked 2026-11-05: 26 TB free on data2, 3 TB RAM, GPUs 1-2 80 GB each).

## Winds for LAGRANTO
- omega from continuity in sigma coordinates: omega = sigma V.grad(ps) - integral_0^sigma div(ps V) dsigma'
  (no time derivative needed); checked against the hourly surface-pressure tendency.
- Regrid the northern-hemisphere Gaussian fields to a regular 1-degree grid 0–90N with the pole row; P files with
  U, V, OMEGA (Pa/s), PS (hPa), T; pure sigma: ak = 0, bk = sigma, bottom level first, identical lev/lay arrays.
- LAGRANTO full 3-D mode (no -2d); the vertical scaling factor must be 1 (caltra.sh guesses it from the 850-hPa omega
  range) — verified in each caltra log.

## Tracks
- Windows: 6.5 days starting on days 2 and 8 of each life cycle (growth/breaking, breaking/decay).
- Patches: 1000 parcels in a 300-km disk at one model level; 8 patch centres per level and window (30–65N, at least
  2000 km apart); 5 levels: sigma 0.967 (surface layer), 0.887, 0.674, 0.477, 0.241. Start pressure = sigma x local ps.
- 56 flows x 2 windows x 5 levels x 8 patches x 1000 parcels = 4.5e6 tracks (3.84e6 in the training flows).
- Dataset: hourly positions (polar-stereographic x, y in km; log-pressure height z = -H ln(p/1000 hPa), H = 7 km),
  spline velocity and acceleration, latitude and pressure.

## Learning
- 3-D context model: tokens and queries (tau, x, y, z, sin lat, u, v, w), per-axis scales; larger Perceiver
  (width 384, 96 latents, 6 encoder layers, 3 decoder layers); DDP on GPUs 1+2.
- Split by patch within each training flow, level and window: patches 0-5 train, 6 validation, 7 test.
- Evaluation: unseen patches of the training flows and every patch of the unseen flows; anchors 48/72/96 h;
  forecast and backcast 6/12/24 h; horizontal (great-circle) and vertical (pressure) errors; straight motion as
  reference; results by start level.
- Deliverables: `ki_dinosaur.ipynb` (ocean-notebook layout), README with Mermaid chart, figures with fresh-reader
  audits, tests, push.
