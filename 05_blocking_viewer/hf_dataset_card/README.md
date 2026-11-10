---
license: cc-by-4.0
tags: [atmospheric-science, lagrangian, trajectories, era5, blocking]
---
# Air-parcel tracks in five blocking events (ERA5, LAGRANTO)

Dense clouds of true air-parcel tracks used by the page "Air-Parcel Tracks in Blocking Events"
(yanxingjianken.github.io/blocking-plots/rwb-prediction/) and its Hugging Face Space. Per event folder:

- `tracks.npy`: float16 (patches, 53, 480, 6) — x, y (north-polar stereographic, km), z = -7 km ln(p / 1000 hPa), and
  their rates of change (km/h) of the 480 parcels of each patch every 3 h from release to 156 h;
- `index.json`: per patch row its release name and time, start level (sigma of `sigma`), patch centre; the event's
  window, region and how they were chosen;
- `z500.bin` / `z500.json`: int16 metres (2, times, lat, lon): ERA5 500-hPa height and its departure from the zonal
  mean, every 6 h at 1 deg.

Patches of 480 parcels fill 300-km disks released every 12 h at start sigma 0.38, 0.285, 0.20 (pressure as a
fraction of the surface pressure); tracks computed with LAGRANTO (Sprenger and Wernli, 2015,
doi:10.5194/gmd-8-2569-2015) from hourly ERA5 winds on 37 pressure levels at 0.25 deg. Contains modified Copernicus
Climate Change Service information. Code: github.com/yanxingjianken/ki-lagrangian-tracks (05_blocking_viewer).
