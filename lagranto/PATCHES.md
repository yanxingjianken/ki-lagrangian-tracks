# Source edits to lagranto.ecmwf trunk r47

Everything else is the unmodified trunk; gfortran 15 needs only compiler flags (see `build.sh`). The edits below
were unavoidable for our use; each is the smallest change that removes the problem.

| # | File | Line(s) after edit | Change | Why |
|---|------|--------------------|--------|-----|
| 1 | `lib/iotra.f` | 57 (`ropen_tra`), 109 (`wopen_tra`) | inserted `fid = 10` before `open(fid,file=filename,form='unformatted')` | Trajectory format 3 (binary, `.3`) opened a Fortran unit that was never set; the ASCII modes already set `fid = 10`. With gfortran 15, one of four consecutive `caltra ... out.3` runs failed with `Fortran runtime error: Bad unit number in OPEN statement` (iotra.f line 108 before the edit); the others happened to find a valid value in the uninitialised variable. Format 3 is the only output that keeps full single precision (ASCII format 1 rounds lon/lat to 0.01 deg, about 1 km), so the verification and our training data need it. |
| 2 | `caltra/caltra.f` | 24 | `parameter (ndatmax=500)` -> `parameter (ndatmax=5000)` | Maximum number of input P files per run. 30 days of hourly files is 721 files; unpatched caltra stops with `ERROR: too many input files 721 500`. The array it sizes is `character*11 dat(ndatmax)`, so the cost is 55 kB. |
| 3 | `trace/trace.f90` | 27 | `parameter (ndatmax=500)` -> `parameter (ndatmax=5000)` | Same limit in trace. |
| 4 | `caltra/caltra.sh` | 64 (`set flag_w = "nil"`), 103-107 (`case -wfactor`), 432-457 (wfactor block) | New option `-wfactor <value>`; without it the 3-D vertical-velocity factor is 1. `-wfactor auto` runs the trunk's guess unchanged. The status line now names the source: `Vertical scaling factor (wfactor)  : 1. (default: OMEGA in Pa/s)`. | caltra advances pressure as `p1 = p0 + wfactor*w*deltat/100.` (`caltra.f` line 1057; p in hPa, dt in s), so OMEGA in Pa/s needs 1. The trunk script guessed the factor from `getmima OMEGA P850` on the first P file: it strips the decimal point of the f10.3 print (1000 x the value) and sets 100 when max - min < 400, i.e. when the 850-hPa OMEGA range is below 0.4 Pa/s. That catches every weak or horizontally uniform OMEGA field, including early baroclinic waves. Measured with the guess (`-wfactor auto`, which reproduces the trunk): omega = 0.02 cos(lon) sigma (1 - sigma) Pa/s (850-hPa range 0.005 Pa/s) gave wfactor 100 and a 48-h pressure error of 360 hPa median (623 max) instead of 0.018 (0.13); parcels following sigma surfaces over a 20-hPa surface-pressure wave (omega up to 0.016 Pa/s) gave 203 hPa instead of 0.0015. `lagranto_io.caltra_wfactor` reads the line back from the caltra log and raises unless the factor is 1. |
| 5 | `docu/man/caltra.0` | 67-69 | Added the `-wfactor value` entry under OPTIONAL. | Keeps `lagrantohelp caltra` in step with edit 4. |

Not edited (left as known upstream issues, see README "Gotchas"):

- `caltra.f` `euler_2d`/`euler_3d`: the zonal step uses `cos` of the start latitude of the step, which makes the
  scheme first order in time when parcels move meridionally (measured; README accuracy table).
- `caltra.f` `euler_3d` line 1028/1047: between the lowest model level and the surface, OMEGA is tapered linearly to
  zero at PS (`w0=w0*pind`). In a sigma model the surface OMEGA is DPS/Dt, so parcels below the lowest level do not
  follow a falling surface pressure and reach the ground; `-j` (line 1087) then puts them back at PS - 10 hPa.
  Measured in README "3-D: parcels near the ground".
- `docu/man/caltra.0` says that without `-j` a ground-touching trajectory "would stick at the same position". The
  code (`caltra.f` line 1092) flags it as leaving the domain: its positions are -999 for the rest of the run.
- `lib/ioinp_ive.f` (3-D input): when every `bklev`/`bklay` is below 0.001 the reader skips PS and sets it to 1000 hPa,
  so pure pressure-level files carry no topography; `lagranto_io.write_pfile_plev` sets bk = 0.002 on the bottom level
  (README "3-D trajectories from pressure levels").
- `caltra.sh -p` writes `per = 1` (degree) into `caltra.param`; on a global grid that wraps longitudes modulo 1 deg.
- `trace.f90` line 1762: `calc_DTDY` is called with one argument too many (only matters when tracing `DTHDY`).
- `trace` ends with Fortran `stop` (exit status 0) on its input errors, so `trace.sh` reports success; check that the
  output file exists.
