# Source edits to lagranto.ecmwf trunk r47

Everything else is the unmodified trunk; gfortran 15 needs only compiler flags (see `build.sh`). The two edits
below were unavoidable for our use; each is the smallest change that removes the problem.

| # | File | Line(s) after edit | Change | Why |
|---|------|--------------------|--------|-----|
| 1 | `lib/iotra.f` | 57 (`ropen_tra`), 109 (`wopen_tra`) | inserted `fid = 10` before `open(fid,file=filename,form='unformatted')` | Trajectory format 3 (binary, `.3`) opened a Fortran unit that was never set; the ASCII modes already set `fid = 10`. With gfortran 15, one of four consecutive `caltra ... out.3` runs failed with `Fortran runtime error: Bad unit number in OPEN statement` (iotra.f line 108 before the edit); the others happened to find a valid value in the uninitialised variable. Format 3 is the only output that keeps full single precision (ASCII format 1 rounds lon/lat to 0.01 deg, about 1 km), so the verification and our training data need it. |
| 2 | `caltra/caltra.f` | 24 | `parameter (ndatmax=500)` -> `parameter (ndatmax=5000)` | Maximum number of input P files per run. 30 days of hourly files is 721 files; unpatched caltra stops with `ERROR: too many input files 721 500`. The array it sizes is `character*11 dat(ndatmax)`, so the cost is 55 kB. |
| 3 | `trace/trace.f90` | 27 | `parameter (ndatmax=500)` -> `parameter (ndatmax=5000)` | Same limit in trace. |

Not edited (left as known upstream issues, see README "Gotchas"):

- `caltra.f` `euler_2d`/`euler_3d`: the zonal step uses `cos` of the start latitude of the step, which makes the
  scheme first order in time when parcels move meridionally (measured; README accuracy table).
- `caltra.sh -p` writes `per = 1` (degree) into `caltra.param`; on a global grid that wraps longitudes modulo 1 deg.
- `trace.f90` line 1762: `calc_DTDY` is called with one argument too many (only matters when tracing `DTHDY`).
- `trace` ends with Fortran `stop` (exit status 0) on its input errors, so `trace.sh` reports success; check that the
  output file exists.
