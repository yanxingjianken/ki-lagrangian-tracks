#!/usr/bin/env bash
# Clean rebuild of LAGRANTO (lagranto.ecmwf trunk r47) with gfortran + netCDF-Fortran from a micromamba env.
#
#   bash build.sh            # rebuild everything, log to build.log
#
# Environment overrides:
#   LAGRANTO_ENV    micromamba env with gfortran and netcdf-fortran (default: isca)
#   LAGRANTO_INPUT  ive (default): lib/ioinp_ive.f, the IVE netCDF reader; the only reader with caltra -2d (P.ML)
#                   cf:            lib/ioinp.f, the CF/CDO reader (lon/lat/lev/time, hyam/hybm); no -2d support
set -euo pipefail

LAGRANTO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LAGRANTO
ENV_NAME="${LAGRANTO_ENV:-isca}"
INPUT="${LAGRANTO_INPUT:-ive}"
LOG="$LAGRANTO/build.log"
exec > >(tee "$LOG") 2>&1

PREFIX="$(micromamba run -n "$ENV_NAME" bash -c 'echo "$CONDA_PREFIX"')"
export PATH="$PREFIX/bin:$PATH"
FC="$PREFIX/bin/gfortran"
NETCDF_INC="$("$PREFIX/bin/nf-config" --fflags)"
NETCDF_LIB="$("$PREFIX/bin/nf-config" --flibs) -Wl,-rpath,$PREFIX/lib"
FFLAGS="-O -fallow-argument-mismatch"

case "$INPUT" in
  ive) IOINP_SRC=ioinp_ive.f ;;
  cf)  IOINP_SRC=ioinp.f ;;
  *)   echo "LAGRANTO_INPUT must be ive or cf, got '$INPUT'"; exit 1 ;;
esac

echo "date        : $(date -Is)"
echo "host        : $(hostname)"
echo "LAGRANTO    : $LAGRANTO"
echo "env         : $ENV_NAME ($PREFIX)"
echo "compiler    : $("$FC" --version | head -1)"
echo "netcdf-f    : $("$PREFIX/bin/nf-config" --version)"
echo "FFLAGS      : $FFLAGS"
echo "NETCDF_INC  : $NETCDF_INC"
echo "NETCDF_LIB  : $NETCDF_LIB"
echo "input reader: $INPUT (lib/$IOINP_SRC -> lib/ioinp.a)"

run() { echo "+ $*"; "$@"; }

# ---- clean -------------------------------------------------------------------------------------------------
echo; echo "==== clean"
cd "$LAGRANTO"
rm -f lib/*.o lib/*.a lib/*.mod
for prog in caltra create_startf select trace density lidar; do
  rm -f "$prog/$prog" "$prog"/*.o "$prog"/*.mod
done
goodies="lsl2rdf changet extract getmima gettidiff getvars list2lsl lsl2list mergetra newtime reformat timeres trainfo difference datelist"
for tool in $goodies; do rm -f "goodies/$tool"; done
rm -f goodies/*.o goodies/*.mod
find bin -maxdepth 1 -type l -delete
rm -f startf

# ---- libraries ---------------------------------------------------------------------------------------------
echo; echo "==== lib"
cd "$LAGRANTO/lib"
for lib in iotra inter times libcdfio libcdfplus; do
  run "$FC" -c $FFLAGS $NETCDF_INC "$lib.f" -o "$lib.o"
  run ar rcs "$lib.a" "$lib.o"
  rm -f "$lib.o"
done
run "$FC" -c $FFLAGS $NETCDF_INC "$IOINP_SRC" -o ioinp.o
run ar rcs ioinp.a ioinp.o
rm -f ioinp.o *.mod

L="$LAGRANTO/lib"
LIBS_IN="-Wl,--start-group $L/times.a $L/iotra.a $L/ioinp.a $L/inter.a $L/libcdfio.a $L/libcdfplus.a -Wl,--end-group"
LIBS_TRA="-Wl,--start-group $L/iotra.a $L/times.a $L/libcdfio.a $L/libcdfplus.a -Wl,--end-group"
LIBS_VAR="-Wl,--start-group $L/ioinp.a $L/libcdfio.a $L/libcdfplus.a -Wl,--end-group"

link() {  # link <dir> <exe> <libs> <sources...>
  local dir=$1 exe=$2 libs=$3; shift 3
  cd "$LAGRANTO/$dir"
  run "$FC" $FFLAGS $NETCDF_INC -o "$exe" "$@" $libs $NETCDF_LIB
  rm -f ./*.o ./*.mod
  [[ -x "$exe" ]] || { echo "FAILED: $dir/$exe"; exit 1; }
}

# ---- core programs -----------------------------------------------------------------------------------------
echo; echo "==== core"
link caltra        caltra        "$LIBS_IN"  caltra.f
link create_startf create_startf "$LIBS_IN"  create_startf.f
link trace         trace         "$LIBS_IN"  trace.f90 calvar.f
link select        select        "$LIBS_TRA" special.f select.f
link density       density       "$LIBS_IN"  density.f
link lidar         lidar         "$LIBS_IN"  lidar.f

# ---- goodies -----------------------------------------------------------------------------------------------
echo; echo "==== goodies"
for tool in $goodies; do
  case "$tool" in
    gettidiff|newtime) libs="$L/times.a" ;;
    getmima|getvars)   libs="$LIBS_VAR" ;;
    *)                 libs="$LIBS_TRA" ;;
  esac
  link goodies "$tool" "$libs" "$tool.f"
done

# ---- links (as in install.csh links) -----------------------------------------------------------------------
echo; echo "==== links"
cd "$LAGRANTO"
chmod +x bin/lagranto bin/lagrantohelp ./*/*.sh   # the SVN export carries no exec bits
ln -sfn "$LAGRANTO/create_startf" "$LAGRANTO/startf"
cd "$LAGRANTO/bin"
lnk() { ln -sf "$1" "$2"; }
for suffix in .sh .ecmwf; do
  lnk "$LAGRANTO/bin/lagranto"     "lagranto$suffix"
  lnk "$LAGRANTO/bin/lagrantohelp" "lagrantohelp$suffix"
done
for suffix in "" .sh .ecmwf; do
  lnk "$LAGRANTO/caltra/caltra.sh"               "caltra$suffix"
  lnk "$LAGRANTO/create_startf/create_startf.sh" "create_startf$suffix"
  lnk "$LAGRANTO/create_startf/create_startf.sh" "startf$suffix"
  lnk "$LAGRANTO/select/select.sh"               "select$suffix"
  lnk "$LAGRANTO/select/select.sh"               "seltra$suffix"
  lnk "$LAGRANTO/trace/trace.sh"                 "trace$suffix"
  lnk "$LAGRANTO/density/density.sh"             "density$suffix"
  lnk "$LAGRANTO/lidar/lidar.sh"                 "lidar$suffix"
  for tool in $goodies; do lnk "$LAGRANTO/goodies/$tool.sh" "$tool$suffix"; done
done
ls -l "$LAGRANTO/bin" | sed -n '1,6p'
echo "... $(find "$LAGRANTO/bin" -maxdepth 1 -type l | wc -l) links in bin/"

# ---- check -------------------------------------------------------------------------------------------------
echo; echo "==== check"
missing=0
for exe in caltra/caltra create_startf/create_startf trace/trace select/select density/density lidar/lidar; do
  [[ -x "$LAGRANTO/$exe" ]] || { echo "missing $exe"; missing=1; }
done
for tool in $goodies; do [[ -x "$LAGRANTO/goodies/$tool" ]] || { echo "missing goodies/$tool"; missing=1; }; done
[[ $missing -eq 0 ]] || exit 1
"$LAGRANTO/goodies/gettidiff" 20000102_00 20000101_00
"$LAGRANTO/goodies/newtime" 20000101_00 25
echo
echo "Build complete. Use:  export LAGRANTO=$LAGRANTO; export PATH=\$LAGRANTO/bin:\$PATH"
