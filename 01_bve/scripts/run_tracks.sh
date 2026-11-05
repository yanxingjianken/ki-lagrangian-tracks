#!/usr/bin/env bash
# Tracks for every flow of a flow set and every window: P files -> caltra -2d -> dataset; P files removed afterwards.
# Usage: bash scripts/run_tracks.sh [configs/flows.json] [windows, e.g. "0 1 2 3"] [parallel jobs]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FLOWS="${1:-$ROOT/configs/flows.json}"
WINDOWS="${2:-0 1 2 3}"
N_PAR="${3:-8}"
PY="${PY:-python}"  # run inside the kinematic_interpolants environment, or set PY
export PY ROOT OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$ROOT/outputs/logs/tracks"
names=$("$PY" -c "import json,sys; print(' '.join(f['name'] for f in json.load(open(sys.argv[1]))['flows']))" "$FLOWS")
for n in $names; do for w in $WINDOWS; do echo "$n $w"; done; done | xargs -P "$N_PAR" -L 1 bash -c '
  n=$0; w=$1; done="$ROOT/outputs/logs/tracks/${n}_w${w}.done"; log="$ROOT/outputs/logs/tracks/${n}_w${w}.log"
  [ -f "$done" ] && exit 0
  { "$PY" "$ROOT/scripts/02_lagranto_input.py" --run "$n" --window "$w" &&
    "$PY" "$ROOT/scripts/03_tracks.py" --run "$n" --window "$w" --n-patches 20 --min-sep-km 1500 --lat-range 30 65 &&
    "$PY" "$ROOT/scripts/04_dataset.py" --run "$n" --window "$w" --every-h 1 &&
    rm -rf "$ROOT/data/lagranto/w${w}_${n}" && touch "$done"; } > "$log" 2>&1'
