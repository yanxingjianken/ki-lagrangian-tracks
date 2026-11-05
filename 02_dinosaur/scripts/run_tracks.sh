#!/usr/bin/env bash
# Tracks for every flow and window: 3-D P files -> caltra -> per-level datasets; P files removed afterwards.
# Usage: bash scripts/run_tracks.sh [configs/flows.json] [windows, e.g. "0 1"] [parallel jobs]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FLOWS="${1:-$ROOT/configs/flows.json}"
WINDOWS="${2:-0 1}"
N_PAR="${3:-16}"
PY="${PY:-python}"  # run inside the kinematic_interpolants environment, or set PY
export PY ROOT OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$ROOT/outputs/logs/tracks"
names=$("$PY" -c "import json,sys; c=json.load(open(sys.argv[1])); c=c['flows'] if isinstance(c, dict) else c; print(' '.join(f['name'] for f in c if f.get('status', 'ok') == 'ok'))" "$FLOWS")
for n in $names; do for w in $WINDOWS; do echo "$n $w"; done; done | xargs -P "$N_PAR" -L 1 bash -c '
  n=$0; w=$1; done="$ROOT/outputs/logs/tracks/${n}_w${w}.done"; log="$ROOT/outputs/logs/tracks/${n}_w${w}.log"
  [ -f "$done" ] && exit 0
  [ -f "$ROOT/data/dino_${n}.nc" ] || { echo "missing data/dino_${n}.nc" > "$log"; exit 0; }
  { "$PY" "$ROOT/scripts/10_lagranto_input.py" --flow "$n" --window "$w" &&
    "$PY" "$ROOT/scripts/11_tracks.py" --flow "$n" --window "$w" &&
    "$PY" "$ROOT/scripts/12_dataset.py" --flow "$n" --window "$w" && rm -f "$ROOT/data/lagranto_${n}_w${w}.npz" &&
    rm -f "$ROOT/data/lagranto/${n}_w${w}"/P* "$ROOT/data/lagranto/${n}_w${w}/traj.3" && touch "$done"; } > "$log" 2>&1'
