#!/usr/bin/env bash
# Whole ERA5 pipeline, safe to run detached: tracks for every release (run_production.py), DDP training as soon as
# every training release has tracks (the test year keeps downloading meanwhile), then evaluation and figures once
# the test releases are done. Waits first for any production run already going (for example a pilot).
# Usage: setsid nohup bash scripts/run_all.sh [gpus, e.g. 1,2] > outputs/logs/run_all.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPUS="${1:-1,2}"
PYG="${PYG:-$HOME/.conda/envs/era5_gcs/bin/python}"               # download, LAGRANTO, datasets
PYT="${PYT:-$HOME/.conda/envs/kinematic_interpolants/bin/python}"  # training, evaluation
LOGS="$ROOT/outputs/logs"
mkdir -p "$LOGS/production"
cd "$ROOT"
export NUMPY_MADVISE_HUGEPAGE=0
stamp() { echo "$(date '+%F %T') $*"; }

while pgrep -f "scripts/run_[p]roduction.py" > /dev/null; do sleep 60; done
stamp "production for all releases"
rm -f "$LOGS/production/DONE" "$LOGS/production/FAILED"
"$PYG" scripts/run_production.py --split all --max-releases 4 > "$LOGS/production/production.log" 2>&1 &
PROD=$!

n_train=$("$PYG" -c "import json; c = json.load(open('configs/releases.json')); print(sum(r['split'] == 'train' for r in c['releases']))")
n_have() { ls data/tracks/ 2>/dev/null | grep -c '^tracks_tr[0-9]*\.npz$'; }
while [ "$(n_have)" -lt "$n_train" ]; do
  if ! kill -0 "$PROD" 2>/dev/null; then stamp "production stopped with $(n_have) of $n_train training releases"; exit 1; fi
  sleep 300
done
stamp "all $n_train training releases have tracks: training on GPUs $GPUS"
n=$(awk -F, '{print NF}' <<< "$GPUS")
CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=4 "$PYT" -m torch.distributed.run --standalone --nproc-per-node "$n" \
  scripts/20_train_context.py > "$LOGS/train_era5.log" 2>&1 || { stamp "training failed (train_era5.log)"; exit 1; }
stamp "training finished; waiting for the test releases"
wait "$PROD"
[ -f "$LOGS/production/DONE" ] || { stamp "production did not finish (production.log)"; exit 1; }
stamp "evaluation"
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PYT" scripts/21_eval_context.py > "$LOGS/eval_era5.log" 2>&1 &&
  "$PYT" scripts/22_figures.py >> "$LOGS/eval_era5.log" 2>&1 || { stamp "evaluation failed (eval_era5.log)"; exit 1; }
stamp "done"
touch "$LOGS/RUN_ALL_DONE"
