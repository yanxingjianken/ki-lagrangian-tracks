#!/usr/bin/env bash
# Train one model per patch on the window [T_START, T_END] h after release, spreading the patches over the GPUs.
# Usage: bash scripts/run_training.sh [data npz] [gpus, e.g. 1,2] [jobs per gpu]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${1:-$ROOT/data/tracks_lagranto_w0.npz}"
IFS=, read -r -a GPUS <<< "${2:-1,2}"
PER_GPU="${3:-4}"
T_START="${T_START:-24}"
T_END="${T_END:-120}"
PY="${PY:-python}"  # run inside the kinematic_interpolants environment, or set PY
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$ROOT/outputs/logs" "$ROOT/outputs/models"
stem="$(basename "$DATA" .npz)"
i=0
for patch in 0 1 2 3 4 5 6 7; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  name="${stem}_p${patch}_s${T_START}_t${T_END}"
  if [ ! -f "$ROOT/outputs/models/${name}.done" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/05_train.py" --data "$DATA" --patch "$patch" \
      --t-start-h "$T_START" --t-end-h "$T_END" > "$ROOT/outputs/logs/train_${name}.log" 2>&1 \
      && touch "$ROOT/outputs/models/${name}.done" &
  fi
  i=$((i + 1))
  if [ $((i % (${#GPUS[@]} * PER_GPU))) -eq 0 ]; then wait; fi
done
wait
touch "$ROOT/outputs/logs/run_training_${stem}.done"
