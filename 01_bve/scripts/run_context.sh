#!/usr/bin/env bash
# Train the conditional KI with DDP on the given GPUs (one node; for several nodes add --nnodes/--rdzv-* to torchrun),
# then score it. Usage: bash scripts/run_context.sh [gpus, e.g. 1,2] [extra args for 08_train_context.py]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPUS="${1:-1,2}"; shift || true
PY="${PY:-python}"  # run inside the kinematic_interpolants environment, or set PY
n=$(awk -F, '{print NF}' <<< "$GPUS")
export CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=4
"$PY" -m torch.distributed.run --standalone --nproc-per-node "$n" "$ROOT/scripts/08_train_context.py" "$@"
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PY" "$ROOT/scripts/09_eval_context.py"
"$PY" "$ROOT/scripts/10_figures_context.py"
