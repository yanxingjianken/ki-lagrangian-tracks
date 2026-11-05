#!/usr/bin/env bash
# Train the ERA5 conditional KI with DDP on the given GPUs (one node; for several nodes add --nnodes/--rdzv-* to
# torchrun; each rank holds its own shard of the training releases), then score it.
# Usage: bash scripts/run_context.sh [gpus, e.g. 1,2] [training args]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPUS="${1:-1,2}"; shift || true
PY="${PY:-$HOME/.conda/envs/kinematic_interpolants/bin/python}"
n=$(awk -F, '{print NF}' <<< "$GPUS")
export CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=4
"$PY" -m torch.distributed.run --standalone --nproc-per-node "$n" "$ROOT/scripts/20_train_context.py" "$@"
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PY" "$ROOT/scripts/21_eval_context.py"
