#!/usr/bin/env bash
# Whole ERA5 pipeline, safe to run detached and to rerun: tracks for every release (run_production.py; a production
# run already going is adopted, otherwise one is started), DDP training once every training release has tracks and
# the GPUs have room for their shards, then evaluation and figures once the test releases are done.
# Usage: setsid nohup bash scripts/run_all.sh [gpus] > outputs/logs/run_all.log 2>&1 < /dev/null &
#   gpus: a list (e.g. 1,2) to wait for, or auto (default): the first NGPU of CANDIDATES (default 0,1,2) that have
#   room; GPU 0 only while no other user's process is on it.
# NEED_MB: free memory each GPU must have before training starts (a shard of about 72 GB plus the training step).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPUS="${1:-auto}"
NGPU="${NGPU:-2}"
CANDIDATES="${CANDIDATES:-0,1,2}"
NEED_MB="${NEED_MB:-78000}"
PYG="${PYG:-$HOME/.conda/envs/era5_gcs/bin/python}"               # download, LAGRANTO, datasets
PYT="${PYT:-$HOME/.conda/envs/kinematic_interpolants/bin/python}"  # training, evaluation
LOGS="$ROOT/outputs/logs"
mkdir -p "$LOGS/production"
cd "$ROOT"
export NUMPY_MADVISE_HUGEPAGE=0
stamp() { echo "$(date '+%F %T') $*"; }
production_alive() { pgrep -f "scripts/run_[p]roduction.py" > /dev/null; }
pick_gpus() {  # pick_gpus <candidates> <n>: first n candidates with NEED_MB free (GPU 0 only without other users)
  local ok=() i free uuid others
  for i in ${1//,/ }; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$i")
    [ "$free" -ge "$NEED_MB" ] || continue
    if [ "$i" = 0 ]; then
      uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0)
      others=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader |
               awk -F', ' -v u="$uuid" '$1 == u {print $2}' | xargs -r ps -o user= -p | grep -cv "^$USER$")
      [ "${others:-0}" -eq 0 ] || continue
    fi
    ok+=("$i")
  done
  [ "${#ok[@]}" -ge "$2" ] && (IFS=,; echo "${ok[*]:0:$2}")
}

if production_alive; then
  stamp "adopting the production run already going"
elif [ ! -f "$LOGS/production/DONE" ]; then
  stamp "production for all releases"
  rm -f "$LOGS/production/FAILED"
  "$PYG" scripts/run_production.py --split all --max-releases 4 >> "$LOGS/production/production.log" 2>&1 &
fi

n_train=$("$PYG" -c "import json; c = json.load(open('configs/releases.json')); print(sum(r['split'] == 'train' for r in c['releases']))")
n_have() { ls data/tracks/ 2>/dev/null | grep -c '^tracks_tr[0-9]*\.npz$'; }
while [ "$(n_have)" -lt "$n_train" ]; do
  production_alive || { stamp "production stopped with $(n_have) of $n_train training releases"; exit 1; }
  sleep 300
done

if [ -f "$LOGS/TRAIN_DONE" ]; then
  stamp "training already done"
else
  i=0
  while :; do
    if [ "$GPUS" = auto ]; then
      USE=$(pick_gpus "$CANDIDATES" "$NGPU")
    else
      USE=$(pick_gpus "$GPUS" "$(awk -F, '{print NF}' <<< "$GPUS")")
    fi
    [ -n "$USE" ] && break
    [ $((i % 12)) -eq 0 ] && stamp "waiting for GPUs (${GPUS}): free MiB $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr '\n' ' '), need ${NEED_MB} each"
    i=$((i + 1))
    sleep 300
  done
  GPUS="$USE"
  stamp "all $n_train training releases have tracks and GPUs $GPUS have room: training"
  n=$(awk -F, '{print NF}' <<< "$GPUS")
  CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=4 "$PYT" -m torch.distributed.run --standalone --nproc-per-node "$n" \
    scripts/20_train_context.py > "$LOGS/train_era5.log" 2>&1 || { stamp "training failed (train_era5.log)"; exit 1; }
  touch "$LOGS/TRAIN_DONE"
  stamp "training finished"
fi

while [ ! -f "$LOGS/production/DONE" ]; do
  if [ -f "$LOGS/production/FAILED" ] || ! production_alive; then
    sleep 60
    [ -f "$LOGS/production/DONE" ] && break
    stamp "production ended without finishing (production.log)"; exit 1
  fi
  sleep 300
done
until EVAL_GPU=$(NEED_MB=20000 pick_gpus "${CANDIDATES}" 1); [ -n "$EVAL_GPU" ]; do sleep 300; done
stamp "evaluation on GPU $EVAL_GPU"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYT" scripts/21_eval_context.py > "$LOGS/eval_era5.log" 2>&1 &&
  "$PYT" scripts/22_figures.py >> "$LOGS/eval_era5.log" 2>&1 || { stamp "evaluation failed (eval_era5.log)"; exit 1; }
stamp "done"
touch "$LOGS/RUN_ALL_DONE"
