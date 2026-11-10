#!/usr/bin/env bash
# After run_production.py has finished (outputs/logs/production/DONE): package every event and upload it to the
# Hugging Face dataset. Usage: bash scripts/package_all.sh  (env: PYG with gcsfs, PYH with huggingface_hub)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYG="${PYG:-$HOME/.conda/envs/era5_gcs/bin/python}"
PYH="${PYH:-$HOME/.conda/envs/blocking/bin/python}"
cd "$ROOT"
until [ -f outputs/logs/production/DONE ] || [ -f outputs/logs/production/FAILED ]; do sleep 60; done
[ -f outputs/logs/production/FAILED ] && { echo "production failed"; exit 1; }
for ev in $("$PYG" -c "import json; print(' '.join(json.load(open('configs/events.json'))))"); do
  "$PYG" -I scripts/03_package.py "$ev" || exit 1
  HF_HUB_DISABLE_PROGRESS_BARS=1 "$PYH" -c "
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='yanxingjianken/blocking-tracks', repo_type='dataset', folder_path='pkg/$ev', path_in_repo='$ev',
                      commit_message='$ev: full event')
print('uploaded $ev')" || exit 1
done
touch outputs/logs/PACKAGED
