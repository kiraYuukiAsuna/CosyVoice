#!/bin/bash
# Split from run.sh stage 3: Make parquet only
# Usage: ./03_make_parquet.sh <role_name>

eval "$(conda shell.bash hook)"
conda activate cosyvoice

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"

. ./path.sh || exit 1

ROLE_NAME=${1:-}
if [[ -z "$ROLE_NAME" ]]; then
  echo "Usage: $0 <role_name>" >&2
  exit 2
fi

DATASET_DIR="$ROOT_DIR/Dataset"
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset directory not found: $DATASET_DIR" >&2
  exit 3
fi

for x in "${ROLE_NAME}"; do
  echo "  - making parquet for $x"
  mkdir -p "data/$x/parquet"
  tools/make_parquet_list.py --num_utts_per_parquet 1000 \
    --num_processes 10 \
    --src_dir "data/$x" \
    --des_dir "data/$x/parquet"
done

echo "Done: parquet lists created under data/${ROLE_NAME}_*/parquet"