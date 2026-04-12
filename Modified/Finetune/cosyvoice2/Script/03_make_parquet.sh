#!/bin/bash
# Split from run.sh stage 3: Make parquet only
# Usage: ./03_make_parquet.sh <role_name>

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate cosyvoice

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"

. ./path.sh || exit 1

ROLE_NAME=${1:-}

if [[ -z "$ROLE_NAME" ]]; then
  echo "Usage: $0 <role_name>" >&2
  exit 2
fi

echo "Make parquet for role: $ROLE_NAME"
for split in train test; do
  data_dir="data/${ROLE_NAME}_${split}"
  parquet_dir="$data_dir/parquet"
  if [[ ! -d "$data_dir" ]]; then
    echo "Prepared data not found: $data_dir. Run previous stages first." >&2
    exit 3
  fi
  echo "  - making parquet for ${ROLE_NAME}_${split}"
  mkdir -p "$parquet_dir"
  python ../../../tools/make_parquet_list.py \
    --num_utts_per_parquet 1000 \
    --num_processes 10 \
    --src_dir "$data_dir" \
    --des_dir "$parquet_dir"
done

echo "Done: parquet lists created under data/${ROLE_NAME}_{train,test}/parquet"
