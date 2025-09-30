#!/bin/bash
# Split from run.sh stage 0: Data preparation
# Usage: ./00_prepare_data.sh <role_name>
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

DATASET_DIR="$ROOT_DIR/Dataset"
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset directory not found: $DATASET_DIR" >&2
  exit 3
fi

echo "Data preparation for role: $ROLE_NAME"
for x in "${ROLE_NAME}"; do
  echo "  - preparing $x"
  mkdir -p "data/$x"
  python local/prepare_data.py --src_dir "$DATASET_DIR/$x" --des_dir "data/$x"
done

echo "Done: data prepared under $(pwd)/data/${ROLE_NAME}_*"