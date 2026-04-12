#!/bin/bash
# Split from run.sh stage 0: Data preparation
# Usage: ./00_prepare_data.sh <role_name> [cv_ratio]

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate cosyvoice

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"

. ./path.sh || exit 1

ROLE_NAME=${1:-}
CV_RATIO=${2:-${CV_RATIO:-0.05}}
DATASET_DIR="$ROOT_DIR/Dataset"
TMP_BASE="$ROOT_DIR/.split_cache/$ROLE_NAME"
TRAIN_SRC="$TMP_BASE/train"
CV_SRC="$TMP_BASE/test"
PREPARE_ARGS=(--instruct "You are a helpful assistant.<|endofprompt|>")

if [[ -z "$ROLE_NAME" ]]; then
  echo "Usage: $0 <role_name> [cv_ratio]" >&2
  exit 2
fi

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset directory not found: $DATASET_DIR" >&2
  exit 3
fi

SRC_DIR="$DATASET_DIR/$ROLE_NAME"
if [[ ! -d "$SRC_DIR" ]]; then
  echo "Role dataset directory not found: $SRC_DIR" >&2
  exit 4
fi

mapfile -t WAV_FILES < <(find "$SRC_DIR" -maxdepth 1 -type f -name '*.wav' | sort)
TOTAL=${#WAV_FILES[@]}
if [[ $TOTAL -eq 0 ]]; then
  echo "No wav files found under: $SRC_DIR" >&2
  exit 5
fi

CV_COUNT=$(awk -v total="$TOTAL" -v ratio="$CV_RATIO" 'BEGIN {
  c = int(total * ratio + 0.5);
  if (ratio > 0 && c < 1 && total > 1) c = 1;
  if (c >= total && total > 1) c = total - 1;
  if (total == 1) c = 0;
  print c;
}')
TRAIN_COUNT=$((TOTAL - CV_COUNT))

rm -rf "$TMP_BASE"
mkdir -p "$TRAIN_SRC" "$CV_SRC"

TMP_HASH_LIST=$(mktemp)
trap 'rm -f "$TMP_HASH_LIST"' EXIT

for wav in "${WAV_FILES[@]}"; do
  base=$(basename "$wav" .wav)
  hash=$(printf '%s' "$base" | md5sum | awk '{print $1}')
  printf '%s\t%s\n' "$hash" "$base" >> "$TMP_HASH_LIST"
done

mapfile -t SORTED_BASES < <(sort "$TMP_HASH_LIST" | awk '{print $2}')

for idx in "${!SORTED_BASES[@]}"; do
  base="${SORTED_BASES[$idx]}"
  if [[ $idx -lt $CV_COUNT ]]; then
    target_dir="$CV_SRC"
  else
    target_dir="$TRAIN_SRC"
  fi
  ln -sf "$SRC_DIR/$base.wav" "$target_dir/$base.wav"
  if [[ -f "$SRC_DIR/$base.normalized.txt" ]]; then
    ln -sf "$SRC_DIR/$base.normalized.txt" "$target_dir/$base.normalized.txt"
  fi
done

echo "Data preparation for role: $ROLE_NAME"
echo "  - total wavs: $TOTAL"
echo "  - train split: $TRAIN_COUNT"
echo "  - cv split: $CV_COUNT"

mkdir -p "data/${ROLE_NAME}_train" "data/${ROLE_NAME}_test"
python local/prepare_data.py \
  --src_dir "$TRAIN_SRC" \
  --des_dir "data/${ROLE_NAME}_train" \
  "${PREPARE_ARGS[@]}"
python local/prepare_data.py \
  --src_dir "$CV_SRC" \
  --des_dir "data/${ROLE_NAME}_test" \
  "${PREPARE_ARGS[@]}"

echo "Done: data prepared under $(pwd)/data/${ROLE_NAME}_{train,test}"
