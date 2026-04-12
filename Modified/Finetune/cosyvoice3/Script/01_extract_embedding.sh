#!/bin/bash
# Split from run.sh stage 1: Extract speaker embeddings
# Usage: ./01_extract_embedding.sh <role_name>

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate cosyvoice

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"

. ./path.sh || exit 1

ROLE_NAME=${1:-}
PRETRAINED_DIR="$ROOT_DIR/../../../pretrained_models/Fun-CosyVoice3-0.5B"
ONNX_CAMPPLUS="$PRETRAINED_DIR/campplus.onnx"

if [[ -z "$ROLE_NAME" ]]; then
  echo "Usage: $0 <role_name>" >&2
  exit 2
fi

if [[ ! -d "$PRETRAINED_DIR" ]]; then
  echo "Pretrained model dir not found: $PRETRAINED_DIR" >&2
  exit 3
fi

if [[ ! -f "$ONNX_CAMPPLUS" ]]; then
  echo "campplus.onnx not found: $ONNX_CAMPPLUS" >&2
  exit 4
fi

echo "Extract speaker embeddings for role: $ROLE_NAME"
for split in train test; do
  data_dir="data/${ROLE_NAME}_${split}"
  if [[ ! -d "$data_dir" ]]; then
    echo "Prepared data not found: $data_dir. Run 00_prepare_data.sh first." >&2
    exit 5
  fi
  echo "  - extracting embeddings for ${ROLE_NAME}_${split}"
  python ../../../tools/extract_embedding.py \
    --dir "$data_dir" \
    --onnx_path "$ONNX_CAMPPLUS"
done

echo "Done: spk2embedding.pt and utt2embedding.pt under data/${ROLE_NAME}_{train,test}"
