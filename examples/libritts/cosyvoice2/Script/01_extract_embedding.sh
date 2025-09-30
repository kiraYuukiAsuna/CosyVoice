#!/bin/bash
# Split from run.sh stage 1: Extract speaker embeddings
# Usage: ./01_extract_embedding.sh <role_name>

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
PRETRAINED_DIR="$ROOT_DIR/../../../pretrained_models/CosyVoice2-0.5B"

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset directory not found: $DATASET_DIR" >&2
  exit 3
fi
if [[ ! -d "$PRETRAINED_DIR" ]]; then
  echo "Pretrained model dir not found: $PRETRAINED_DIR" >&2
  exit 4
fi

ONNX_CAMPPLUS="$PRETRAINED_DIR/campplus.onnx"
if [[ ! -f "$ONNX_CAMPPLUS" ]]; then
  echo "campplus.onnx not found: $ONNX_CAMPPLUS" >&2
  exit 5
fi

echo "Extract speaker embeddings for role: $ROLE_NAME"
for x in "${ROLE_NAME}"; do
  echo "  - extracting embeddings for $x"
  tools/extract_embedding.py --dir "data/$x" \
    --onnx_path "$ONNX_CAMPPLUS"
done

echo "Done: spk2embedding.pt and utt2embedding.pt under data/${ROLE_NAME}_*"