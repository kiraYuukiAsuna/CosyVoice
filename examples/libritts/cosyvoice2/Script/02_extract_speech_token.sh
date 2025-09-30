#!/bin/bash
# Split from run.sh stage 2: Extract speech tokens
# Usage: ./02_extract_speech_token.sh <role_name>

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

ONNX_ST="$PRETRAINED_DIR/speech_tokenizer_v2.onnx"
if [[ ! -f "$ONNX_ST" ]]; then
  echo "speech_tokenizer_v2.onnx not found: $ONNX_ST" >&2
  exit 5
fi

echo "Extract speech tokens for role: $ROLE_NAME"
for x in "${ROLE_NAME}"; do
  echo "  - extracting speech tokens for $x"
  tools/extract_speech_token.py --dir "data/$x" \
    --onnx_path "$ONNX_ST"
done

echo "Done: utt2speech_token.pt under data/${ROLE_NAME}_*"