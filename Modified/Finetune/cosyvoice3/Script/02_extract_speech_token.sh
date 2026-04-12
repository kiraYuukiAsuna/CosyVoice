#!/bin/bash
# Split from run.sh stage 2: Extract speech tokens
# Usage: ./02_extract_speech_token.sh <role_name>

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate cosyvoice

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"

. ./path.sh || exit 1

ROLE_NAME=${1:-}
PRETRAINED_DIR="$ROOT_DIR/../../../pretrained_models/Fun-CosyVoice3-0.5B"
ONNX_ST="$PRETRAINED_DIR/speech_tokenizer_v3.onnx"

if [[ -z "$ROLE_NAME" ]]; then
  echo "Usage: $0 <role_name>" >&2
  exit 2
fi

if [[ ! -d "$PRETRAINED_DIR" ]]; then
  echo "Pretrained model dir not found: $PRETRAINED_DIR" >&2
  exit 3
fi

if [[ ! -f "$ONNX_ST" ]]; then
  echo "speech_tokenizer_v3.onnx not found: $ONNX_ST" >&2
  exit 4
fi

echo "Extract speech tokens for role: $ROLE_NAME"
for split in train test; do
  data_dir="data/${ROLE_NAME}_${split}"
  if [[ ! -d "$data_dir" ]]; then
    echo "Prepared data not found: $data_dir. Run 00_prepare_data.sh first." >&2
    exit 5
  fi
  echo "  - extracting speech tokens for ${ROLE_NAME}_${split}"
  python ../../../tools/extract_speech_token.py \
    --dir "$data_dir" \
    --onnx_path "$ONNX_ST"
done

echo "Done: utt2speech_token.pt under data/${ROLE_NAME}_{train,test}"
