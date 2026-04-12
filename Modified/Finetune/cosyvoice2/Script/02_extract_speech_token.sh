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
PRETRAINED_DIR="$ROOT_DIR/../../../pretrained_models/CosyVoice2-0.5B"
ONNX_CANDIDATES=(
  "$PRETRAINED_DIR/speech_tokenizer_v3.onnx"
  "$PRETRAINED_DIR/speech_tokenizer_v2.onnx"
)
ONNX_ST=""

if [[ -z "$ROLE_NAME" ]]; then
  echo "Usage: $0 <role_name>" >&2
  exit 2
fi

if [[ ! -d "$PRETRAINED_DIR" ]]; then
  echo "Pretrained model dir not found: $PRETRAINED_DIR" >&2
  exit 3
fi

for candidate in "${ONNX_CANDIDATES[@]}"; do
  if [[ -f "$candidate" ]]; then
    ONNX_ST="$candidate"
    break
  fi
done

if [[ ! -f "$ONNX_ST" ]]; then
  echo "speech tokenizer onnx not found under: $PRETRAINED_DIR" >&2
  printf 'checked:\n' >&2
  printf '  %s\n' "${ONNX_CANDIDATES[@]}" >&2
  exit 4
fi

echo "Extract speech tokens for role: $ROLE_NAME"
echo "  - using speech tokenizer: $ONNX_ST"
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
