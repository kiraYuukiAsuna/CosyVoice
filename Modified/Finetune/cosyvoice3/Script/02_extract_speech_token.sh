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
ONNX_CANDIDATES=(
  "$PRETRAINED_DIR/speech_tokenizer_v3.onnx"
  "$PRETRAINED_DIR/speech_tokenizer_v2.onnx"
)
ONNX_ST=""
TOKEN_EXTRACTOR="../../../tools/extract_speech_token_S3Tokenizer.py"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F "," '{print NF}')
TOKEN_BATCH_SIZE="${TOKEN_BATCH_SIZE:-32}"
TOKEN_NUM_THREAD="${TOKEN_NUM_THREAD:-8}"
TOKEN_DEVICE="${TOKEN_DEVICE:-cuda}"

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

if [[ ! -f "$TOKEN_EXTRACTOR" ]]; then
  echo "S3Tokenizer extractor not found: $TOKEN_EXTRACTOR" >&2
  exit 6
fi

echo "Extract speech tokens for role: $ROLE_NAME"
echo "  - using speech tokenizer: $ONNX_ST"
echo "  - using extractor: $TOKEN_EXTRACTOR"
echo "  - device: $TOKEN_DEVICE, gpus: $NUM_GPUS, batch_size: $TOKEN_BATCH_SIZE, num_thread: $TOKEN_NUM_THREAD"
for split in train test; do
  data_dir="data/${ROLE_NAME}_${split}"
  if [[ ! -d "$data_dir" ]]; then
    echo "Prepared data not found: $data_dir. Run 00_prepare_data.sh first." >&2
    exit 5
  fi
  echo "  - extracting speech tokens for ${ROLE_NAME}_${split}"
  if [[ "$TOKEN_DEVICE" == cuda* && "$NUM_GPUS" -gt 1 ]]; then
    torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" \
      --rdzv_id=1986 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
      "$TOKEN_EXTRACTOR" \
      --dir "$data_dir" \
      --onnx_path "$ONNX_ST" \
      --device "$TOKEN_DEVICE" \
      --batch_size "$TOKEN_BATCH_SIZE" \
      --num_thread "$TOKEN_NUM_THREAD"
  else
    python "$TOKEN_EXTRACTOR" \
      --dir "$data_dir" \
      --onnx_path "$ONNX_ST" \
      --device "$TOKEN_DEVICE" \
      --batch_size "$TOKEN_BATCH_SIZE" \
      --num_thread "$TOKEN_NUM_THREAD"
  fi
done

echo "Done: utt2speech_token.pt under data/${ROLE_NAME}_{train,test}"
