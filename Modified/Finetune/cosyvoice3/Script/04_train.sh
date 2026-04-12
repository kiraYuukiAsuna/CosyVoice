#!/bin/bash
# Split from run.sh stage 5: Train only
# Usage: ./04_train.sh <role_name>

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate cosyvoice

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"

. ./path.sh || exit 1

ROLE_NAME=${1:-}
PRETRAINED_DIR="$ROOT_DIR/../../../pretrained_models/Fun-CosyVoice3-0.5B"
TRAIN_LIST="data/${ROLE_NAME}_train/parquet/data.list"
CV_LIST="data/${ROLE_NAME}_test/parquet/data.list"

MODELS=(llm flow)
CONFIGS=(conf/cosyvoice_llm.yaml conf/cosyvoice_flow.yaml)

if [[ -z "$ROLE_NAME" ]]; then
  echo "Usage: $0 <role_name>" >&2
  exit 2
fi

if [[ ! -d "$PRETRAINED_DIR" ]]; then
  echo "Pretrained model dir not found: $PRETRAINED_DIR" >&2
  exit 3
fi

if [[ ! -f "$TRAIN_LIST" ]]; then
  echo "Missing data list: $TRAIN_LIST. Run 03_make_parquet.sh first." >&2
  exit 4
fi

if [[ ! -f "$CV_LIST" ]]; then
  echo "Missing data list: $CV_LIST. Run 03_make_parquet.sh first." >&2
  exit 5
fi

cat "$TRAIN_LIST" > data/train.data.list
cat "$CV_LIST" > data/test.data.list

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=8
prefetch=100
train_engine=torch_ddp

for idx in "${!MODELS[@]}"; do
  model="${MODELS[$idx]}"
  config_file="${CONFIGS[$idx]}"
  echo "Start training $model for role $ROLE_NAME"
  torchrun --nnodes=1 --nproc_per_node="$num_gpus" \
      --rdzv_id="$job_id" --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
    ../../../cosyvoice/bin/train.py \
    --train_engine "$train_engine" \
    --config "$config_file" \
    --train_data data/train.data.list \
    --cv_data data/test.data.list \
    --qwen_pretrain_path "$PRETRAINED_DIR/CosyVoice-BlankEN" \
    --onnx_path "$PRETRAINED_DIR" \
    --model "$model" \
    --checkpoint "$PRETRAINED_DIR/$model.pt" \
    --model_dir "$(pwd)/exp/cosyvoice3/$model/$train_engine/$ROLE_NAME" \
    --tensorboard_dir "$(pwd)/tensorboard/cosyvoice3/$model/$train_engine/$ROLE_NAME" \
    --ddp.dist_backend "$dist_backend" \
    --num_workers "$num_workers" \
    --prefetch "$prefetch" \
    --use_amp \
    --deepspeed_config ./conf/ds_stage2.json \
    --deepspeed.save_states model+optimizer
  echo "Training finished for $model"
done

echo "Done: training artifacts under exp/cosyvoice3/{llm,flow}/$train_engine/$ROLE_NAME"
