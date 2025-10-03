#!/bin/bash
# Split from run.sh stage 4: Train only
# Usage: ./04_train.sh <role_name>

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

PRETRAINED_DIR="$ROOT_DIR/../../../pretrained_models/CosyVoice2-0.5B"
if [[ ! -d "$PRETRAINED_DIR" ]]; then
  echo "Pretrained model dir not found: $PRETRAINED_DIR" >&2
  exit 3
fi

# Assemble data lists from parquet step
if [[ ! -f "data/${ROLE_NAME}/parquet/data.list" ]]; then
  echo "Missing data list: data/${ROLE_NAME}/parquet/data.list. Run 03_make_parquet.sh first." >&2
  exit 4
fi

cat "data/${ROLE_NAME}/parquet/data.list" > data/train.data.list

# Train settings
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=8
prefetch=100
train_engine=torch_ddp


echo "Start training llm for role $ROLE_NAME"
torchrun --nnodes=1 --nproc_per_node=$num_gpus \
    --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
cosyvoice/bin/train.py \
--train_engine $train_engine \
--config conf/cosyvoice2_llm.yaml \
--train_data data/train.data.list \
--cv_data data/train.data.list \
--qwen_pretrain_path "$PRETRAINED_DIR/CosyVoice-BlankEN" \
--model llm \
--checkpoint "$PRETRAINED_DIR/llm.pt" \
--model_dir "$(pwd)/exp/cosyvoice2/llm/$train_engine/${ROLE_NAME}" \
--tensorboard_dir "$(pwd)/tensorboard/cosyvoice2/llm/$train_engine/${ROLE_NAME}" \
--ddp.dist_backend $dist_backend \
--num_workers ${num_workers} \
--prefetch ${prefetch} \
--pin_memory \
--use_amp \
--deepspeed_config ./conf/ds_stage2.json \
--deepspeed.save_states model+optimizer
echo "Training finished for llm"


echo "Start training flow for role $ROLE_NAME"
torchrun --nnodes=1 --nproc_per_node=$num_gpus \
    --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
cosyvoice/bin/train.py \
--train_engine $train_engine \
--config conf/cosyvoice2_flow.yaml \
--train_data data/train.data.list \
--cv_data data/train.data.list \
--qwen_pretrain_path "$PRETRAINED_DIR/CosyVoice-BlankEN" \
--model flow \
--checkpoint "$PRETRAINED_DIR/flow.pt" \
--model_dir "$(pwd)/exp/cosyvoice2/flow/$train_engine/${ROLE_NAME}" \
--tensorboard_dir "$(pwd)/tensorboard/cosyvoice2/flow/$train_engine/${ROLE_NAME}" \
--ddp.dist_backend $dist_backend \
--num_workers ${num_workers} \
--prefetch ${prefetch} \
--pin_memory \
--use_amp \
--deepspeed_config ./conf/ds_stage2.json \
--deepspeed.save_states model+optimizer
echo "Training finished for flow"


echo "Done: training artifacts under exp/cosyvoice2/flow/$train_engine/${ROLE_NAME}"
