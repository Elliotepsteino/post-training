#!/bin/bash

# Debug-friendly wrapper around accelerate launch with a total batch size of 1.
# Usage:
#   sh scripts/finetune_with_accelerate_debug.sh [CONFIG_FILE] [--extra-accelerate-args ...]

CONFIG_FILE="${1:-configs/train_configs/tulu3/tulu3_sft_year2007_debug.yaml}"
if [ "$#" -gt 0 ]; then
  shift
fi

NUM_GPUS=1
PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUM=1
TOTAL_BATCH_SIZE=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE * GRAD_ACCUM))

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "Debug finetune using $NUM_GPUS GPU, per-device batch $PER_DEVICE_BATCH_SIZE, grad accumulation $GRAD_ACCUM (total batch size $TOTAL_BATCH_SIZE)"

accelerate launch \
  --mixed_precision bf16 \
  --num_machines 1 \
  --num_processes $NUM_GPUS \
  --use_deepspeed \
  "$@" \
  -- \
  open_instruct/finetune.py \
  "$CONFIG_FILE"
