#!/bin/bash

# Simple wrapper around accelerate that consumes a YAML config without extra CLI args.
# Usage:
#   bash scripts/finetune_with_accelerate_config_simple.sh <num_gpus> <config_file> [accelerate-launch-args...]

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <num_gpus> <config_file> [accelerate-launch-args...]"
    exit 1
fi

NUM_GPUS="$1"
shift
CONFIG_FILE="$1"
shift

echo "Launching accelerate with $NUM_GPUS process(es) and config: $CONFIG_FILE"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    --use_deepspeed \
    "$@" \
    -- \
    open_instruct/finetune.py \
    "$CONFIG_FILE"
