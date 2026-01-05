#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLMES_DIR="$REPO_DIR/olmes"
MODEL_PATH="$REPO_DIR/model_weights/Qwen3-4B-Base"
OUTPUT_ROOT="$OLMES_DIR/workspace_tulu3_dev_limit8"
LOG_ROOT="$OLMES_DIR/logs/tulu3_dev_limit8"
TASKS=(
  "gsm8k::tulu"
  "drop::llama3"
  "minerva_math::tulu"
  "codex_humaneval::tulu"
  "codex_humanevalplus::tulu"
  "ifeval::tulu"
  "popqa::tulu"
  "mmlu:mc::tulu"
  "alpaca_eval_v2::tulu"
  "bbh:cot-v1::tulu"
  "truthfulqa::tulu"
)
GPU_IDS=(0 1 2 3 4)
NUM_GPUS=${#GPU_IDS[@]}
TOTAL_TASKS=${#TASKS[@]}

COUNTER_FILE="$(mktemp)"
cleanup() {
  rm -f "$COUNTER_FILE"
}
trap cleanup EXIT
echo 0 > "$COUNTER_FILE"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

next_task_index() {
  (
    flock 200
    local current
    current=$(<"$COUNTER_FILE")
    if (( current >= TOTAL_TASKS )); then
      echo -1
    else
      echo $((current))
      echo $((current + 1)) > "$COUNTER_FILE"
    fi
  ) 200<>"$COUNTER_FILE"
}

run_worker() {
  local worker_idx=$1
  local gpu_id=${GPU_IDS[$worker_idx]}

  while true; do
    local task_idx
    task_idx=$(next_task_index)
    if (( task_idx < 0 )); then
      break
    fi
    local task="${TASKS[$task_idx]}"
    local safe_task
    safe_task=$(echo "$task" | tr '/:{}' '_')
    local out_dir="$OUTPUT_ROOT/$safe_task"
    local log_file="$LOG_ROOT/${safe_task}_gpu${gpu_id}.log"

    mkdir -p "$out_dir"
    echo "[$(date -Is)] [GPU $gpu_id] Starting $task" | tee -a "$log_file"
    (
      export CUDA_VISIBLE_DEVICES="$gpu_id"
      uv run olmes \
        --model "$MODEL_PATH" \
        --task "$task" \
        --output-dir "$out_dir" \
        --limit 8 \
        --num-workers 1 \
        --gpus 1 \
        --save-raw-requests true
    ) >>"$log_file" 2>&1
    echo "[$(date -Is)] [GPU $gpu_id] Finished $task" | tee -a "$log_file"
  done
}

pids=()
for ((worker=0; worker<NUM_GPUS; ++worker)); do
  run_worker "$worker" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[$(date -Is)] All TÜLU-3 dev tasks completed. Outputs in $OUTPUT_ROOT"
