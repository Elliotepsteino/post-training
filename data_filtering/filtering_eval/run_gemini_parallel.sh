#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/epsteine/post-training/data_filtering/filtering_eval"
SAMPLES="$ROOT/data/gold_dataset_dev.jsonl"
PRED_DIR="$ROOT/predictions"

echo "Starting Gemini runs..."
python "$ROOT/run_gemini_eval.py" --model gemini-3-pro-preview \
  --samples "$SAMPLES" \
  --out "$PRED_DIR/preds_gemini-3-pro-preview.jsonl" \
  --parallel --max-workers 10 &
pid_pro=$!
echo "gemini-3-pro-preview PID: $pid_pro"

python "$ROOT/run_gemini_eval.py" --model gemini-3-flash-preview \
  --samples "$SAMPLES" \
  --out "$PRED_DIR/preds_gemini-3-flash-preview.jsonl" \
  --parallel --max-workers 10 &
pid_flash=$!
echo "gemini-3-flash-preview PID: $pid_flash"

completed=0
while [[ $completed -lt 2 ]]; do
  if wait -n "$pid_pro" "$pid_flash"; then
    completed=$((completed + 1))
    if ! kill -0 "$pid_pro" 2>/dev/null; then
      echo "gemini-3-pro-preview finished."
    fi
    if ! kill -0 "$pid_flash" 2>/dev/null; then
      echo "gemini-3-flash-preview finished."
    fi
  else
    echo "A Gemini run failed. See output above."
    exit 1
  fi
done
echo "All Gemini runs completed."
