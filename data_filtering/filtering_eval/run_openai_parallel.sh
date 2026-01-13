#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES="${SAMPLES:-$ROOT/data/gold_dataset.jsonl}"
PRED_DIR="${PRED_DIR:-$ROOT/predictions}"

OPENAI_MODELS="${OPENAI_MODELS:-gpt-5-mini,gpt-5.2}"
OPENAI_MAX_WORKERS="${OPENAI_MAX_WORKERS:-35}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set" >&2
  exit 1
fi

mkdir -p "$PRED_DIR"

pids=()
IFS=',' read -r -a OPENAI_LIST <<< "$OPENAI_MODELS"
for model in "${OPENAI_LIST[@]}"; do
  model_trimmed="$(echo "$model" | xargs)"
  if [[ -n "$model_trimmed" ]]; then
    python "$ROOT/run_openai_eval.py" \
      --samples "$SAMPLES" \
      --model "$model_trimmed" \
      --out "$PRED_DIR/preds_${model_trimmed}.jsonl" \
      --parallel --max-workers "$OPENAI_MAX_WORKERS" &
    pids+=("$!")
  fi
done

if [[ ${#pids[@]} -eq 0 ]]; then
  echo "No models configured. Set OPENAI_MODELS." >&2
  exit 1
fi

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done
if [[ $fail -ne 0 ]]; then
  echo "One or more OpenAI runs failed." >&2
  exit 1
fi
