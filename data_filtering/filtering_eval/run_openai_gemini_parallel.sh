#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES="${SAMPLES:-$ROOT/data/gold_dataset_dev_categorized.jsonl}"
PRED_DIR="${PRED_DIR:-$ROOT/predictions}"
OUT_JSON="${OUT_JSON:-$ROOT/results/summary.json}"
OUT_TEX="${OUT_TEX:-$ROOT/results/filtering_eval_table.tex}"

OPENAI_MODELS="${OPENAI_MODELS-gpt-5-mini}"
OPENAI_MAX_WORKERS="${OPENAI_MAX_WORKERS:-20}"
OPENAI_PARALLEL="${OPENAI_PARALLEL:-0}"
OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-60}"
OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-3}"
GEMINI_MODELS="${GEMINI_MODELS-gemini-3-flash-preview}"
GEMINI_MAX_WORKERS="${GEMINI_MAX_WORKERS:-20}"
NUM_SAMPLES="${NUM_SAMPLES:-2}"

if [[ -n "$OPENAI_MODELS" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set" >&2
  exit 1
fi
if [[ -n "$GEMINI_MODELS" && -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "GEMINI_API_KEY or GOOGLE_API_KEY is not set" >&2
  exit 1
fi

mkdir -p "$PRED_DIR"

pids=()

IFS=',' read -r -a OPENAI_LIST <<< "$OPENAI_MODELS"
for model in "${OPENAI_LIST[@]}"; do
  model_trimmed="$(echo "$model" | xargs)"
  if [[ -n "$model_trimmed" ]]; then
    openai_args=(
      "$ROOT/run_openai_eval.py"
      --samples "$SAMPLES"
      --model "$model_trimmed"
      --out "$PRED_DIR/preds_${model_trimmed}.jsonl"
      --num-samples "$NUM_SAMPLES"
      --timeout "$OPENAI_TIMEOUT"
      --max-retries "$OPENAI_MAX_RETRIES"
    )
    if [[ "$OPENAI_PARALLEL" == "1" ]]; then
      openai_args+=(--parallel --max-workers "$OPENAI_MAX_WORKERS")
    fi
    python -u "${openai_args[@]}" &
    pids+=("$!")
  fi
done

IFS=',' read -r -a GEMINI_LIST <<< "$GEMINI_MODELS"
for model in "${GEMINI_LIST[@]}"; do
  model_trimmed="$(echo "$model" | xargs)"
  if [[ -n "$model_trimmed" ]]; then
    python -u "$ROOT/run_gemini_eval.py" \
      --samples "$SAMPLES" \
      --model "$model_trimmed" \
      --out "$PRED_DIR/preds_${model_trimmed}.jsonl" \
      --parallel --max-workers "$GEMINI_MAX_WORKERS" \
      --num-samples "$NUM_SAMPLES" &
    pids+=("$!")
  fi
done

if [[ ${#pids[@]} -eq 0 ]]; then
  echo "No models configured. Set OPENAI_MODELS and/or GEMINI_MODELS." >&2
  exit 1
fi

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done
if [[ $fail -ne 0 ]]; then
  echo "One or more eval runs failed." >&2
  exit 1
fi

python "$ROOT/score_predictions.py" \
  --gold-path "$SAMPLES" \
  --gold-field gold_year \
  --pred-dir "$PRED_DIR" \
  --out-json "$OUT_JSON" \
  --out-tex "$OUT_TEX"
