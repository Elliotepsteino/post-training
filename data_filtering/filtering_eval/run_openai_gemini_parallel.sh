#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES="${SAMPLES:-$ROOT/data/gold_dataset_dev.jsonl}"
PRED_DIR="${PRED_DIR:-$ROOT/predictions}"
OUT_JSON="${OUT_JSON:-$ROOT/results/summary.json}"
OUT_TEX="${OUT_TEX:-$ROOT/results/filtering_eval_table.tex}"

OPENAI_MODELS="${OPENAI_MODELS-gpt-5-mini,gpt-5.2}"
OPENAI_MAX_WORKERS="${OPENAI_MAX_WORKERS:-200}"
GEMINI_MODELS="${GEMINI_MODELS-gemini-3-pro-preview,gemini-3-flash-preview}"
GEMINI_MAX_WORKERS="${GEMINI_MAX_WORKERS:-200}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"

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
    python "$ROOT/run_openai_eval.py" \
      --samples "$SAMPLES" \
      --model "$model_trimmed" \
      --out "$PRED_DIR/preds_${model_trimmed}.jsonl" \
      --parallel --max-workers "$OPENAI_MAX_WORKERS" \
      --num-samples "$NUM_SAMPLES" &
    pids+=("$!")
  fi
done

IFS=',' read -r -a GEMINI_LIST <<< "$GEMINI_MODELS"
for model in "${GEMINI_LIST[@]}"; do
  model_trimmed="$(echo "$model" | xargs)"
  if [[ -n "$model_trimmed" ]]; then
    python "$ROOT/run_gemini_eval.py" \
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
