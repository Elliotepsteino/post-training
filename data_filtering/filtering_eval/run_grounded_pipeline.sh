#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRED_DIR="${PRED_DIR:-$ROOT/predictions}"
GOLD_PATH="${GOLD_PATH:-$ROOT/data/gold_dataset_dev.jsonl}"
OUT_JSON="${OUT_JSON:-$ROOT/results/grounding_summary.json}"
OUT_PLOT="${OUT_PLOT:-$ROOT/results/grounding_impact.pdf}"

OPENAI_MODELS="${OPENAI_MODELS-gpt-5-mini,gpt-5.2}"
GEMINI_MODELS="${GEMINI_MODELS-gemini-3-flash-preview,gemini-3-pro-preview}"
OPENAI_MAX_WORKERS="${OPENAI_MAX_WORKERS:-10}"
GEMINI_MAX_WORKERS="${GEMINI_MAX_WORKERS:-10}"

SEARCH_MODEL="${SEARCH_MODEL:-gpt-4o-mini}"
SEARCH_MAX_WORKERS="${SEARCH_MAX_WORKERS:-10}"
UPDATE_MAX_WORKERS="${UPDATE_MAX_WORKERS:-8}"

GROUNDED_SUFFIX="${GROUNDED_SUFFIX:-_grounded.jsonl}"
UPDATED_SUFFIX="${UPDATED_SUFFIX:-_grounded_updated.jsonl}"
NONCONSERVATIVE_SUFFIX="${NONCONSERVATIVE_SUFFIX:-_nonconservative.jsonl}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"

./run_openai_gemini_parallel.sh

models=""
if [[ -n "${OPENAI_MODELS-}" ]]; then
  models="${OPENAI_MODELS}"
fi
if [[ -n "${GEMINI_MODELS-}" ]]; then
  if [[ -n "${models}" ]]; then
    models="${models},${GEMINI_MODELS}"
  else
    models="${GEMINI_MODELS}"
  fi
fi

IFS=',' read -r -a MODEL_LIST <<< "$models"
mkdir -p "$PRED_DIR"

echo "Grounding entities with search..."
pids=()
for model in "${MODEL_LIST[@]}"; do
  model_trimmed="$(echo "$model" | xargs)"
  [[ -z "$model_trimmed" ]] && continue
  python "$ROOT/ground_entities_with_search.py" \
    --preds "$PRED_DIR/preds_${model_trimmed}.jsonl" \
    --out "$PRED_DIR/preds_${model_trimmed}${GROUNDED_SUFFIX}" \
    --model "$SEARCH_MODEL" \
    --max-workers "$SEARCH_MAX_WORKERS" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "Updating estimates with evidence..."
pids=()
for model in "${MODEL_LIST[@]}"; do
  model_trimmed="$(echo "$model" | xargs)"
  [[ -z "$model_trimmed" ]] && continue
  python "$ROOT/update_estimates_with_evidence.py" \
    --preds "$PRED_DIR/preds_${model_trimmed}${GROUNDED_SUFFIX}" \
    --out "$PRED_DIR/preds_${model_trimmed}${UPDATED_SUFFIX}" \
    --max-workers "$UPDATE_MAX_WORKERS" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "Classifying non-conservative cases..."
pids=()
for model in "${MODEL_LIST[@]}"; do
  model_trimmed="$(echo "$model" | xargs)"
  [[ -z "$model_trimmed" ]] && continue
  python "$ROOT/classify_nonconservative_cases.py" \
    --preds "$PRED_DIR/preds_${model_trimmed}${UPDATED_SUFFIX}" \
    --gold-path "$GOLD_PATH" \
    --out "$PRED_DIR/preds_${model_trimmed}${NONCONSERVATIVE_SUFFIX}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

python "$ROOT/analyze_grounding_impact.py" \
  --pred-dir "$PRED_DIR" \
  --updated-dir "$PRED_DIR" \
  --updated-suffix "$UPDATED_SUFFIX" \
  --gold-path "$GOLD_PATH" \
  --out-json "$OUT_JSON" \
  --out-plot "$OUT_PLOT"
