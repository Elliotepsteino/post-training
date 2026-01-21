#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRED_DIR="${PRED_DIR:-$ROOT/predictions}"
GOLD_PATH="${GOLD_PATH:-$ROOT/data/gold_dataset_dev.jsonl}"
OUT_JSON="${OUT_JSON:-$ROOT/results/grounding_summary.json}"
OUT_PLOT="${OUT_PLOT:-$ROOT/results/grounding_impact.pdf}"
UPDATED_SUFFIX="${UPDATED_SUFFIX:-_grounded_updated.jsonl}"
NONCONSERVATIVE_SUFFIX="${NONCONSERVATIVE_SUFFIX:-_nonconservative.jsonl}"

MODELS="${MODELS:-gpt-5-mini gpt-5.2 gemini-3-flash-preview gemini-3-pro-preview}"
MAX_WORKERS="${MAX_WORKERS:-8}"

pids=()
for m in $MODELS; do
  python "$ROOT/update_estimates_with_evidence.py" \
    --preds "$PRED_DIR/preds_${m}_grounded.jsonl" \
    --out "$PRED_DIR/preds_${m}${UPDATED_SUFFIX}" \
    --max-workers "$MAX_WORKERS" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "Classifying non-conservative cases..."
pids=()
for m in $MODELS; do
  python "$ROOT/classify_nonconservative_cases.py" \
    --preds "$PRED_DIR/preds_${m}${UPDATED_SUFFIX}" \
    --gold-path "$GOLD_PATH" \
    --out "$PRED_DIR/preds_${m}${NONCONSERVATIVE_SUFFIX}" &
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
