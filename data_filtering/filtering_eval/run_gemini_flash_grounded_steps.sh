#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRED_DIR="${PRED_DIR:-$ROOT/predictions}"
GOLD_PATH="${GOLD_PATH:-$ROOT/data/gold_dataset_dev.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"

MODEL="${MODEL:-gemini-3-flash-preview}"
SEARCH_MODEL="${SEARCH_MODEL:-gpt-4o-mini}"
SEARCH_MAX_WORKERS="${SEARCH_MAX_WORKERS:-10}"
UPDATE_MAX_WORKERS="${UPDATE_MAX_WORKERS:-8}"

INPUT_PREDS="${INPUT_PREDS:-$PRED_DIR/preds_${MODEL}.jsonl}"
GROUNDED_PREDS="${GROUNDED_PREDS:-$PRED_DIR/preds_${MODEL}_grounded.jsonl}"
UPDATED_PREDS="${UPDATED_PREDS:-$PRED_DIR/preds_${MODEL}_grounded_updated.jsonl}"
NONCONSERVATIVE_PREDS="${NONCONSERVATIVE_PREDS:-$PRED_DIR/preds_${MODEL}_nonconservative.jsonl}"

OUT_JSON="${OUT_JSON:-$RESULTS_DIR/grounding_summary.json}"
OUT_PLOT="${OUT_PLOT:-$RESULTS_DIR/grounding_impact.pdf}"

python "$ROOT/ground_entities_with_search.py" \
  --preds "$INPUT_PREDS" \
  --out "$GROUNDED_PREDS" \
  --model "$SEARCH_MODEL" \
  --max-workers "$SEARCH_MAX_WORKERS"

python "$ROOT/update_estimates_with_evidence.py" \
  --preds "$GROUNDED_PREDS" \
  --out "$UPDATED_PREDS" \
  --max-workers "$UPDATE_MAX_WORKERS" \
  --aggregation "${AGGREGATION:-max}"

python "$ROOT/classify_nonconservative_cases.py" \
  --preds "$UPDATED_PREDS" \
  --gold-path "$GOLD_PATH" \
  --out "$NONCONSERVATIVE_PREDS"

python "$ROOT/analyze_grounding_impact.py" \
  --pred-dir "$PRED_DIR" \
  --updated-dir "$PRED_DIR" \
  --updated-suffix "_grounded_updated.jsonl" \
  --updated-year-field "${UPDATED_YEAR_FIELD:-updated_year}" \
  --gold-path "$GOLD_PATH" \
  --out-json "$OUT_JSON" \
  --out-plot "$OUT_PLOT"
