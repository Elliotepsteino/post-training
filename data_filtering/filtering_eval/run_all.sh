#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/epsteine/post-training/data_filtering/filtering_eval"
SAMPLES="$ROOT/data/samples.jsonl"
PRED_DIR="$ROOT/predictions"
OUT_TEX="$ROOT/results/filtering_eval_table.tex"
SAMPLE_SIZE="${SAMPLE_SIZE:-50}"

MODE="${1:-batch}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set" >&2
  exit 1
fi

python "$ROOT/sample_questions.py" --num-samples "$SAMPLE_SIZE" --out "$SAMPLES"

if [[ "$MODE" == "batch" ]]; then
  echo "Submitting batch jobs..."
  python "$ROOT/run_openai_eval.py" --batch --model gpt-5-mini --out "$PRED_DIR/preds_gpt-5-mini.jsonl" &
  python "$ROOT/run_openai_eval.py" --batch --model gpt-5.2 --out "$PRED_DIR/preds_gpt-5.2.jsonl" &
  python "$ROOT/run_openai_eval.py" --batch --model gpt-5.2-pro --out "$PRED_DIR/preds_gpt-5.2-pro.jsonl" &
  wait
  echo "Batch jobs submitted. Fetch with:"
  echo "  $ROOT/run_all.sh fetch <BATCH_ID_MINI> <BATCH_ID_52> <BATCH_ID_52PRO>"
  exit 0
fi

if [[ "$MODE" == "batch-wait" ]]; then
  python "$ROOT/run_openai_eval.py" --batch --wait --model gpt-5-mini --out "$PRED_DIR/preds_gpt-5-mini.jsonl" &
  python "$ROOT/run_openai_eval.py" --batch --wait --model gpt-5.2 --out "$PRED_DIR/preds_gpt-5.2.jsonl" &
  python "$ROOT/run_openai_eval.py" --batch --wait --model gpt-5.2-pro --out "$PRED_DIR/preds_gpt-5.2-pro.jsonl" &
  wait
  if [[ -n "${GEMINI_MODELS:-}" ]]; then
    IFS=',' read -r -a GEMINI_LIST <<< "$GEMINI_MODELS"
    for model in "${GEMINI_LIST[@]}"; do
      model_trimmed="$(echo "$model" | xargs)"
      if [[ -n "$model_trimmed" ]]; then
        python "$ROOT/run_gemini_eval.py" --model "$model_trimmed" --out "$PRED_DIR/preds_${model_trimmed}.jsonl" &
      fi
    done
    wait
  fi
  python "$ROOT/score_predictions.py" --gold-from-model gpt-5.2-pro --out-tex "$OUT_TEX"
  exit 0
fi

if [[ "$MODE" == "fetch" ]]; then
  if [[ $# -ne 4 ]]; then
    echo "Usage: $ROOT/run_all.sh fetch <BATCH_ID_MINI> <BATCH_ID_52> <BATCH_ID_52PRO>" >&2
    exit 1
  fi
  python "$ROOT/run_openai_eval.py" --fetch-batch "$2" --model gpt-5-mini --out "$PRED_DIR/preds_gpt-5-mini.jsonl"
  python "$ROOT/run_openai_eval.py" --fetch-batch "$3" --model gpt-5.2 --out "$PRED_DIR/preds_gpt-5.2.jsonl"
  python "$ROOT/run_openai_eval.py" --fetch-batch "$4" --model gpt-5.2-pro --out "$PRED_DIR/preds_gpt-5.2-pro.jsonl"
  python "$ROOT/score_predictions.py" --gold-from-model gpt-5.2-pro --out-tex "$OUT_TEX"
  exit 0
fi

if [[ "$MODE" == "live" ]]; then
  python "$ROOT/run_openai_eval.py" --model gpt-5-mini --out "$PRED_DIR/preds_gpt-5-mini.jsonl"
  python "$ROOT/run_openai_eval.py" --model gpt-5.2 --out "$PRED_DIR/preds_gpt-5.2.jsonl"
  python "$ROOT/run_openai_eval.py" --model gpt-5.2-pro --out "$PRED_DIR/preds_gpt-5.2-pro.jsonl"
  if [[ -n "${GEMINI_MODELS:-}" ]]; then
    IFS=',' read -r -a GEMINI_LIST <<< "$GEMINI_MODELS"
    for model in "${GEMINI_LIST[@]}"; do
      model_trimmed="$(echo "$model" | xargs)"
      if [[ -n "$model_trimmed" ]]; then
        python "$ROOT/run_gemini_eval.py" --model "$model_trimmed" --out "$PRED_DIR/preds_${model_trimmed}.jsonl" &
      fi
    done
    wait
  fi
  python "$ROOT/score_predictions.py" --gold-from-model gpt-5.2-pro --out-tex "$OUT_TEX"
  exit 0
fi

cat <<'USAGE'
Usage:
  run_all.sh batch            # submit batch jobs for all three models
  run_all.sh batch-wait       # submit and block until all batches complete
  run_all.sh fetch <ids...>   # fetch results, then score and write table
  run_all.sh live             # run non-batch calls, then score and write table
Environment:
  SAMPLE_SIZE=50              # sample size for the audit set
  GEMINI_MODELS=...           # comma-separated Gemini model list (optional)
USAGE
exit 1
