# Filtering Eval (Year Labels)

This directory contains a small evaluation pipeline to sanity check year labels on a 50‑question sample.

## 1) Sample questions

```bash
python sample_questions.py \
  --num-samples 50 \
  --out /home/epsteine/post-training/data_filtering/filtering_eval/data/samples.jsonl
```

This draws evenly across available year shards (latest session by default) and stores `question` plus metadata.

## 2) Run model predictions

Batch (recommended):

```bash
python run_openai_eval.py --batch --model gpt-5.2-pro \
  --out /home/epsteine/post-training/data_filtering/filtering_eval/predictions/preds_gpt-5.2-pro.jsonl

# later, fetch results
python run_openai_eval.py --fetch-batch <BATCH_ID> --model gpt-5.2-pro \
  --out /home/epsteine/post-training/data_filtering/filtering_eval/predictions/preds_gpt-5.2-pro.jsonl
```

Non‑batch:

```bash
python run_openai_eval.py --model gpt-5-mini \
  --out /home/epsteine/post-training/data_filtering/filtering_eval/predictions/preds_gpt-5-mini.jsonl
```

Repeat for `gpt-5.2` and `gpt-5.2-pro`.

Parallel OpenAI + Gemini on the gold dataset:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...  # or GOOGLE_API_KEY

./run_openai_gemini_parallel.sh
```

Run all four models in parallel with 10 workers per model (OpenAI + Gemini):

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...  # or GOOGLE_API_KEY

OPENAI_MODELS=gpt-5-mini,gpt-5.2 \
GEMINI_MODELS=gemini-3-flash-preview,gemini-3-pro-preview \
OPENAI_MAX_WORKERS=10 \
GEMINI_MAX_WORKERS=10 \
./run_openai_gemini_parallel.sh
```

Parallel OpenAI only (live):

```bash
export OPENAI_API_KEY=...
./run_openai_parallel.sh
```

To fire all samples concurrently for a single model:

```bash
export OPENAI_API_KEY=...
python run_openai_eval.py --model gpt-5-mini --samples "$SAMPLES" \
  --out "$PRED_DIR/preds_gpt-5-mini.jsonl" --parallel --max-workers 35
```

Parallel OpenAI only (batch with progress output):

```bash
export OPENAI_API_KEY=...
SAMPLES=/home/epsteine/post-training/data_filtering/filtering_eval/data/gold_dataset.jsonl
PRED_DIR=/home/epsteine/post-training/data_filtering/filtering_eval/predictions

python run_openai_eval.py --batch --wait --model gpt-5-mini \
  --samples "$SAMPLES" --out "$PRED_DIR/preds_gpt-5-mini.jsonl" &
python run_openai_eval.py --batch --wait --model gpt-5.2 \
  --samples "$SAMPLES" --out "$PRED_DIR/preds_gpt-5.2.jsonl" &
wait
```
python run_openai_eval.py --model gpt-5-mini --samples "$SAMPLES" --out "$PRED_DIR/preds_gpt-5-mini.jsonl"

```bash
OPENAI_MODELS=gpt-5.2-pro \
GEMINI_MODELS=gemini-3-pro-preview,gemini-3-flash-preview \
SAMPLES=/home/epsteine/post-training/data_filtering/filtering_eval/data/gold_dataset.jsonl \
PRED_DIR=/home/epsteine/post-training/data_filtering/filtering_eval/predictions \
OUT_JSON=/home/epsteine/post-training/data_filtering/filtering_eval/results/summary.json \
OUT_TEX=/home/epsteine/post-training/data_filtering/filtering_eval/results/filtering_eval_table.tex \
./run_openai_gemini_parallel.sh
```

## 3) Score and emit LaTeX table

Gold labels are taken from the `gpt-5.2-pro` predictions:

```bash
python score_predictions.py \
  --gold-from-model gpt-5.2-pro \
  --out-tex /home/epsteine/post-training/data_filtering/filtering_eval/results/filtering_eval_table.tex
```

If you manually label years in `samples.jsonl` (e.g., add a `gold_year` field), use:

```bash
python score_predictions.py \
  --gold-field gold_year \
  --gold-path /home/epsteine/post-training/data_filtering/filtering_eval/data/gold_dataset.jsonl
```

The script writes `results/summary.json` and `results/filtering_eval_table.tex`.

## Metrics

- Exact accuracy: predicted year equals gold year.
- Conservative accuracy: predicted year >= gold year.
- Weighted accuracy: mean of exact and conservative.

## Notes

- Requires `OPENAI_API_KEY` in the environment.
- Batch mode uses the OpenAI Batch API and writes request JSONL to `requests/`.
