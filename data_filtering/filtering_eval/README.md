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
