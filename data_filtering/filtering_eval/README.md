# Filtering Eval (Year Labels)

This directory contains a small evaluation pipeline to sanity check year labels on the gold set.

## 1) Sample questions

```bash
python sample_questions.py \
  --num-samples 50 \
  --out /home/epsteine/post-training/data_filtering/filtering_eval/data/samples.jsonl
```

This draws evenly across available year shards (latest session by default) and stores `question` plus metadata.

## 2) Run model predictions and grounding via Make

Use `make help` to see all targets. The main workflow is:

1) Generate predictions (optionally multi-sample).
2) Ground entities via search.
3) Update estimates using evidence (rule or LLM aggregation).
4) Analyze and plot.

Example: full grounded pipeline (generate → search → update → analyze):

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...  # or GOOGLE_API_KEY

make grounded_pipeline
```

Multi-sample grounded pipeline:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...  # or GOOGLE_API_KEY

NUM_SAMPLES=5 make grounded_pipeline_multisample
```

Gemini 3 Flash only:

```bash
export GEMINI_API_KEY=...  # or GOOGLE_API_KEY

NUM_SAMPLES=5 make grounded_pipeline_flash
```

Manual steps (single model example):

```bash
export GEMINI_API_KEY=...  # or GOOGLE_API_KEY

MODEL=gemini-3-flash-preview make ground
MODEL=gemini-3-flash-preview AGGREGATION=llm make update
MODEL=gemini-3-flash-preview make classify
make analyze
```

### Aggregation comparisons

Run the update step once and compare LLM vs rank1-5:

```bash
MODEL=gemini-3-flash-preview AGGREGATION=llm make update
MODEL=gemini-3-flash-preview make agg_plot
```

Pipeline outputs:
- `file_storage/data_filtering/filtering_eval/results/grounding_summary.json`
- `file_storage/data_filtering/filtering_eval/results/grounding_impact_*.pdf`
- `file_storage/data_filtering/filtering_eval/results/grounding_aggregation_delta_grid.pdf`

## 3) Score and emit LaTeX table

Gold labels are taken from the `gpt-5.2-pro` predictions:

```bash
make score
```

If you manually label years in `samples.jsonl` (e.g., add a `gold_year` field), use:

```bash
make score GOLD_PATH=/home/epsteine/post-training/data_filtering/filtering_eval/data/gold_dataset_dev.jsonl GOLD_FIELD=gold_year
```

The script writes `file_storage/data_filtering/filtering_eval/results/summary.json` and `file_storage/data_filtering/filtering_eval/results/filtering_eval_table.tex`.

To score against the manually annotated `gold_dataset_test_elliot.jsonl`, use:

```bash
make score_elliot
```

This writes `file_storage/data_filtering/filtering_eval/results/summary_elliot.json` and `file_storage/data_filtering/filtering_eval/results/filtering_eval_table_elliot.tex`.
It expects predictions in `file_storage/data_filtering/filtering_eval/predictions/test_elliot/` by default.

## Metrics

- Exact accuracy: predicted year equals gold year.
- Conservative accuracy: predicted year >= gold year.
- Weighted accuracy: mean of exact and conservative.

## Notes

- Requires `OPENAI_API_KEY` in the environment.
- Batch mode uses the OpenAI Batch API and writes request JSONL to `requests/`.
