# Qwen Training Utilities

This directory holds lightweight scripts for fine-tuning Qwen 3 models with LoRA adapters. The original `train_qwen3_sft.py` targets generic JSON datasets containing `{question, response}` pairs.

## Workflow map

Use these three entry points to navigate the pipeline from data prep to evaluation:

1. **Data filtering**: `post-training/data_filtering/README.md` (year-sharded dataset preparation).
2. **Post training**: `post-training/open-instruct/README.md` (SFT/LoRA/DPO training recipes).
3. **Model eval**: `post-training/olmes/README.md` (evaluation runs and summaries).

## LaTeX results summary

The `post-training/paper/` folder contains the paper sources; see `post-training/paper/main.tex`.

## Tulu-3 Filtered SFT

Use `train_qwen3_sft_tulu3.py` to fine-tune on the year-filtered shards produced by `data_filtering/tulu-3.py`. It automatically loads every `year=YYYY.jsonl` shard up to a target cutoff year, mixes the samples, and reuses the same Fermi-style prompt template as the base trainer.

## Create a merged dataset with dataloader.py

Use `dataloader.py` to export a merged, HF-friendly JSONL dataset that contains `question`, `response`, and a `messages` chat column. The output filename includes the number of samples (for example, `_n12345.jsonl`).

```bash
cd /home/epsteine/post-training
python dataloader.py \
  --task allenai-tulu-3-sft-mixture \
  --max-year 2007 \
  --output-dir datasets
```
#--session-stamp 2026-01-06_09-32PT \
Optional flags:

- `--session-stamp`: Pick a specific PT session (defaults to the latest).
- `--run-name`: Pick a specific `year_shards_*` run (defaults to the latest under the task).
- `--dataset-name`: Override the output filename prefix (sample count suffix is still appended).
- `--max-samples`: Cap the merged dataset size before saving.

### Example

```bash
cd /home/epsteine/qwen
python train_qwen3_sft_tulu3.py \
  --shard-dir data_filtering/tulu_year_shards/year_shards_2025-12-01_19-03Z_n3000 \
  --max-year 2015 \
  --output-dir outputs/qwen3-tulu3-y2015 \
  --batch-size 2 \
  --grad-accum 8
```

Key flags:

- `--shard-dir`: Path containing `year=YYYY.jsonl` files (can be a run-specific `year_shards_<run_id>` folder).
- `--max-year`: Include all shards `<=` this year when building the mixture.
- `--max-samples`: Optional cap to subsample the mixture for quick experiments.
- `--model-name`, `--output-dir`, `--batch-size`, `--grad-accum`, etc. mirror the options in `train_qwen3_sft.py`.

The script respects the LoRA, optimizer, and confidence interval settings exposed through CLI flags or environment variables, saving the adapter and tokenizer to `--output-dir` once training completes.

### Merging LoRA adapters into a full checkpoint

After SFT (or follow-on DPO/RLVR) training you may want a standalone checkpoint that no longer depends on PEFT adapters. Use `merge_lora.py` to fold the adapter weights into the Qwen3-4B base model:

```bash
cd /home/epsteine/post-training
python merge_lora.py \
  --base-model model_weights/Qwen3-4B-Base \
  --lora-adapter qwen3-4b-instruct-bf16-lora \
  --output-dir model_weights/Qwen3-4B-Base-lora-merged
```

The script loads both directories, calls `merge_and_unload()`, and saves the merged weights plus tokenizer to `model_weights/Qwen3-4B-Base-lora-merged`. Point any downstream eval (e.g., `run_tulu3_dev_limit100.sh`) at that merged directory to evaluate the fully materialized model.

## TÜLU-3 Dev Sanity Eval (multi-GPU)

The evaluation-specific workflow (including the `run_tulu3_dev_limit8.sh` helper, log locations, and summary regeneration snippet) now lives in `olmes/README.md`. See that file for the most up-to-date instructions.
