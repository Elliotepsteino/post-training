# Qwen Training Utilities

This directory holds lightweight scripts for fine-tuning Qwen 3 models with LoRA adapters. The original `train_qwen3_sft.py` targets generic JSON datasets containing `{question, response}` pairs.

## Tulu-3 Filtered SFT

Use `train_qwen3_sft_tulu3.py` to fine-tune on the year-filtered shards produced by `data_filtering/tulu-3.py`. It automatically loads every `year=YYYY.jsonl` shard up to a target cutoff year, mixes the samples, and reuses the same Fermi-style prompt template as the base trainer.

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

## TÜLU-3 Dev Sanity Eval (multi-GPU)

The evaluation-specific workflow (including the `run_tulu3_dev_limit8.sh` helper, log locations, and summary regeneration snippet) now lives in `olmes/README.md`. See that file for the most up-to-date instructions.
