# Tulu-3 Year Classification

This folder contains a batch-friendly pipeline that labels samples from `allenai/tulu-3-sft-mixture` with the minimum calendar year (between 2001 and 2025) needed to answer each question without temporal leakage. The output is one JSONL shard per year plus a lightweight dataloader that merges shards up to a desired cutoff year.

## Prerequisites

- Hugging Face auth (the dataset requires `huggingface-cli login` beforehand).
- Python deps: `pip install datasets openai`.
- Optional for post-processing plots: `pip install matplotlib`.
- `OPENAI_API_KEY` exported in your environment (`export OPENAI_API_KEY=...`).
- OpenAI Batch API access (the script uses `gpt-5-mini` with the 24h completion window).

### Optional: UV virtual environment

If you prefer isolating dependencies with [uv](https://github.com/astral-sh/uv):

```bash
cd /home/epsteine/qwen
uv venv .venv-tulu-year
source .venv-tulu-year/bin/activate
uv pip install datasets openai matplotlib
uv pip install torch
uv pip install transformers
uv pip install peft
```

The activated shell now has the required packages while keeping the global Python clean. Re-run `source .venv-tulu-year/bin/activate` whenever you come back to the project.
The extra torch line pulls the CUDA 12.5 wheels directly from the official PyTorch index; omit or change the index URL if you target a different CUDA runtime.

## Running the classifier

```bash
cd /home/epsteine/qwen
python -m data_filtering.tulu-3 \
  --dataset-name allenai/tulu-3-sft-mixture \
  --subset-size 50 \
  --output-dir data_filtering/tulu_year_shards
```

Switch `--dataset-name` (default `allenai/tulu-3-sft-mixture`) to target additional corpora such as `allenai/llama-3.1-tulu-3-8b-preference-mixture`, `allenai/RLVR-GSM`, `allenai/RLVR-MATH`, or `allenai/RLVR-IFeval`. Pair it with `--dataset-split` (default `train`) if you need a different partition. The sampler now understands each schema: DPO runs feed both the preferred and rejected completions to the classifier, while RLVR runs include the full conversation, gold rationale, and any constraint metadata so the assigned year reflects the newest fact anywhere in the bundle.

To run the exact same pipeline without the Batch API (synchronous Chat Completions, still emitting batch-style JSONL files), add `--no-use-batch`:

```bash
python -m data_filtering.tulu-3 \
  --subset-size 50 \
  --output-dir data_filtering/tulu_year_shards \
  --no-use-batch
```

### End-to-end helper

To run the classifier and produce both histograms in one go, pass any desired flags to the helper script (they are forwarded to `tulu-3.py`):

```bash
bash data_filtering/run_full_pipeline.sh \
  --subset-size 50 \
  --output-dir data_filtering/tulu_year_shards
```

The script forwards every argument to `tulu-3.py`, detects the `--output-dir` value (defaulting to `data_filtering/tulu_year_shards`), reads `batch_metadata_latest.json` from that directory to locate the newest `year_shards_<run_id>` folder, and then renders both histograms there.

### Multi-dataset helper

To classify several datasets in one sweep (currently the SFT mixture, the Tulu-3 DPO mixture, and the three RLVR sets) with a uniform configuration, use:

```bash
bash data_filtering/run_all_filters.sh --no-use-batch
```

Environment variables control the shared knobs:

- `SUBSET_SIZE` (default `10`)
- `OUTPUT_ROOT` (default `data_filtering/tulu_year_shards`)
- `DATASET_SPLIT` (default `train`)

Any additional CLI flags (e.g., `--no-use-batch`, `--model gpt-4.1`) are forwarded to each `python -m data_filtering.tulu-3` invocation. The helper now launches every dataset pipeline in parallel, so batches for SFT/DPO/RLVR are submitted simultaneously; expect higher instantaneous OpenAI/HF traffic. After each dataset finishes it calls `python -m data_filtering.year_histogram --shard-dir ...` so year/category PDFs are rendered automatically (requires `matplotlib`). Every invocation of the helper creates a session directory named after the current U.S. Pacific time (PT) stamp (override with `SESSION_STAMP=...`). All dataset-specific outputs for that session live under `data_filtering/tulu_year_shards/<session_stamp>/<dataset-slug>/...`, making it easy to compare artifacts produced in the same sweep.

Key behavior:

- Loads the train split, samples 50 examples (shuffled by `--seed`), and extracts user/assistant turns.
- Builds an OpenAI Batch-style input file with `gpt-5-mini` prompts.
- Writes the payload to `batch_input_<timestamp>_n<count>.jsonl` (with an easily readable UTC timestamp), then (by default) submits it via the Batch API; `--no-use-batch` reuses the same file for synchronous calls.
- Batch output files mirror the same identifier: `batch_output_<timestamp>_n<count>_<fileid>.jsonl` for batch mode and `batch_output_local_<timestamp>_n<count>.jsonl` for live mode. Metadata is stored in `batch_metadata_<timestamp>_n<count>.json`, with `batch_metadata_latest.json` pointing at the most recent run.
- Polls the batch until completion (or runs live Chat Completions), parses the assigned year and question category for every sample, and writes shards named `year=YYYY_<run_id>.jsonl` inside a run-scoped folder `year_shards_<run_id>` so each file name carries the timestamp and sample count.
- Each shard directory also receives a `manifest.json` that records the covered years, total count, and run identifier.

Flags of interest:

- `--subset-size`: change the number of rows (default 50; start small before scaling).
- `--resume-batch-id`: skip submission and resume polling/downloading an existing batch.
- `--seed`: controls subset sampling and shard shuffle reproducibility.
- `--use-batch/--no-use-batch`: toggle whether requests go through the OpenAI Batch API (default on). The non-batch mode reuses the same input/output JSONL schema for downstream parity.
- The run identifier `<timestamp>_n<count>` is embedded everywhere (inputs, outputs, metadata, per-year shards) so you can trace artifacts from start to finish.
- Each prediction also includes `category`, chosen from: `general_knowledge`, `math`, `coding`, `science`, `history`, `law`, `finance`, `health`, `creative_writing`, `multi_lingual`, `instruction_following`, `reasoning`, `other`.

To scale beyond the default, increase `--subset-size` or update the sampling logic; the Batch payload and polling loop already work for larger jobs, but be mindful of rate limits and storage.

## Consuming the shards

The script exposes `YearBoundedTuluLoader` for merging and shuffling data up to a cutoff. Point it at a specific run directory (e.g., `data_filtering/tulu_year_shards/2026-01-05_12-21PT/allenai-tulu-3-sft-mixture/year_shards_allenai-tulu-3-sft-mixture_2026-01-05_20-21Z_n10`):

```python
from pathlib import Path
from data_filtering.tulu-3 import YearBoundedTuluLoader

run_dir = Path("data_filtering/tulu_year_shards/2026-01-05_12-21PT/allenai-tulu-3-sft-mixture/year_shards_allenai-tulu-3-sft-mixture_2026-01-05_20-21Z_n10")
loader = YearBoundedTuluLoader(run_dir)
dataset = loader.load(max_year=2014, shuffle=True, seed=7)
print(dataset[:2])
```

This returns a Hugging Face `Dataset` assembled from every `year=YYYY_<run_id>.jsonl` shard where `YYYY <= max_year`, shuffled if requested. Use the resulting dataset directly in downstream dataloaders or convert it to other formats as needed.

Each record exposes: `sample_index`, `year`, `category`, `confidence`, `justification`, `evidence_years`, `question`, `answer`, and the original metadata fields that were present in the source row.

## Post-processing histograms

To inspect the distributions for any run, point the helper at its shard directory (requires `matplotlib`):

```bash
python -m data_filtering.year_histogram \
  --shard-dir data_filtering/tulu_year_shards/year_shards_2025-12-01_19-03Z_n3000
```

The script prints counts per year and saves publication-grade PDF histograms for both year and question-category distributions (default filenames `year_histogram.pdf` and `category_histogram.pdf`) in the shard directory. Year plots use the last two digits for readability and a log-scaled y-axis by default to handle heavy 2001 skew. Override the year plot location with `--output-file` if needed.
