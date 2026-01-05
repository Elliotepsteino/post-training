#!/usr/bin/env bash
set -euo pipefail

# Runs the year-classification filter across all supported datasets with a shared configuration.
# Override SUBSET_SIZE / OUTPUT_ROOT / DATASET_SPLIT via environment variables if needed.

DATASETS=(
  "allenai/tulu-3-sft-mixture"
  "allenai/llama-3.1-tulu-3-8b-preference-mixture"
  "allenai/RLVR-GSM"
  "allenai/RLVR-MATH"
  "allenai/RLVR-IFeval"
)

SUBSET_SIZE=${SUBSET_SIZE:-10}
OUTPUT_ROOT=${OUTPUT_ROOT:-"data_filtering/tulu_year_shards"}
DATASET_SPLIT=${DATASET_SPLIT:-"train"}
SESSION_STAMP=${SESSION_STAMP:-$(TZ=America/Los_Angeles date +"%Y-%m-%d_%H-%MPT")}
SESSION_DIR="${OUTPUT_ROOT}/${SESSION_STAMP}"
EXTRA_ARGS=("$@")

mkdir -p "$SESSION_DIR"
echo "Session output root: ${SESSION_DIR}"

slugify() {
  python - "$1" <<'PY'
import importlib
import sys

mod = importlib.import_module("data_filtering.tulu-3")
print(mod.slugify_dataset(sys.argv[1]), end="")
PY
}

process_dataset() {
  local dataset="$1"
  local slug output_dir metadata_path run_id shard_dir
  slug="$(slugify "$dataset")"
  output_dir="${SESSION_DIR}/${slug}"
  mkdir -p "$output_dir"
  echo "=== Running filter for ${dataset} (slug: ${slug}) ==="
  if ! python -m data_filtering.tulu-3 \
    --dataset-name "$dataset" \
    --dataset-split "$DATASET_SPLIT" \
    --subset-size "$SUBSET_SIZE" \
    --output-dir "$output_dir" \
    "${EXTRA_ARGS[@]}"; then
    echo "ERROR: filtering failed for ${dataset}"
    return 1
  fi

  metadata_path="${output_dir}/batch_metadata_latest.json"
  if [[ ! -f "$metadata_path" ]]; then
    echo "Warning: missing metadata at ${metadata_path}, skipping plots."
    return 0
  fi

  run_id="$(python - "$metadata_path" <<'PY'
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
meta = json.loads(path.read_text())
print(meta.get("run_id", ""), end="")
PY
)"
  if [[ -z "$run_id" ]]; then
    echo "Warning: could not determine run_id from ${metadata_path}, skipping plots."
    return 0
  fi
  shard_dir="${output_dir}/year_shards_${run_id}"
  if [[ ! -d "$shard_dir" ]]; then
    echo "Warning: shard directory ${shard_dir} not found, skipping plots."
    return 0
  fi

  echo "Rendering histograms for ${shard_dir}"
  if ! python -m data_filtering.year_histogram --shard-dir "$shard_dir"; then
    echo "Warning: failed to generate histograms for ${shard_dir}"
  fi
}

PIDS=()
LABELS=()
for dataset in "${DATASETS[@]}"; do
  process_dataset "$dataset" &
  PIDS+=("$!")
  LABELS+=("$dataset")
done

STATUS=0
for idx in "${!PIDS[@]}"; do
  pid=${PIDS[$idx]}
  dataset=${LABELS[$idx]}
  if ! wait "$pid"; then
    echo "ERROR: dataset ${dataset} failed"
    STATUS=1
  fi
done

exit $STATUS
