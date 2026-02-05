#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  cat <<'USAGE' >&2
Usage: run_full_pipeline.sh [tulu-3.py args]
  Example:
    run_full_pipeline.sh --subset-size 50 --output-dir data_filtering/tulu_year_shards
USAGE
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/tulu_year_shards"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"

# Preserve original args for forwarding
FORWARDED_ARGS=("$@")

# Detect --output-dir flag value (either --output-dir path or --output-dir=path)
i=0
while [ $i -lt ${#FORWARDED_ARGS[@]} ]; do
  arg="${FORWARDED_ARGS[$i]}"
  case "$arg" in
    --output-dir)
      if [ $((i + 1)) -ge ${#FORWARDED_ARGS[@]} ]; then
        echo "Error: --output-dir flag provided without a value." >&2
        exit 1
      fi
      OUTPUT_DIR="${FORWARDED_ARGS[$((i + 1))]}"
      ;;
    --output-dir=*)
      OUTPUT_DIR="${arg#*=}"
      ;;
  esac
  i=$((i + 1))
done

cd "$REPO_ROOT"
python -m data_filtering.tulu-3 "${FORWARDED_ARGS[@]}"

LATEST_META="$OUTPUT_DIR/batch_metadata_latest.json"
if [ ! -f "$LATEST_META" ]; then
  echo "Unable to locate $LATEST_META" >&2
  exit 1
fi

RUN_ID=$(
  python - "$LATEST_META" <<'PY'
import json, sys
from pathlib import Path

meta_path = Path(sys.argv[1])
with meta_path.open() as fh:
    data = json.load(fh)
run_id = data.get("run_id")
if not run_id:
    raise SystemExit("run_id missing in batch_metadata_latest.json")
print(run_id)
PY
)

SHARD_DIR="$OUTPUT_DIR/year_shards_${RUN_ID}"
if [ ! -d "$SHARD_DIR" ]; then
  echo "Shard directory $SHARD_DIR not found" >&2
  exit 1
fi

python -m data_filtering.year_histogram --shard-dir "$SHARD_DIR"

echo "Full pipeline complete. Shards: $SHARD_DIR"
printf "Plots saved under: %s/year_histogram.pdf and %s/category_histogram.pdf\n" "$SHARD_DIR" "$SHARD_DIR"
