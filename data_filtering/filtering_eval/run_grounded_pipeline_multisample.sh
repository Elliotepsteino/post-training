#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NUM_SAMPLES="${NUM_SAMPLES:-5}"

cd "$ROOT"
./run_grounded_pipeline.sh
