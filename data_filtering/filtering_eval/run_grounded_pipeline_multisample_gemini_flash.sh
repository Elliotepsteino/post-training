#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NUM_SAMPLES="${NUM_SAMPLES:-5}"
export OPENAI_MODELS="${OPENAI_MODELS:-}"
export GEMINI_MODELS="${GEMINI_MODELS:-gemini-3-flash-preview}"

cd "$ROOT"
./run_grounded_pipeline.sh
