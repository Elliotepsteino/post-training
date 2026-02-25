SHELL := /bin/bash

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))/
SUPPRESS_ALTS_WARNING ?= 1
ALTS_WARNING_FILTER = $(if $(filter 1,$(SUPPRESS_ALTS_WARNING)),2> >(grep -v "ALTS creds ignored" >&2),)

INPUT ?= $(ROOT)synthetic_sentences_120_by_entity.json
SYNTH_INPUT ?= $(ROOT)entities_100_general_domain.json
SYNTH_BY_ENTITY ?= $(ROOT)synthetic_sentences_120_by_entity.json
SYNTH_FLAT_JSONL ?= $(ROOT)synthetic_sentences_120_flat.jsonl
SYNTH_MODEL ?= gemini-3-pro-preview
SYNTH_MAX_WORKERS ?= 20
SYNTH_MAX_WORKERS_GENERATION ?= 5
SYNTH_MAX_RETRIES ?= 3
SYNTH_REQUEST_TIMEOUT ?= 90
SYNTH_EXPLICIT_COUNT ?= 60
SYNTH_EXPLICIT_MULTI_COUNT ?= 20
SYNTH_IMPLICIT_COUNT ?= 20
SYNTH_MULTI_IMPLICIT_COUNT ?= 20
IMPLICIT_CONCEPTS ?= $(ROOT)implicit_concepts_20_evolving.json
SYNTH_SEED ?= 13
OUT_JSONL ?= $(ROOT)results/synthetic_three_agent_predictions.jsonl
OUT_METRICS ?= $(ROOT)results/synthetic_three_agent_metrics.json
OUT_DIR ?= $(ROOT)results/plots

AGENT_MODELS ?= gpt-5-mini,gemini-3-flash,claude-4.5-haiku
JUDGE_MODEL ?= gemini-3-flash
MAX_WORKERS ?= 20
CLAUDE_MAX_WORKERS ?= 20
CLAUDE_FALLBACK_WORKERS ?= 0
MAX_RETRIES ?= 2
REQUEST_TIMEOUT ?= 90

.PHONY: help
help:
	@echo "Synthetic temporal eval targets:"
	@echo "  make generate_synthetic_data       # build synthetic by_entity + flat data (60 explicit, 20 explicit_multi, 20 implicit, 20 multi_implicit by default)"
	@echo "  make eval_three_agent              # run three-agent eval and generate plots (no synthetic data regeneration)"
	@echo "  make three_agent_eval              # run 3-agent + judge pipeline"
	@echo "  make three_agent_plots             # plot metrics from existing predictions"
	@echo "  make full_synthetic_eval           # generate data + run pipeline + plots"
	@echo "  # Claude controls: CLAUDE_MAX_WORKERS (default 20), CLAUDE_FALLBACK_WORKERS (default 0)"

.PHONY: generate_synthetic_data
generate_synthetic_data:
	mkdir -p $$(dirname $(SYNTH_BY_ENTITY))
	mkdir -p $$(dirname $(SYNTH_FLAT_JSONL))
	python $(ROOT)generate_synthetic_sentences.py \
	  --input $(SYNTH_INPUT) \
	  --out-by-entity $(SYNTH_BY_ENTITY) \
	  --out-flat-jsonl $(SYNTH_FLAT_JSONL) \
	  --model $(SYNTH_MODEL) \
	  --max-workers $(SYNTH_MAX_WORKERS_GENERATION) \
	  --max-retries $(SYNTH_MAX_RETRIES) \
	  --request-timeout $(SYNTH_REQUEST_TIMEOUT) \
	  --explicit-count $(SYNTH_EXPLICIT_COUNT) \
	  --explicit-multi-count $(SYNTH_EXPLICIT_MULTI_COUNT) \
	  --implicit-count $(SYNTH_IMPLICIT_COUNT) \
	  --multi-implicit-count $(SYNTH_MULTI_IMPLICIT_COUNT) \
	  --implicit-concepts $(IMPLICIT_CONCEPTS) \
	  --seed $(SYNTH_SEED) $(ALTS_WARNING_FILTER)

.PHONY: three_agent_eval
three_agent_eval:
	mkdir -p $(ROOT)results
	mkdir -p $$(dirname $(OUT_JSONL))
	mkdir -p $$(dirname $(OUT_METRICS))
	python $(ROOT)run_three_agent_eval.py \
	  --input $(INPUT) \
	  --out-jsonl $(OUT_JSONL) \
	  --agent-models "$(AGENT_MODELS)" \
	  --judge-model "$(JUDGE_MODEL)" \
	  --max-workers $(MAX_WORKERS) \
	  --claude-max-workers $(CLAUDE_MAX_WORKERS) \
	  --claude-fallback-workers $(CLAUDE_FALLBACK_WORKERS) \
	  --max-retries $(MAX_RETRIES) \
	  --request-timeout $(REQUEST_TIMEOUT) $(ALTS_WARNING_FILTER)

.PHONY: three_agent_plots
three_agent_plots:
	mkdir -p $(OUT_DIR)
	mkdir -p $$(dirname $(OUT_METRICS))
	python $(ROOT)plot_three_agent_results.py \
	  --preds $(OUT_JSONL) \
	  --out-dir $(OUT_DIR) \
	  --out-metrics $(OUT_METRICS) \
	  --model-names "$(AGENT_MODELS)"

.PHONY: full_synthetic_eval
full_synthetic_eval: generate_synthetic_data three_agent_eval three_agent_plots

.PHONY: eval_three_agent
eval_three_agent: three_agent_eval three_agent_plots
