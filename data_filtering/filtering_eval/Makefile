SHELL := /bin/bash

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PRED_DIR ?= $(ROOT)/predictions
RESULTS_DIR ?= $(ROOT)/results
SAMPLES ?= $(ROOT)/data/gold_dataset_dev.jsonl
GOLD_PATH ?= $(ROOT)/data/gold_dataset_dev.jsonl

OPENAI_MODELS ?= gpt-5-mini,gpt-5.2
GEMINI_MODELS ?= gemini-3-flash-preview,gemini-3-pro-preview
OPENAI_MAX_WORKERS ?= 200
GEMINI_MAX_WORKERS ?= 200
NUM_SAMPLES ?= 5

SEARCH_MODEL ?= gpt-4o-mini
SEARCH_MAX_WORKERS ?= 10
UPDATE_MAX_WORKERS ?= 8
AGGREGATION ?= max
AGGREGATE_MAX_WORKERS ?= 4

MODEL ?= gemini-3-flash-preview

.PHONY: help
help:
	@echo "Filtering eval targets:"
	@echo "  make sample                         # sample questions from shards"
	@echo "  make predict                         # generate predictions (OpenAI+Gemini)"
	@echo "  make predict_flash                   # generate predictions (Gemini Flash only)"
	@echo "  make grounded_pipeline               # generate -> ground -> update -> analyze"
	@echo "  make grounded_pipeline_multisample   # grounded pipeline with NUM_SAMPLES"
	@echo "  make grounded_pipeline_flash         # grounded pipeline for Gemini Flash only"
	@echo "  make gemini_flash_full               # Gemini Flash end-to-end (5 samples + plots)"
	@echo "  make ground MODEL=...                # step 2: ground entities for one model"
	@echo "  make update MODEL=...                # step 3: update estimates for one model"
	@echo "  make classify MODEL=...              # classify non-conservative cases"
	@echo "  make analyze                         # plots + grounding_summary.json"
	@echo "  make agg_plot                        # 3x2 delta grid (LLM + rank1-5)"
	@echo "  make score                           # score predictions and emit table"

.PHONY: sample
sample:
	python $(ROOT)/sample_questions.py \
	  --num-samples 50 \
	  --out $(ROOT)/data/samples.jsonl

.PHONY: predict
predict:
	OPENAI_MODELS=$(OPENAI_MODELS) \
	GEMINI_MODELS=$(GEMINI_MODELS) \
	OPENAI_MAX_WORKERS=$(OPENAI_MAX_WORKERS) \
	GEMINI_MAX_WORKERS=$(GEMINI_MAX_WORKERS) \
	NUM_SAMPLES=$(NUM_SAMPLES) \
	$(ROOT)/run_openai_gemini_parallel.sh

.PHONY: predict_flash
predict_flash:
	OPENAI_MODELS= \
	GEMINI_MODELS=gemini-3-flash-preview \
	GEMINI_MAX_WORKERS=$(GEMINI_MAX_WORKERS) \
	NUM_SAMPLES=$(NUM_SAMPLES) \
	$(ROOT)/run_openai_gemini_parallel.sh

.PHONY: grounded_pipeline
grounded_pipeline:
	$(ROOT)/run_grounded_pipeline.sh

.PHONY: grounded_pipeline_multisample
grounded_pipeline_multisample:
	NUM_SAMPLES=$(NUM_SAMPLES) \
	$(ROOT)/run_grounded_pipeline_multisample.sh

.PHONY: grounded_pipeline_flash
grounded_pipeline_flash:
	NUM_SAMPLES=$(NUM_SAMPLES) \
	OPENAI_MODELS= \
	GEMINI_MODELS=gemini-3-flash-preview \
	$(ROOT)/run_grounded_pipeline_multisample_gemini_flash.sh

.PHONY: gemini_flash_full
gemini_flash_full:
	NUM_SAMPLES=5 \
	OPENAI_MODELS= \
	GEMINI_MODELS=gemini-3-flash-preview \
	GEMINI_MAX_WORKERS=$(GEMINI_MAX_WORKERS) \
	$(ROOT)/run_openai_gemini_parallel.sh
	python $(ROOT)/ground_entities_with_search.py \
	  --preds $(PRED_DIR)/preds_gemini-3-flash-preview.jsonl \
	  --out $(PRED_DIR)/preds_gemini-3-flash-preview_grounded.jsonl \
	  --model $(SEARCH_MODEL) \
	  --max-workers $(SEARCH_MAX_WORKERS)
	python $(ROOT)/update_estimates_with_evidence.py \
	  --preds $(PRED_DIR)/preds_gemini-3-flash-preview_grounded.jsonl \
	  --out $(PRED_DIR)/preds_gemini-3-flash-preview_grounded_updated.jsonl \
	  --max-workers $(UPDATE_MAX_WORKERS) \
	  --aggregation llm \
	  --aggregate-max-workers $(AGGREGATE_MAX_WORKERS)
	python $(ROOT)/classify_nonconservative_cases.py \
	  --preds $(PRED_DIR)/preds_gemini-3-flash-preview_grounded_updated.jsonl \
	  --gold-path $(GOLD_PATH) \
	  --out $(PRED_DIR)/preds_gemini-3-flash-preview_nonconservative.jsonl
	python $(ROOT)/analyze_grounding_impact.py \
	  --pred-dir $(PRED_DIR) \
	  --updated-dir $(PRED_DIR) \
	  --updated-suffix _grounded_updated.jsonl \
	  --gold-path $(GOLD_PATH) \
	  --out-json $(RESULTS_DIR)/grounding_summary.json \
	  --out-plot $(RESULTS_DIR)/grounding_impact.pdf
	python $(ROOT)/plot_aggregation_delta_grid.py \
	  --preds $(PRED_DIR)/preds_gemini-3-flash-preview_grounded_updated.jsonl \
	  --gold-path $(GOLD_PATH) \
	  --out-pdf $(RESULTS_DIR)/grounding_aggregation_delta_grid.pdf

.PHONY: ground
ground:
	python $(ROOT)/ground_entities_with_search.py \
	  --preds $(PRED_DIR)/preds_$(MODEL).jsonl \
	  --out $(PRED_DIR)/preds_$(MODEL)_grounded.jsonl \
	  --model $(SEARCH_MODEL) \
	  --max-workers $(SEARCH_MAX_WORKERS)

.PHONY: update
update:
	python $(ROOT)/update_estimates_with_evidence.py \
	  --preds $(PRED_DIR)/preds_$(MODEL)_grounded.jsonl \
	  --out $(PRED_DIR)/preds_$(MODEL)_grounded_updated.jsonl \
	  --max-workers $(UPDATE_MAX_WORKERS) \
	  --aggregation $(AGGREGATION) \
	  --aggregate-max-workers $(AGGREGATE_MAX_WORKERS)

.PHONY: classify
classify:
	python $(ROOT)/classify_nonconservative_cases.py \
	  --preds $(PRED_DIR)/preds_$(MODEL)_grounded_updated.jsonl \
	  --gold-path $(GOLD_PATH) \
	  --out $(PRED_DIR)/preds_$(MODEL)_nonconservative.jsonl

.PHONY: analyze
analyze:
	python $(ROOT)/analyze_grounding_impact.py \
	  --pred-dir $(PRED_DIR) \
	  --updated-dir $(PRED_DIR) \
	  --updated-suffix _grounded_updated.jsonl \
	  --gold-path $(GOLD_PATH) \
	  --out-json $(RESULTS_DIR)/grounding_summary.json \
	  --out-plot $(RESULTS_DIR)/grounding_impact.pdf

.PHONY: agg_plot
agg_plot:
	python $(ROOT)/plot_aggregation_delta_grid.py \
	  --preds $(PRED_DIR)/preds_$(MODEL)_grounded_updated.jsonl \
	  --gold-path $(GOLD_PATH) \
	  --out-pdf $(RESULTS_DIR)/grounding_aggregation_delta_grid.pdf

.PHONY: score
score:
	python $(ROOT)/score_predictions.py \
	  --gold-path $(SAMPLES) \
	  --gold-field gold_year \
	  --pred-dir $(PRED_DIR) \
	  --out-json $(RESULTS_DIR)/summary.json \
	  --out-tex $(RESULTS_DIR)/filtering_eval_table.tex
