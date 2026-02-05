SHELL := /bin/bash

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(ROOT)/../..)
FILE_STORAGE_ROOT ?= $(REPO_ROOT)/file_storage
PRED_DIR ?= $(FILE_STORAGE_ROOT)/data_filtering/filtering_eval/predictions
RESULTS_DIR ?= $(FILE_STORAGE_ROOT)/data_filtering/filtering_eval/results
SAMPLES ?= $(ROOT)/data/gold_dataset_dev.jsonl
GOLD_PATH ?= $(ROOT)/data/gold_dataset_dev.jsonl
GOLD_FIELD ?= gold_year

OPENAI_MODELS ?= gpt-5-mini,gpt-5.2
GEMINI_MODELS ?= gemini-3-flash-preview,gemini-3-pro-preview
OPENAI_MAX_WORKERS ?= 400
GEMINI_MAX_WORKERS ?= 400
NUM_SAMPLES ?= 5

SEARCH_MODEL ?= gpt-4o-mini
SEARCH_MAX_WORKERS ?= 50
UPDATE_MAX_WORKERS ?= 32
AGGREGATION ?= max
AGGREGATE_MAX_WORKERS ?= 8

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
	@echo "  make score_elliot                    # score predictions vs gold_dataset_test_elliot"
	@echo "  make sync_paper                      # copy result figures into paper/figures"

.PHONY: sample
sample:
	mkdir -p $(ROOT)/data
	python $(ROOT)/sample_questions.py \
	  --num-samples 50 \
	  --out $(ROOT)/data/samples.jsonl

.PHONY: predict
predict:
	mkdir -p $(PRED_DIR)
	mkdir -p $(RESULTS_DIR)
	OPENAI_MODELS=$(OPENAI_MODELS) \
	GEMINI_MODELS=$(GEMINI_MODELS) \
	OPENAI_MAX_WORKERS=$(OPENAI_MAX_WORKERS) \
	GEMINI_MAX_WORKERS=$(GEMINI_MAX_WORKERS) \
	NUM_SAMPLES=$(NUM_SAMPLES) \
	PRED_DIR=$(PRED_DIR) \
	OUT_JSON=$(RESULTS_DIR)/summary.json \
	OUT_TEX=$(RESULTS_DIR)/filtering_eval_table.tex \
	$(ROOT)/run_openai_gemini_parallel.sh

.PHONY: predict_flash
predict_flash:
	mkdir -p $(PRED_DIR)
	mkdir -p $(RESULTS_DIR)
	OPENAI_MODELS= \
	GEMINI_MODELS=gemini-3-flash-preview \
	GEMINI_MAX_WORKERS=$(GEMINI_MAX_WORKERS) \
	NUM_SAMPLES=$(NUM_SAMPLES) \
	PRED_DIR=$(PRED_DIR) \
	OUT_JSON=$(RESULTS_DIR)/summary.json \
	OUT_TEX=$(RESULTS_DIR)/filtering_eval_table.tex \
	$(ROOT)/run_openai_gemini_parallel.sh

.PHONY: grounded_pipeline
grounded_pipeline:
	mkdir -p $(PRED_DIR) $(RESULTS_DIR)
	PRED_DIR=$(PRED_DIR) \
	OUT_JSON=$(RESULTS_DIR)/grounding_summary.json \
	OUT_PLOT=$(RESULTS_DIR)/grounding_impact.pdf \
	$(ROOT)/run_grounded_pipeline.sh

.PHONY: grounded_pipeline_multisample
grounded_pipeline_multisample:
	mkdir -p $(PRED_DIR) $(RESULTS_DIR)
	NUM_SAMPLES=$(NUM_SAMPLES) \
	PRED_DIR=$(PRED_DIR) \
	OUT_JSON=$(RESULTS_DIR)/grounding_summary.json \
	OUT_PLOT=$(RESULTS_DIR)/grounding_impact.pdf \
	$(ROOT)/run_grounded_pipeline_multisample.sh

.PHONY: grounded_pipeline_flash
grounded_pipeline_flash:
	mkdir -p $(PRED_DIR) $(RESULTS_DIR)
	NUM_SAMPLES=$(NUM_SAMPLES) \
	OPENAI_MODELS= \
	GEMINI_MODELS=gemini-3-flash-preview \
	PRED_DIR=$(PRED_DIR) \
	OUT_JSON=$(RESULTS_DIR)/grounding_summary.json \
	OUT_PLOT=$(RESULTS_DIR)/grounding_impact.pdf \
	$(ROOT)/run_grounded_pipeline_multisample_gemini_flash.sh

.PHONY: gemini_flash_full
gemini_flash_full:
	mkdir -p $(PRED_DIR) $(RESULTS_DIR)
	NUM_SAMPLES=5 \
	OPENAI_MODELS= \
	GEMINI_MODELS=gemini-3-flash-preview \
	GEMINI_MAX_WORKERS=$(GEMINI_MAX_WORKERS) \
	PRED_DIR=$(PRED_DIR) \
	OUT_JSON=$(RESULTS_DIR)/summary.json \
	OUT_TEX=$(RESULTS_DIR)/filtering_eval_table.tex \
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
	mkdir -p $(PRED_DIR)
	python $(ROOT)/ground_entities_with_search.py \
	  --preds $(PRED_DIR)/preds_$(MODEL).jsonl \
	  --out $(PRED_DIR)/preds_$(MODEL)_grounded.jsonl \
	  --model $(SEARCH_MODEL) \
	  --max-workers $(SEARCH_MAX_WORKERS)

.PHONY: update
update:
	mkdir -p $(PRED_DIR)
	python $(ROOT)/update_estimates_with_evidence.py \
	  --preds $(PRED_DIR)/preds_$(MODEL)_grounded.jsonl \
	  --out $(PRED_DIR)/preds_$(MODEL)_grounded_updated.jsonl \
	  --max-workers $(UPDATE_MAX_WORKERS) \
	  --aggregation $(AGGREGATION) \
	  --aggregate-max-workers $(AGGREGATE_MAX_WORKERS)

.PHONY: classify
classify:
	mkdir -p $(PRED_DIR)
	python $(ROOT)/classify_nonconservative_cases.py \
	  --preds $(PRED_DIR)/preds_$(MODEL)_grounded_updated.jsonl \
	  --gold-path $(GOLD_PATH) \
	  --out $(PRED_DIR)/preds_$(MODEL)_nonconservative.jsonl

.PHONY: analyze
analyze:
	mkdir -p $(RESULTS_DIR)
	python $(ROOT)/analyze_grounding_impact.py \
	  --pred-dir $(PRED_DIR) \
	  --updated-dir $(PRED_DIR) \
	  --updated-suffix _grounded_updated.jsonl \
	  --gold-path $(GOLD_PATH) \
	  --out-json $(RESULTS_DIR)/grounding_summary.json \
	  --out-plot $(RESULTS_DIR)/grounding_impact.pdf

.PHONY: agg_plot
agg_plot:
	mkdir -p $(RESULTS_DIR)
	python $(ROOT)/plot_aggregation_delta_grid.py \
	  --preds $(PRED_DIR)/preds_$(MODEL)_grounded_updated.jsonl \
	  --gold-path $(GOLD_PATH) \
	  --out-pdf $(RESULTS_DIR)/grounding_aggregation_delta_grid.pdf

.PHONY: score
score:
	mkdir -p $(RESULTS_DIR)
	python $(ROOT)/score_predictions.py \
	  --gold-path $(SAMPLES) \
	  --gold-field $(GOLD_FIELD) \
	  --pred-dir $(PRED_DIR) \
	  --out-json $(RESULTS_DIR)/summary.json \
	  --out-tex $(RESULTS_DIR)/filtering_eval_table.tex

.PHONY: score_elliot
score_elliot:
	mkdir -p $(RESULTS_DIR)
	python $(ROOT)/score_predictions.py \
	  --gold-path $(ROOT)/data/gold_dataset_test_elliot.jsonl \
	  --gold-field gold_year \
	  --pred-dir $(PRED_DIR)/test_elliot \
	  --out-json $(RESULTS_DIR)/summary_elliot.json \
	  --out-tex $(RESULTS_DIR)/filtering_eval_table_elliot.tex

.PHONY: sync_paper
sync_paper:
	mkdir -p $(REPO_ROOT)/paper/figures
	@if [ -f $(RESULTS_DIR)/grounding_impact_conservative.pdf ]; then cp $(RESULTS_DIR)/grounding_impact_conservative.pdf $(REPO_ROOT)/paper/figures/; fi
	@if [ -f $(RESULTS_DIR)/grounding_impact_delta_search_grid.pdf ]; then cp $(RESULTS_DIR)/grounding_impact_delta_search_grid.pdf $(REPO_ROOT)/paper/figures/; fi
	@if [ -f $(RESULTS_DIR)/grounding_impact_test_delta_search_grid.pdf ]; then cp $(RESULTS_DIR)/grounding_impact_test_delta_search_grid.pdf $(REPO_ROOT)/paper/figures/; fi
	@if [ -f $(RESULTS_DIR)/conservatism_counts_max_combo.pdf ]; then cp $(RESULTS_DIR)/conservatism_counts_max_combo.pdf $(REPO_ROOT)/paper/figures/; fi
