SHELL := /bin/bash

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(ROOT)/..)
FILE_STORAGE_ROOT ?= $(REPO_ROOT)/file_storage

.PHONY: help
help:
	@echo "Data filtering targets:"
	@echo "  make full_pipeline ARGS='--subset-size 50'     # run tulu-3 filter + histograms"
	@echo "  make all_filters SUBSET_SIZE=10                # run all datasets in parallel"
	@echo "  make year_hist SHARD_DIR=...                   # plot year/category histograms"
	@echo "  make eval ...                                  # delegate to filtering_eval/Makefile"

.PHONY: full_pipeline
full_pipeline:
	mkdir -p $(FILE_STORAGE_ROOT)/data_filtering/tulu_year_shards
	$(ROOT)/run_full_pipeline.sh --output-dir $(FILE_STORAGE_ROOT)/data_filtering/tulu_year_shards $(ARGS)

.PHONY: all_filters
all_filters:
	mkdir -p $(FILE_STORAGE_ROOT)/data_filtering/tulu_year_shards
	OUTPUT_ROOT=$(FILE_STORAGE_ROOT)/data_filtering/tulu_year_shards \
	SUBSET_SIZE=$(SUBSET_SIZE) \
	DATASET_SPLIT=$(DATASET_SPLIT) \
	$(ROOT)/run_all_filters.sh $(ARGS)

.PHONY: year_hist
year_hist:
	python -m data_filtering.year_histogram --shard-dir $(SHARD_DIR)

.PHONY: eval
eval:
	$(MAKE) -C $(ROOT)/filtering_eval $(ARGS)
