#!/usr/bin/env python3
"""Plot year distributions for multiple prediction files."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Set

import matplotlib.pyplot as plt

YEARS = list(range(2001, 2026))


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def normalize_temporal_type(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"explicit", "implicit"}:
        return text
    if text == "timeless":
        return "explicit"
    return "joint"


def load_subset_ids(gold_path: str, gold_field: str, split: str) -> Set[str]:
    rows = load_jsonl(gold_path)
    if split == "joint":
        return {str(row["id"]) for row in rows if gold_field in row}
    wanted = split
    ids = set()
    for row in rows:
        if gold_field not in row:
            continue
        temporal_type = normalize_temporal_type(
            row.get("sample_temporal_type", row.get("temporal_type", row.get("gold_temporal_type", "joint")))
        )
        if temporal_type == wanted:
            ids.add(str(row["id"]))
    return ids


def count_years(rows: List[dict], subset_ids: Set[str] | None = None) -> Dict[int, int]:
    counts = {y: 0 for y in YEARS}
    for row in rows:
        row_id = str(row.get("id", ""))
        if subset_ids is not None and row_id not in subset_ids:
            continue
        year = int(row.get("year", row.get("pred_year", 2001)))
        if year in counts:
            counts[year] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot year distributions across models.")
    file_storage = os.environ.get("FILE_STORAGE_ROOT", "/home/epsteine/post-training/file_storage")
    default_pred_dir = os.path.join(file_storage, "data_filtering/filtering_eval/predictions")
    default_results_dir = os.path.join(file_storage, "data_filtering/filtering_eval/results")
    parser.add_argument(
        "--pred-dir",
        default=default_pred_dir,
    )
    parser.add_argument(
        "--models",
        default="gpt-5-mini;gpt-5.2;gemini-3-pro-preview;gemini-3-flash-preview;max(gemini-3-flash-preview,gpt-5-mini);max(gemini-3-pro-preview,gpt-5.2)",
        help="Semicolon-separated model names (matching preds_<model>.jsonl).",
    )
    parser.add_argument(
        "--gold-path",
        default="",
        help="Optional categorized gold JSONL for subset plotting.",
    )
    parser.add_argument(
        "--gold-field",
        default="gold_year",
        help="Gold field in gold JSONL when --gold-path is provided.",
    )
    parser.add_argument(
        "--split",
        choices=["joint", "explicit", "implicit"],
        default="joint",
        help="Subset split when --gold-path is provided.",
    )
    parser.add_argument(
        "--out-pdf",
        default=os.path.join(default_results_dir, "year_distributions.pdf"),
    )
    args = parser.parse_args()

    subset_ids = None
    if args.gold_path:
        subset_ids = load_subset_ids(args.gold_path, args.gold_field, args.split)

    models = [m.strip() for m in args.models.split(";") if m.strip()]
    if not models:
        raise RuntimeError("No models provided.")

    nrows = len(models)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8, 1.8 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        if model.startswith("max(") and model.endswith(")"):
            inner = model[len("max(") : -1]
            base_models = [m.strip() for m in inner.split(",") if m.strip()]
            if len(base_models) < 2:
                raise RuntimeError(f"Invalid max() spec: {model}")
            base_rows = []
            for base in base_models:
                path = os.path.join(args.pred_dir, f"preds_{base}.jsonl")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing predictions: {path}")
                base_rows.append(load_jsonl(path))
            id_sets = [set(str(row["id"]) for row in rows) for rows in base_rows]
            common_ids = set.intersection(*id_sets)
            max_rows = []
            for qid in common_ids:
                years = []
                for rows in base_rows:
                    for row in rows:
                        if str(row["id"]) == qid:
                            years.append(int(row.get("year", row.get("pred_year", 2001))))
                            break
                max_rows.append({"id": qid, "year": max(years)})
            counts = count_years(max_rows, subset_ids)
        else:
            path = os.path.join(args.pred_dir, f"preds_{model}.jsonl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing predictions: {path}")
            counts = count_years(load_jsonl(path), subset_ids)
        label = model if not args.gold_path else f"{model} [{args.split}]"
        ax.bar([str(y) for y in YEARS], [counts[y] for y in YEARS], width=0.9, label=label)
        ax.set_ylim(0, 15)
        ax.legend(frameon=False, loc="upper right")

    axes[-1].set_xlabel("Year")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_pdf), exist_ok=True)
    fig.savefig(args.out_pdf)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
