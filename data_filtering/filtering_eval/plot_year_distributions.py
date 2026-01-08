#!/usr/bin/env python3
"""Plot year distributions for multiple prediction files."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

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


def count_years(rows: List[dict]) -> Dict[int, int]:
    counts = {y: 0 for y in YEARS}
    for row in rows:
        year = int(row.get("pred_year", 2001))
        if year in counts:
            counts[year] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot year distributions across models.")
    parser.add_argument(
        "--pred-dir",
        default="/home/epsteine/post-training/data_filtering/filtering_eval/predictions",
    )
    parser.add_argument(
        "--models",
        default="gpt-5-mini;gpt-5.2;gemini-3-pro-preview;gemini-3-flash-preview;max(gemini-3-flash-preview,gpt-5-mini);max(gemini-3-pro-preview,gpt-5.2)",
        help="Semicolon-separated model names (matching preds_<model>.jsonl).",
    )
    parser.add_argument(
        "--out-pdf",
        default="/home/epsteine/post-training/data_filtering/filtering_eval/results/year_distributions.pdf",
    )
    args = parser.parse_args()

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
            id_sets = [set(row["id"] for row in rows) for rows in base_rows]
            common_ids = set.intersection(*id_sets)
            max_rows = []
            for qid in common_ids:
                years = []
                for rows in base_rows:
                    for row in rows:
                        if row["id"] == qid:
                            years.append(int(row.get("pred_year", 2001)))
                            break
                max_rows.append({"pred_year": max(years)})
            counts = count_years(max_rows)
        else:
            path = os.path.join(args.pred_dir, f"preds_{model}.jsonl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing predictions: {path}")
            counts = count_years(load_jsonl(path))
        ax.bar([str(y) for y in YEARS], [counts[y] for y in YEARS], width=0.9, label=model)
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
