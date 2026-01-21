#!/usr/bin/env python3
"""Plot pred - gold histograms for aggregation methods in a 2x2 grid."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_gold(path: str, field: str) -> Dict[str, int]:
    gold: Dict[str, int] = {}
    for row in load_jsonl(path):
        if field in row:
            gold[row["id"]] = int(row[field])
    return gold


def to_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aggregation deltas in a 2x2 grid.")
    parser.add_argument("--preds", required=True, help="Updated predictions JSONL")
    parser.add_argument("--gold-path", required=True, help="Gold dataset JSONL")
    parser.add_argument("--gold-field", default="gold_year", help="Gold year field")
    parser.add_argument(
        "--fields",
        default=(
            "updated_year_llm,updated_year_rank1,updated_year_rank2,"
            "updated_year_rank3,updated_year_rank4,updated_year_rank5"
        ),
        help="Comma-separated updated-year fields to plot",
    )
    parser.add_argument("--out-pdf", required=True, help="Output PDF path")
    args = parser.parse_args()

    rows = load_jsonl(args.preds)
    gold = load_gold(args.gold_path, args.gold_field)
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    if len(fields) != 6:
        raise RuntimeError("Expected exactly 6 fields for a 3x2 grid.")

    deltas_by_field: Dict[str, List[int]] = {field: [] for field in fields}
    for row in rows:
        gid = row.get("id")
        if gid not in gold:
            continue
        gold_year = gold[gid]
        for field in fields:
            value = to_int(row.get(field))
            if value is None:
                continue
            deltas_by_field[field].append(value - gold_year)

    all_deltas = [d for vals in deltas_by_field.values() for d in vals]
    if not all_deltas:
        raise RuntimeError("No deltas found for requested fields.")
    min_d = min(all_deltas)
    max_d = max(all_deltas)
    bins = [x - 0.5 for x in range(min_d, max_d + 2)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes_flat = [ax for row in axes for ax in row]
    for ax, field in zip(axes_flat, fields):
        vals = deltas_by_field.get(field, [])
        ax.hist(vals, bins=bins, color="#4C78A8", alpha=0.8)
        ax.axvline(0, color="black", linewidth=1)
        title = field.replace("updated_year_", "").replace("_", " ").title()
        if title == "Llm":
            title = "LLM Aggregation"
        ax.set_title(title)
        ax.set_xlabel("Predicted year - Ground Truth year")
        ax.set_ylabel("Count")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_pdf), exist_ok=True)
    fig.savefig(args.out_pdf)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
