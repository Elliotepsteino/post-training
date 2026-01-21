#!/usr/bin/env python3
"""Plot intraclass correlation for sample-year consistency."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_samples(row: dict) -> List[Tuple[str, dict]]:
    sample_keys = [k for k in row.keys() if k.startswith("sample_") and isinstance(row.get(k), dict)]
    if sample_keys:
        return [(key, row[key]) for key in sorted(sample_keys)]
    return [("sample_1", row)]


def _to_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def sample_year(sample: dict) -> int | None:
    entities = sample.get("entities", {})
    upper_bounds: List[int] = []
    if isinstance(entities, dict):
        for ent in entities.values():
            if not isinstance(ent, dict):
                continue
            ci = ent.get("updated_confidence_interval") or ent.get("confidence_interval_95")
            if isinstance(ci, list) and len(ci) == 2:
                upper = _to_int(ci[1])
                if upper is not None:
                    upper_bounds.append(upper)
    if upper_bounds:
        return max(2001, max(upper_bounds))
    fallback = _to_int(sample.get("year")) or _to_int(sample.get("pred_year"))
    if fallback is not None:
        return max(2001, fallback)
    return None


def compute_icc(groups: Dict[str, List[int]]) -> float:
    all_values = [v for vals in groups.values() for v in vals]
    n = len(groups)
    if n < 2 or len(all_values) < 2:
        return 0.0
    grand_mean = sum(all_values) / len(all_values)
    k_bar = sum(len(vals) for vals in groups.values()) / n

    ss_between = 0.0
    ss_within = 0.0
    total_n = 0
    for vals in groups.values():
        if not vals:
            continue
        mean_i = sum(vals) / len(vals)
        ss_between += len(vals) * (mean_i - grand_mean) ** 2
        ss_within += sum((v - mean_i) ** 2 for v in vals)
        total_n += len(vals)

    if n <= 1 or total_n <= n:
        return 0.0
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (total_n - n)
    denom = ms_between + (k_bar - 1) * ms_within
    if denom <= 0:
        return 0.0
    return (ms_between - ms_within) / denom


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ICC for sample-year consistency.")
    parser.add_argument(
        "--preds",
        required=True,
        help="Grounded+updated predictions JSONL (with sample_ keys).",
    )
    parser.add_argument(
        "--out-pdf",
        required=True,
        help="Output PDF path.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.preds)
    groups: Dict[str, List[int]] = {}
    for row in rows:
        row_id = str(row.get("id"))
        values: List[int] = []
        for _, sample in iter_samples(row):
            year = sample_year(sample)
            if year is not None:
                values.append(year)
        if values:
            groups[row_id] = values

    if not groups:
        raise RuntimeError("No sample years found to compute ICC.")

    icc_value = compute_icc(groups)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Gemini 3 Flash"], [icc_value], color="#4C78A8")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Intraclass correlation (ICC)")
    ax.set_title("Sample Consistency (Search-Grounded Year)")
    ax.text(0, icc_value + 0.02, f"{icc_value:.2f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_pdf), exist_ok=True)
    fig.savefig(args.out_pdf)
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
