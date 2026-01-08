#!/usr/bin/env python3
"""Compare model conservatism by min/max year per question and plot counts."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Set

import matplotlib.pyplot as plt


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model conservatism.")
    parser.add_argument("--pred-dir", default="/home/epsteine/post-training/data_filtering/filtering_eval/predictions")
    parser.add_argument("--out-json", default="/home/epsteine/post-training/data_filtering/filtering_eval/results/conservatism_counts.json")
    parser.add_argument("--out-pdf", default="/home/epsteine/post-training/data_filtering/filtering_eval/results/conservatism_counts.pdf")
    parser.add_argument(
        "--max-ensemble",
        default="",
        help="Create one composite model: name:model1,model2",
    )
    parser.add_argument(
        "--max-ensembles",
        default="",
        help="Semicolon-separated composites: name:model1,model2;name2:model3,model4",
    )
    args = parser.parse_args()

    preds: Dict[str, Dict[str, int]] = {}
    for fname in sorted(os.listdir(args.pred_dir)):
        if not fname.startswith("preds_") or not fname.endswith(".jsonl"):
            continue
        model = fname.replace("preds_", "").replace(".jsonl", "")
        rows = load_jsonl(os.path.join(args.pred_dir, fname))
        preds[model] = {row["id"]: int(row["pred_year"]) for row in rows}

    if not preds:
        raise RuntimeError("No prediction files found.")

    model_ids: List[Set[str]] = [set(p.keys()) for p in preds.values()]
    common_ids = set.intersection(*model_ids)
    if not common_ids:
        raise RuntimeError("No shared ids across models.")

    def add_max_ensemble(name: str, model_list: List[str]) -> None:
        missing = [m for m in model_list if m not in preds]
        if missing:
            raise RuntimeError(f"Missing predictions for ensemble models: {', '.join(missing)}")
        ensemble_years = {}
        for qid in common_ids:
            ensemble_years[qid] = max(preds[m][qid] for m in model_list)
        preds[name] = ensemble_years

    ensemble_names: List[str] = []
    if args.max_ensemble:
        if ":" not in args.max_ensemble:
            raise RuntimeError("Use name:model1,model2 for --max-ensemble")
        name, model_spec = args.max_ensemble.split(":", 1)
        model_list = [m.strip() for m in model_spec.split(",") if m.strip()]
        add_max_ensemble(name, model_list)
        ensemble_names.append(name)

    if args.max_ensembles:
        for item in args.max_ensembles.split(";"):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise RuntimeError("Use name:model1,model2 for --max-ensembles entries")
            name, model_spec = item.split(":", 1)
            model_list = [m.strip() for m in model_spec.split(",") if m.strip()]
            add_max_ensemble(name, model_list)
            ensemble_names.append(name)

    if ensemble_names:
        common_ids = set.intersection(*[set(p.keys()) for p in preds.values()])

    most_counts = defaultdict(int)
    least_counts = defaultdict(int)

    for qid in sorted(common_ids, key=lambda x: int(x) if str(x).isdigit() else x):
        year_by_model = {m: preds[m][qid] for m in preds}
        min_year = min(year_by_model.values())
        max_year = max(year_by_model.values())
        for model, year in year_by_model.items():
            if year == min_year:
                least_counts[model] += 1
            if year == max_year:
                most_counts[model] += 1

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n": len(common_ids),
                "most_conservative": dict(most_counts),
                "least_conservative": dict(least_counts),
            },
            f,
            indent=2,
        )

    models = sorted(preds.keys())
    if ensemble_names:
        models = ensemble_names + [m for m in models if m not in ensemble_names]
    most_vals = [most_counts[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(len(models)))
    ax.bar(x, most_vals, width=0.6, label="Most conservative (max year)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("Count (ties included)")
    ax.set_title("Most-conservative counts by model")
    ax.set_ylim(0, 50)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out_pdf)
    print(f"Wrote {args.out_pdf} and {args.out_json}")


if __name__ == "__main__":
    main()
