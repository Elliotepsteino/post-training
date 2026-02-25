#!/usr/bin/env python3
"""Score predictions against gold labels and produce summary tables."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def canonical_temporal_type(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"explicit", "implicit"}:
        return text
    if text == "timeless":
        return "explicit"
    return "unknown"


def load_gold_from_predictions(pred_path: str) -> Dict[str, dict]:
    gold: Dict[str, dict] = {}
    for row in load_jsonl(pred_path):
        year = row.get("year", row.get("pred_year"))
        if year is None:
            continue
        gold[row["id"]] = {
            "year": int(year),
            "sample_temporal_type": canonical_temporal_type(row.get("sample_temporal_type", "unknown")),
        }
    return gold


def load_gold_from_samples(samples_path: str, field: str) -> Dict[str, dict]:
    gold: Dict[str, dict] = {}
    for row in load_jsonl(samples_path):
        if field not in row:
            continue
        temporal_type = canonical_temporal_type(
            row.get("sample_temporal_type", row.get("temporal_type", row.get("gold_temporal_type", "unknown")))
        )
        gold[row["id"]] = {
            "year": int(row[field]),
            "sample_temporal_type": temporal_type,
        }
    return gold


def score(preds: Dict[str, int], gold: Dict[str, int]) -> Tuple[float, float, float, int]:
    total = 0
    exact = 0
    conservative = 0
    for k, gold_year in gold.items():
        if k not in preds:
            continue
        total += 1
        pred_year = preds[k]
        if pred_year == gold_year:
            exact += 1
        if pred_year >= gold_year:
            conservative += 1
    if total == 0:
        return 0.0, 0.0, 0.0, 0
    exact_acc = exact / total
    conservative_acc = conservative / total
    weighted = 0.5 * (exact_acc + conservative_acc)
    return exact_acc, conservative_acc, weighted, total


def write_tex_table(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("    \\centering\n")
        f.write("    \\begin{tabular}{lccc}\n")
        f.write("        \\toprule\n")
        f.write("        Model & Joint (weighted) & Explicit (weighted) & Implicit (weighted) \\\\\n")
        f.write("        \\midrule\n")
        for row in rows:
            splits = row.get("splits", {})
            joint = splits.get("joint", {}).get("weighted", 0.0)
            explicit = splits.get("explicit", {}).get("weighted", 0.0)
            implicit = splits.get("implicit", {}).get("weighted", 0.0)
            f.write(f"        {row['model']} & {joint:.2f} & {explicit:.2f} & {implicit:.2f} \\\\\n")
        f.write("        \\bottomrule\n")
        f.write("    \\end{tabular}\n")
        f.write(
            "    \\caption{Filtering accuracy by subset. Weighted accuracy is the mean of exact and conservative accuracy.}\n"
        )
        f.write("    \\label{tab:filtering-eval}\n")
        f.write("\\end{table}\n")


def select_gold_subset(gold_meta: Dict[str, dict], split: str) -> Dict[str, int]:
    if split == "joint":
        return {gid: data["year"] for gid, data in gold_meta.items()}
    if split == "explicit":
        return {gid: data["year"] for gid, data in gold_meta.items() if data.get("sample_temporal_type") == "explicit"}
    if split == "implicit":
        return {gid: data["year"] for gid, data in gold_meta.items() if data.get("sample_temporal_type") == "implicit"}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score filtering predictions.")
    file_storage = os.environ.get("FILE_STORAGE_ROOT", "/home/epsteine/post-training/file_storage")
    default_pred_dir = os.path.join(file_storage, "data_filtering/filtering_eval/predictions")
    default_results_dir = os.path.join(file_storage, "data_filtering/filtering_eval/results")
    parser.add_argument("--samples", default="/home/epsteine/post-training/data_filtering/filtering_eval/data/samples.jsonl")
    parser.add_argument("--gold-from-model", default="gpt-5.2-pro")
    parser.add_argument(
        "--gold-field",
        default="",
        help="Field in samples.jsonl to use as gold labels (overrides --gold-from-model).",
    )
    parser.add_argument(
        "--gold-path",
        default="",
        help="Path to gold dataset JSONL (overrides --samples for gold labels).",
    )
    parser.add_argument("--pred-dir", default=default_pred_dir)
    parser.add_argument("--out-json", default=os.path.join(default_results_dir, "summary.json"))
    parser.add_argument("--out-tex", default=os.path.join(default_results_dir, "filtering_eval_table.tex"))
    args = parser.parse_args()

    if args.gold_field:
        gold_source = args.gold_path or args.samples
        gold_meta = load_gold_from_samples(gold_source, args.gold_field)
    else:
        gold_pred_path = os.path.join(args.pred_dir, f"preds_{args.gold_from_model}.jsonl")
        gold_meta = load_gold_from_predictions(gold_pred_path)

    results = []
    for fname in sorted(os.listdir(args.pred_dir)):
        if not fname.endswith(".jsonl"):
            continue
        model = fname.replace("preds_", "").replace(".jsonl", "")
        preds = {}
        for row in load_jsonl(os.path.join(args.pred_dir, fname)):
            year = row.get("year", row.get("pred_year"))
            if year is None:
                continue
            preds[row["id"]] = int(year)

        split_scores = {}
        for split in ("joint", "explicit", "implicit"):
            split_gold = select_gold_subset(gold_meta, split)
            exact, cons, weighted, total = score(preds, split_gold)
            split_scores[split] = {
                "exact": exact * 100,
                "conservative": cons * 100,
                "weighted": weighted * 100,
                "n": total,
            }

        results.append(
            {
                "model": model,
                "exact": split_scores["joint"]["exact"],
                "conservative": split_scores["joint"]["conservative"],
                "weighted": split_scores["joint"]["weighted"],
                "n": split_scores["joint"]["n"],
                "splits": split_scores,
            }
        )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"gold_model": args.gold_from_model, "results": results}, f, indent=2)

    write_tex_table(args.out_tex, results)
    print(f"Wrote {args.out_json} and {args.out_tex}")


if __name__ == "__main__":
    main()
