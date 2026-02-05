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


def load_gold_from_predictions(pred_path: str) -> Dict[str, int]:
    gold: Dict[str, int] = {}
    for row in load_jsonl(pred_path):
        year = row.get("year", row.get("pred_year"))
        if year is None:
            continue
        gold[row["id"]] = int(year)
    return gold


def load_gold_from_samples(samples_path: str, field: str) -> Dict[str, int]:
    gold: Dict[str, int] = {}
    for row in load_jsonl(samples_path):
        if field not in row:
            continue
        gold[row["id"]] = int(row[field])
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
        f.write("        Model & Exact acc & Conservative acc & Weighted acc \\\\\n")
        f.write("        \\midrule\n")
        for row in rows:
            f.write(
                f"        {row['model']} & {row['exact']:.2f} & {row['conservative']:.2f} & {row['weighted']:.2f} \\\\\n"
            )
        f.write("        \\bottomrule\n")
        f.write("    \\end{tabular}\n")
        f.write("    \\caption{Filtering label accuracy on 50 questions. Exact is year match; conservative is predicted year $\\ge$ gold year; weighted is mean of the two.}\n")
        f.write("    \\label{tab:filtering-eval}\n")
        f.write("\\end{table}\n")


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
        gold = load_gold_from_samples(gold_source, args.gold_field)
    else:
        gold_pred_path = os.path.join(args.pred_dir, f"preds_{args.gold_from_model}.jsonl")
        gold = load_gold_from_predictions(gold_pred_path)

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
        exact, cons, weighted, total = score(preds, gold)
        results.append(
            {
                "model": model,
                "exact": exact * 100,
                "conservative": cons * 100,
                "weighted": weighted * 100,
                "n": total,
            }
        )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"gold_model": args.gold_from_model, "results": results}, f, indent=2)

    write_tex_table(args.out_tex, results)
    print(f"Wrote {args.out_json} and {args.out_tex}")


if __name__ == "__main__":
    main()
