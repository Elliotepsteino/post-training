#!/usr/bin/env python3
"""Create a gold dataset with question + gold year from model predictions."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create gold dataset from predictions.")
    parser.add_argument("--samples", default="/home/epsteine/post-training/data_filtering/filtering_eval/data/samples.jsonl")
    parser.add_argument("--preds", required=True, help="Predictions JSONL (e.g., preds_gpt-5.2-pro.jsonl)")
    parser.add_argument("--out", default="/home/epsteine/post-training/data_filtering/filtering_eval/data/gold_dataset_dev.jsonl")
    parser.add_argument("--gold-model", default="gpt-5.2")
    parser.add_argument("--sequential-ids", action="store_true", help="Rewrite ids as 1..N")
    args = parser.parse_args()

    preds = {row["id"]: int(row["pred_year"]) for row in load_jsonl(args.preds)}
    samples = load_jsonl(args.samples)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        out_rows = []
        for row in samples:
            gold_year = preds.get(row["id"])
            if gold_year is None:
                continue
            out_rows.append(
                {
                    "id": row["id"],
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "gold_year": gold_year,
                    "gold_model": args.gold_model,
                }
            )

        if args.sequential_ids:
            for idx, row in enumerate(out_rows, start=1):
                row["id"] = str(idx)

        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote gold dataset to {args.out}")


if __name__ == "__main__":
    main()
