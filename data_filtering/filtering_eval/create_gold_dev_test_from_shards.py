#!/usr/bin/env python3
"""Create dev/test gold datasets from year shards."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Dict, List, Tuple


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def year_files(shards_dir: str) -> List[Tuple[int, str]]:
    files: List[Tuple[int, str]] = []
    for name in os.listdir(shards_dir):
        match = re.match(r"year=(\d{4})\.jsonl$", name)
        if match:
            files.append((int(match.group(1)), os.path.join(shards_dir, name)))
    return sorted(files, key=lambda item: item[0])


def to_gold_row(row: dict, gold_model: str) -> dict:
    return {
        "id": "",
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "gold_year": int(row["year"]),
        "gold_model": gold_model,
        "entities": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create gold dev/test datasets from year shards.")
    parser.add_argument("--shards-dir", required=True, help="Directory with year=YYYY.jsonl files")
    parser.add_argument("--dev-out", required=True, help="Output path for dev JSONL")
    parser.add_argument("--test-out", required=True, help="Output path for test JSONL")
    parser.add_argument("--per-year", type=int, default=7, help="Samples per year for dev set")
    parser.add_argument("--test-size", type=int, default=130, help="Total samples for test set")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--gold-model", default="", help="Override gold_model value")
    args = parser.parse_args()

    shards_base = os.path.basename(os.path.normpath(args.shards_dir))
    gold_model = args.gold_model or shards_base

    dev_rows: List[dict] = []
    dev_questions = set()
    remainder: List[dict] = []

    for year, path in year_files(args.shards_dir):
        rows = load_jsonl(path)
        rng = random.Random(args.seed + year)
        rng.shuffle(rows)
        take = min(args.per_year, len(rows))
        picked = rows[:take]
        leftover = rows[take:]
        for row in picked:
            gold_row = to_gold_row(row, gold_model)
            dev_rows.append(gold_row)
            dev_questions.add(gold_row["question"])
        remainder.extend(leftover)

    filtered_remainder = [
        row for row in remainder if row.get("question", "") not in dev_questions
    ]
    if len(filtered_remainder) < args.test_size:
        raise SystemExit(
            f"Not enough remaining rows for test set: {len(filtered_remainder)} < {args.test_size}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(filtered_remainder)
    test_rows = [to_gold_row(row, gold_model) for row in filtered_remainder[: args.test_size]]

    for idx, row in enumerate(dev_rows, start=1):
        row["id"] = str(idx)
    for idx, row in enumerate(test_rows, start=1):
        row["id"] = str(idx)

    os.makedirs(os.path.dirname(args.dev_out), exist_ok=True)
    with open(args.dev_out, "w", encoding="utf-8") as f:
        for row in dev_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(args.test_out, "w", encoding="utf-8") as f:
        for row in test_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote dev: {args.dev_out} ({len(dev_rows)})")
    print(f"Wrote test: {args.test_out} ({len(test_rows)})")


if __name__ == "__main__":
    main()
