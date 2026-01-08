#!/usr/bin/env python3
"""Sample questions across years from year-sharded datasets."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from glob import glob
from typing import Dict, List, Tuple

YEAR_RE = re.compile(r"year=(\d{4})")


def find_latest_year_shards_dir(root: str, dataset_name: str) -> str:
    pattern = os.path.join(root, "*", dataset_name, "year_shards_*")
    candidates = sorted(glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No year_shards dirs found for {dataset_name} under {root}")
    return candidates[-1]


def list_year_files(year_shards_dir: str) -> Dict[int, str]:
    files = sorted(glob(os.path.join(year_shards_dir, "year=*.jsonl")))
    year_to_file: Dict[int, str] = {}
    for path in files:
        match = YEAR_RE.search(os.path.basename(path))
        if not match:
            continue
        year = int(match.group(1))
        year_to_file[year] = path
    if not year_to_file:
        raise FileNotFoundError(f"No year=*.jsonl files found in {year_shards_dir}")
    return year_to_file


def reservoir_sample(path: str, k: int, rng: random.Random) -> List[Tuple[int, dict]]:
    """Return up to k samples (line_index, parsed_json) from a JSONL file."""
    samples: List[Tuple[int, dict]] = []
    if k <= 0:
        return samples
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len(samples) < k:
                samples.append((i, obj))
            else:
                j = rng.randint(0, i)
                if j < k:
                    samples[j] = (i, obj)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample questions across years.")
    parser.add_argument(
        "--year-shards-root",
        default="/home/epsteine/post-training/data_filtering/tulu_year_shards",
        help="Root directory containing session subdirs.",
    )
    parser.add_argument(
        "--dataset-name",
        default="allenai-tulu-3-sft-mixture",
        help="Dataset directory name under tulu_year_shards session.",
    )
    parser.add_argument(
        "--year-shards-dir",
        default="",
        help="Explicit year_shards_* directory (overrides --year-shards-root/--dataset-name).",
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--out",
        default="/home/epsteine/post-training/data_filtering/filtering_eval/data/samples.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.year_shards_dir:
        year_shards_dir = args.year_shards_dir
    else:
        year_shards_dir = find_latest_year_shards_dir(args.year_shards_root, args.dataset_name)

    year_to_file = list_year_files(year_shards_dir)
    years = sorted(year_to_file.keys())
    if not years:
        raise RuntimeError("No years found to sample from.")

    base = args.num_samples // len(years)
    remainder = args.num_samples % len(years)

    counts: Dict[int, int] = {}
    for idx, year in enumerate(years):
        counts[year] = base + (1 if idx < remainder else 0)

    samples: List[dict] = []
    for year in years:
        k = counts[year]
        if k == 0:
            continue
        for line_idx, obj in reservoir_sample(year_to_file[year], k, rng):
            question = obj.get("question", "").strip()
            answer = obj.get("answer", "").strip()
            if not question:
                continue
            sample_id = f"{year}-{line_idx}"
            placeholder_year = rng.randint(2001, 2025)
            samples.append(
                {
                    "id": sample_id,
                    "year": year,
                    "question": question,
                    "answer": answer,
                    "source_dataset": obj.get("dataset_name", ""),
                    "source_file": year_to_file[year],
                    "source_index": obj.get("sample_index", line_idx),
                    "placeholder_gold_year": placeholder_year,
                }
            )

    rng.shuffle(samples)
    samples = samples[: args.num_samples]

    # Reassign sequential IDs for simpler downstream bookkeeping.
    for idx, row in enumerate(samples, start=1):
        row["id"] = str(idx)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} samples to {args.out}")
    print(f"Year shards dir: {year_shards_dir}")


if __name__ == "__main__":
    main()
