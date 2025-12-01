#!/usr/bin/env python3
"""
Post-processing helper that builds a year-level histogram from a run's shards.

Usage:
    python -m data_filtering.year_histogram --shard-dir path/to/year_shards_<run_id> \
        --output-file custom_year_histogram.pdf
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

CATEGORIES = [
    "general_knowledge",
    "math",
    "coding",
    "science",
    "history",
    "law",
    "finance",
    "health",
    "creative_writing",
    "multi_lingual",
    "instruction_following",
    "reasoning",
    "other",
]

CATEGORY_LABELS = {
    "general_knowledge": "General Knowledge",
    "math": "Mathematics",
    "coding": "Coding",
    "science": "Science",
    "history": "History",
    "law": "Law",
    "finance": "Finance",
    "health": "Health",
    "creative_writing": "Creative Writing",
    "multi_lingual": "Multilingual",
    "instruction_following": "Instruction Following",
    "reasoning": "Reasoning",
    "other": "Other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        required=True,
        help="Directory that contains year=*.jsonl shards (e.g., year_shards_2025-01-01_00-00Z_n50).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional path to save the year histogram (defaults to <shard-dir>/year_histogram.pdf).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_dir: Path = args.shard_dir
    if not shard_dir.exists():
        raise FileNotFoundError(f"{shard_dir} does not exist")

    year_counts, category_counts = build_counts(shard_dir)
    if not year_counts:
        raise RuntimeError(f"No year shards found under {shard_dir}")

    print("Year distribution:")
    total = sum(year_counts.values())
    for year in sorted(year_counts):
        pct = year_counts[year] * 100 / total
        print(f"  {year}: {year_counts[year]} ({pct:.1f}%)")
    print(f"Total samples: {total}")

    output_file = args.output_file or (shard_dir / "year_histogram.pdf")
    save_histogram(
        counts=year_counts,
        output_path=output_file,
        title="Tulu-3 Year Distribution",
        xlabel="Year (YY)",
        tick_labels=[str(year)[-2:] for year in sorted(year_counts)],
        x_positions=list(sorted(year_counts)),
        publication_grade=True,
        log_scale=True,
    )

    category_pdf = shard_dir / "category_histogram.pdf"
    if category_counts:
        save_histogram(
            counts=category_counts,
            output_path=category_pdf,
            title="Question Category Distribution",
            xlabel="",
            tick_labels=[CATEGORY_LABELS.get(cat, cat.title()) for cat in CATEGORIES if cat in category_counts],
            x_positions=[cat for cat in CATEGORIES if cat in category_counts],
            rotation=30,
            publication_grade=True,
        )
        print(f"Saved category histogram to {category_pdf}")


def build_counts(shard_dir: Path) -> Tuple[Counter, Counter]:
    year_counter: Counter = Counter()
    category_counter: Counter = Counter()
    for shard_path in sorted(shard_dir.glob("year=*.jsonl")):
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                year = record.get("year")
                try:
                    year = int(year)
                except (TypeError, ValueError):
                    continue
                year_counter[year] += 1
                category = (record.get("category") or "other").lower()
                if category not in CATEGORIES:
                    category = "other"
                category_counter[category] += 1
    return year_counter, category_counter


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to save a histogram image (pip install matplotlib)."
        ) from exc
    return plt


def save_histogram(
    counts: Dict,
    output_path: Path,
    title: str,
    xlabel: str,
    tick_labels: Iterable[str],
    x_positions: Iterable,
    rotation: int = 0,
    publication_grade: bool = False,
    log_scale: bool = False,
) -> None:
    plt = _import_matplotlib()

    positions = list(x_positions)
    values = [counts[pos] if pos in counts else counts.get(str(pos), 0) for pos in positions]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig_width = max(6, len(positions) * 0.55)
    fig_height = 4.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300 if publication_grade else 160)
    ax.bar(range(len(positions)), values, color="#1f77b4", edgecolor="#0f3554", linewidth=0.6)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel("Sample Count", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(list(tick_labels), rotation=rotation, ha="right" if rotation else "center", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram image to {output_path}")


if __name__ == "__main__":
    main()
