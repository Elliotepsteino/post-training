"""
utilities for assembling mixed-year training datasets from the year-sharded
jsonl files produced by the filtering pipeline.
"""

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset

__all__ = ["load_year_mixture", "create_year_dataset"]

SESSION_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_.+PT$")
DEFAULT_SHARD_ROOT = Path(__file__).resolve().parent / "data_filtering" / "tulu_year_shards"
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "datasets"


def _extract_question_response(example: Dict[str, Any]) -> Dict[str, str]:
    """Normalize schema differences by projecting onto question/response."""
    question = (
        example.get("question")
        or example.get("prompt")
        or example.get("input")
        or ""
    )
    answer = (
        example.get("answer")
        or example.get("response")
        or example.get("output")
        or ""
    )
    return {"question": question, "response": answer}


def load_year_mixture(
    shard_dir: str,
    max_year: int,
    seed: int,
    max_samples: int | None,
) -> Dataset:
    """
    Load and shuffle all shards with year <= max_year from shard_dir.

    The resulting dataset only contains `question` and `response` columns and
    can optionally be truncated to `max_samples` examples.
    """
    shard_path = Path(shard_dir)
    if not shard_path.exists():
        raise FileNotFoundError(f"Shard directory not found: {shard_dir}")

    data_files: List[str] = []
    for file in sorted(shard_path.glob("year=*.jsonl")):
        year_token = file.stem.split("_", 1)[0]  # e.g., "year=2007"
        try:
            year = int(year_token.split("=")[-1])
        except (ValueError, IndexError):
            continue
        if year <= max_year:
            data_files.append(str(file))

    if not data_files:
        raise FileNotFoundError(f"No shards with year <= {max_year} found in {shard_dir}")

    ds = load_dataset("json", data_files=data_files, split="train")
    cols_to_remove = [c for c in ds.column_names if c not in {"question", "answer", "response"}]
    ds = ds.map(_extract_question_response, remove_columns=cols_to_remove, desc="Extracting Q/A pairs")
    cols_to_keep = {"question", "response"}
    extra_cols = [c for c in ds.column_names if c not in cols_to_keep]
    if extra_cols:
        ds = ds.remove_columns(extra_cols)

    ds = ds.shuffle(seed=seed)
    if max_samples is not None and max_samples > 0 and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


def _find_latest_child(parent: Path, predicate) -> Path:
    candidates = sorted([p for p in parent.iterdir() if predicate(p)])
    if not candidates:
        raise FileNotFoundError(f"No matching directories under {parent}")
    return candidates[-1]


def _resolve_session(shard_root: Path, session_stamp: Optional[str]) -> Path:
    if session_stamp:
        session_dir = shard_root / session_stamp
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Session '{session_stamp}' not found in {shard_root}")
        return session_dir
    try:
        return _find_latest_child(
            shard_root,
            lambda p: p.is_dir() and SESSION_PATTERN.match(p.name) is not None,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No timestamped PT session directories under {shard_root}. "
            "Specify --session-stamp or create a session with run_all_filters.sh."
        ) from exc


def _resolve_task_dir(session_dir: Path, task: str) -> Path:
    task_dir = session_dir / task
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task '{task}' not found under {session_dir}")
    return task_dir


def _resolve_run_dir(task_dir: Path, run_name: Optional[str]) -> Path:
    if run_name:
        run_dir = task_dir / run_name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run '{run_name}' not found in {task_dir}")
        return run_dir
    return _find_latest_child(task_dir, lambda p: p.is_dir() and p.name.startswith("year_shards_"))


def _add_messages_column(ds: Dataset) -> Dataset:
    def to_messages(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": example.get("question", "")},
                {"role": "assistant", "content": example.get("response", "")},
            ]
        }

    return ds.map(to_messages, desc="Formatting chat messages")


def create_year_dataset(
    task: str,
    max_year: int,
    shard_root: Path = DEFAULT_SHARD_ROOT,
    session_stamp: Optional[str] = None,
    run_name: Optional[str] = None,
    output_dir: Path = DEFAULT_DATASET_ROOT,
    dataset_name: Optional[str] = None,
    seed: int = 17,
    max_samples: Optional[int] = None,
) -> Path:
    session_dir = _resolve_session(shard_root, session_stamp)
    task_dir = _resolve_task_dir(session_dir, task)
    run_dir = _resolve_run_dir(task_dir, run_name)

    ds = load_year_mixture(str(run_dir), max_year=max_year, seed=seed, max_samples=max_samples)
    ds = _add_messages_column(ds)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_task = task.replace("/", "-")
    slug = dataset_name or f"{safe_task}_upto_{max_year}"
    slug += f"_n{len(ds)}"
    output_file = output_dir / f"{slug}.jsonl"
    ds.to_json(str(output_file))
    return output_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export filtered year shards into a HF-friendly JSONL dataset.")
    parser.add_argument("--task", required=True, help="Dataset slug, e.g., allenai-tulu-3-sft-mixture")
    parser.add_argument("--max-year", type=int, required=True, help="Include shards up to and including this year.")
    parser.add_argument(
        "--shard-root",
        type=Path,
        default=DEFAULT_SHARD_ROOT,
        help="Root directory holding the timestamped PT sessions.",
    )
    parser.add_argument(
        "--session-stamp",
        help="Specific PT session (e.g., 2026-01-05_14-31PT). Defaults to the latest available session.",
    )
    parser.add_argument(
        "--run-name",
        help="Explicit year_shards directory to use (defaults to the latest under the task).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Where to write the merged JSONL dataset.",
    )
    parser.add_argument("--dataset-name", help="Custom name for the output file (without extension).")
    parser.add_argument("--seed", type=int, default=17, help="Shuffle seed before optional truncation.")
    parser.add_argument("--max-samples", type=int, help="Optional cap on the merged dataset size.")
    return parser.parse_args()


def _main():
    args = _parse_args()
    output_path = create_year_dataset(
        task=args.task,
        max_year=args.max_year,
        shard_root=args.shard_root,
        session_stamp=args.session_stamp,
        run_name=args.run_name,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        seed=args.seed,
        max_samples=args.max_samples if args.max_samples and args.max_samples > 0 else None,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    _main()
