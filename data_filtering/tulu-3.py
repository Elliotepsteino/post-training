#!/usr/bin/env python3
"""
Utilities for classifying the allenai/tulu-3-sft-mixture by knowledge year.

The script runs an OpenAI Batch job with GPT-5-mini to label the minimum year
(between 2001 and 2025) that contains the knowledge required to answer each
question/answer pair. Results are saved as one JSONL shard per year together
with helper utilities for loading and shuffling data up to a target cutoff.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import Dataset, load_dataset
from openai import OpenAI

YEARS = list(range(2001, 2026))
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
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_SUBSET = 50
DEFAULT_MAX_BATCH_SIZE = 20000
POLLABLE_STATUSES = {"validating", "in_progress", "running", "queued", "finalizing"}
DEFAULT_DATASET = "allenai/tulu-3-sft-mixture"
DEFAULT_SPLIT = "train"
VARIANT_SFT = "sft"
VARIANT_DPO = "dpo"
VARIANT_RLVR = "rlvr"
DATASET_VARIANTS = {
    "allenai/llama-3.1-tulu-3-8b-preference-mixture": VARIANT_DPO,
    "allenai/RLVR-GSM": VARIANT_RLVR,
    "allenai/RLVR-MATH": VARIANT_RLVR,
    "allenai/RLVR-IFeval": VARIANT_RLVR,
}
VARIANT_NOTES = {
    VARIANT_SFT: (
        "These are supervised instruction-tuning pairs: treat the question as the user prompt "
        "and the answer as the assistant's canonical reply."
    ),
    VARIANT_DPO: (
        "These samples include BOTH a preferred answer and a rejected answer. Consider every fact "
        "mentioned in either answer when deciding the minimum safe year."
    ),
    VARIANT_RLVR: (
        "These samples bundle evaluation prompts, solution rationales, and sometimes explicit constraints. "
        "Treat the full message history plus the provided ground-truth/constraint text as a single answer bundle."
    ),
}


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, prompt: int = 0, completion: int = 0, total: int | None = None) -> None:
        self.prompt_tokens += max(0, prompt or 0)
        self.completion_tokens += max(0, completion or 0)
        if total is None:
            total = (prompt or 0) + (completion or 0)
        self.total_tokens += max(0, total or 0)

    def merge(self, other: "TokenUsage") -> None:
        self.add(other.prompt_tokens, other.completion_tokens, other.total_tokens)

    def as_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
}

BATCH_DISCOUNT = 0.5  # OpenAI halves token prices for Batch requests at the time of writing.

MODEL_PRICING_PER_MILLION = {
    # Update these defaults if OpenAI revises pricing for the referenced models.
    "gpt-5-mini": {
        "prompt": 0.25,  # USD per 1M prompt tokens
        "completion": 2.00,  # USD per 1M completion tokens
    },
    "gpt-5.2": {
        "prompt": 1.75,
        "completion": 14.00,
    },
    "gpt-5.2-pro": {
        "prompt": 21.00,
        "completion": 168.00,
    },
}


def normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict) and "text" in chunk:
                parts.append(chunk.get("text", ""))
            else:
                parts.append(str(chunk))
        return " ".join(parts)
    if isinstance(content, dict):
        return content.get("text") or ""
    return str(content)


def flatten_messages(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for msg in messages or []:
        role = msg.get("role") or "unknown"
        text = normalize_content(msg.get("content")).strip()
        if not text:
            continue
        lines.append(f"[{role}] {text}")
    return "\n\n".join(lines).strip()


def extract_generic_question_answer(row: Dict[str, Any]) -> Tuple[str, str]:
    if "messages" in row and row["messages"]:
        user_turns: List[str] = []
        assistant_turns: List[str] = []
        for msg in row["messages"]:
            role = msg.get("role") or ""
            text = normalize_content(msg.get("content"))
            if role in {"user", "instruction"}:
                user_turns.append(text)
            elif role == "assistant":
                assistant_turns.append(text)
        question = user_turns[-1] if user_turns else ""
        answer = assistant_turns[-1] if assistant_turns else ""
        return question.strip(), answer.strip()

    if "prompt" in row and "response" in row:
        return row["prompt"].strip(), row["response"].strip()

    question = (
        row.get("question")
        or row.get("input")
        or row.get("instructions")
        or row.get("source")
        or ""
    )
    answer = row.get("answer") or row.get("output") or row.get("target") or ""
    return str(question).strip(), str(answer).strip()


def extract_dpo_question_answer(row: Dict[str, Any]) -> Tuple[str, str]:
    def split_turns(messages: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        user_turns: List[str] = []
        assistant_turns: List[str] = []
        for msg in messages:
            role = (msg.get("role") or "").lower()
            text = normalize_content(msg.get("content")).strip()
            if not text:
                continue
            if role == "assistant":
                assistant_turns.append(text)
            else:
                user_turns.append(text)
        return user_turns, assistant_turns

    prompt = (row.get("prompt") or "").strip()
    user_chosen, assistant_chosen = split_turns(row.get("chosen") or [])
    user_rejected, assistant_rejected = split_turns(row.get("rejected") or [])

    question_parts: List[str] = []
    if prompt:
        question_parts.append(prompt)
    seen_user_texts = set()
    for part in user_chosen + user_rejected:
        if part not in seen_user_texts:
            question_parts.append(part)
            seen_user_texts.add(part)
    question = "\n\n".join(part for part in question_parts if part).strip()

    answer_sections: List[str] = []
    if assistant_chosen:
        answer_sections.append("preferred answer:\n" + "\n\n".join(assistant_chosen))
    if assistant_rejected:
        answer_sections.append("rejected answer:\n" + "\n\n".join(assistant_rejected))
    answer = "\n\n---\n\n".join(answer_sections).strip()
    return question, answer


def extract_rlvr_question_answer(row: Dict[str, Any]) -> Tuple[str, str]:
    question = flatten_messages(row.get("messages") or [])
    answer_parts: List[str] = []
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, str):
        answer_parts.append(ground_truth.strip())
    dataset_name = row.get("dataset")
    if isinstance(dataset_name, str) and dataset_name:
        answer_parts.append(f"[dataset] {dataset_name}")
    constraint = row.get("constraint")
    if isinstance(constraint, str) and constraint.strip():
        answer_parts.append(f"[constraint] {constraint.strip()}")
    return question, "\n\n".join(part for part in answer_parts if part).strip()


DATASET_EXTRACTORS = {
    "allenai/llama-3.1-tulu-3-8b-preference-mixture": extract_dpo_question_answer,
    "allenai/RLVR-GSM": extract_rlvr_question_answer,
    "allenai/RLVR-MATH": extract_rlvr_question_answer,
    "allenai/RLVR-IFeval": extract_rlvr_question_answer,
}


def slugify_dataset(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "dataset").lower())
    slug = slug.strip("-")
    return slug or "dataset"


@dataclass
class SampleRecord:
    """Lightweight container for the subset we send to the Batch API."""

    sample_index: int
    dataset_row: Dict[str, Any]
    question: str
    answer: str
    dataset_name: str
    dataset_variant: str

    @property
    def custom_id(self) -> str:
        return f"sample-{self.sample_index}"


@dataclass
class PendingBatch:
    run_id: str
    samples: List[SampleRecord]
    batch_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset-size",
        type=int,
        default=DEFAULT_SUBSET,
        help="How many samples to classify (default: 50).",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
        help="Maximum number of samples per OpenAI batch submission (default: 20000). Larger subsets are split "
        "into multiple batches that run in parallel.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET,
        help=f"Hugging Face dataset to sample from (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--dataset-split",
        default=DEFAULT_SPLIT,
        help=f"Dataset split to load (default: {DEFAULT_SPLIT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "tulu_year_shards",
        help="Where to save per-year JSONL files and batch artifacts.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model to use (default: GPT-5-mini).",
    )
    parser.add_argument(
        "--completion-window",
        default="24h",
        help="Batch completion window requested from OpenAI.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=20,
        help="Seconds between batch status polls.",
    )
    parser.add_argument(
        "--resume-batch-id",
        default=None,
        help="Skip submission and resume polling an existing batch id.",
    )
    parser.add_argument(
        "--use-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the OpenAI Batch API (default). Pass --no-use-batch for live synchronous calls.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Seed for deterministic subset selection and shuffling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.use_batch and args.resume_batch_id:
        raise ValueError("--resume-batch-id is only valid when --use-batch is enabled.")
    if args.max_batch_size <= 0:
        raise ValueError("--max-batch-size must be positive.")

    samples = build_subset(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        subset_size=args.subset_size,
        seed=args.seed,
    )
    if not samples:
        raise ValueError("No samples extracted from the dataset.")

    client = OpenAI()
    dataset_slug = slugify_dataset(args.dataset_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%MZ")
    max_batch_size = max(1, args.max_batch_size)

    if len(samples) <= max_batch_size:
        grouped, run_id, usage_stats = run_single_batch(
            samples=samples,
            dataset_slug=dataset_slug,
            output_dir=output_dir,
            args=args,
            client=client,
            timestamp=timestamp,
        )
    else:
        grouped, run_id, usage_stats = run_multi_batch(
            samples=samples,
            dataset_slug=dataset_slug,
            output_dir=output_dir,
            args=args,
            client=client,
            timestamp=timestamp,
            max_batch_size=max_batch_size,
        )

    shard_dir = save_year_shards(
        grouped_records=grouped,
        output_dir=output_dir,
        run_id=run_id,
        timestamp=timestamp,
    )
    print_summary(grouped_records=grouped, shard_dir=shard_dir)
    log_usage_summary(
        usage=usage_stats,
        model=args.model,
        output_dir=output_dir,
        run_id=run_id,
        batch_discount_applied=bool(args.use_batch),
    )


def build_subset(
    dataset_name: str,
    dataset_split: str,
    subset_size: int,
    seed: int,
) -> List[SampleRecord]:
    ds = load_dataset(dataset_name, split=dataset_split)
    total = len(ds)
    if subset_size > total:
        subset_size = total
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    chosen = sorted(indices[:subset_size])
    samples: List[SampleRecord] = []
    extractor = DATASET_EXTRACTORS.get(dataset_name, extract_generic_question_answer)
    dataset_variant = DATASET_VARIANTS.get(dataset_name, VARIANT_SFT)
    for idx in chosen:
        row = ds[idx]
        question, answer = extractor(row)
        if not question or not answer:
            continue
        samples.append(SampleRecord(idx, row, question, answer, dataset_name, dataset_variant))
    return samples


def chunk_samples(samples: List[SampleRecord], max_batch_size: int) -> List[List[SampleRecord]]:
    return [samples[i : i + max_batch_size] for i in range(0, len(samples), max_batch_size)]


def run_single_batch(
    samples: List[SampleRecord],
    dataset_slug: str,
    output_dir: Path,
    args: argparse.Namespace,
    client: OpenAI,
    timestamp: str,
) -> Tuple[Dict[int, List[Dict[str, Any]]], str, TokenUsage]:
    run_id = f"{dataset_slug}_{timestamp}_n{len(samples)}"
    (
        requests,
        batch_input_path,
        request_count,
        _,
        _,
    ) = prepare_batch_payload(
        samples=samples,
        model=args.model,
        output_dir=output_dir,
        dataset_slug=dataset_slug,
        run_id_override=run_id,
        timestamp_override=timestamp,
    )

    if args.use_batch:
        batch = maybe_submit_batch(
            client=client,
            batch_input_path=batch_input_path,
            model=args.model,
            sample_count=request_count,
            output_dir=output_dir,
            completion_window=args.completion_window,
            resume_batch_id=args.resume_batch_id,
            run_id=run_id,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
        )
        print(f"Polling batch {batch.id} (status: {batch.status})")
        batch = wait_for_batch_completion(
            client=client,
            batch_id=batch.id,
            poll_interval=args.poll_interval,
        )
        print(f"Batch finished with status {batch.status}")
        if batch.status != "completed":
            raise RuntimeError(f"Batch failed with status: {batch.status}")

        output_path = download_batch_output(
            client=client,
            file_id=batch.output_file_id,
            output_dir=output_dir,
            run_id=run_id,
        )
    else:
        output_path = run_live_classification(
            client=client,
            requests=requests,
            output_dir=output_dir,
            batch_input_path=batch_input_path,
            model=args.model,
            run_id=run_id,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
        )

    year_map, usage_stats = parse_batch_predictions(output_path=output_path)
    grouped = attach_years(samples=samples, year_map=year_map)
    return grouped, run_id, usage_stats


def run_multi_batch(
    samples: List[SampleRecord],
    dataset_slug: str,
    output_dir: Path,
    args: argparse.Namespace,
    client: OpenAI,
    timestamp: str,
    max_batch_size: int,
) -> Tuple[Dict[int, List[Dict[str, Any]]], str, TokenUsage]:
    if args.resume_batch_id:
        raise ValueError("--resume-batch-id is not supported when multiple batches are required.")

    total = len(samples)
    final_run_id = f"{dataset_slug}_{timestamp}_n{total}"
    chunks = chunk_samples(samples, max_batch_size)
    chunk_count = len(chunks)
    pending_jobs: List[PendingBatch] = []
    chunk_infos: List[Dict[str, Any]] = []
    aggregated: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    usage_totals = TokenUsage()

    if args.use_batch:
        for idx, chunk in enumerate(chunks):
            chunk_run_id = f"{final_run_id}_part{idx + 1}of{chunk_count}_n{len(chunk)}"
            (
                _,
                batch_input_path,
                request_count,
                _,
                _,
            ) = prepare_batch_payload(
                samples=chunk,
                model=args.model,
                output_dir=output_dir,
                dataset_slug=dataset_slug,
                run_id_override=chunk_run_id,
                timestamp_override=timestamp,
            )
            batch = maybe_submit_batch(
                client=client,
                batch_input_path=batch_input_path,
                model=args.model,
                sample_count=request_count,
                output_dir=output_dir,
                completion_window=args.completion_window,
                resume_batch_id=None,
                run_id=chunk_run_id,
                dataset_name=args.dataset_name,
                dataset_split=args.dataset_split,
            )
            print(f"Submitted chunk {idx + 1}/{chunk_count}: batch {batch.id}")
            pending_jobs.append(PendingBatch(run_id=chunk_run_id, samples=chunk, batch_id=batch.id))
            chunk_infos.append(
                {"run_id": chunk_run_id, "batch_id": batch.id, "subset_size": len(chunk)}
            )

        for job in pending_jobs:
            print(f"Polling batch {job.batch_id} (chunk {job.run_id})")
            batch = wait_for_batch_completion(
                client=client,
                batch_id=job.batch_id,
                poll_interval=args.poll_interval,
            )
            print(f"Batch {job.batch_id} finished with status {batch.status}")
            if batch.status != "completed":
                raise RuntimeError(f"Batch {job.batch_id} failed with status: {batch.status}")
            output_path = download_batch_output(
                client=client,
                file_id=batch.output_file_id,
                output_dir=output_dir,
                run_id=job.run_id,
            )
            year_map, usage_stats = parse_batch_predictions(output_path=output_path)
            chunk_grouped = attach_years(samples=job.samples, year_map=year_map)
            merge_grouped(aggregated, chunk_grouped)
            usage_totals.merge(usage_stats)
    else:
        for idx, chunk in enumerate(chunks):
            chunk_run_id = f"{final_run_id}_part{idx + 1}of{chunk_count}_n{len(chunk)}"
            (
                requests,
                batch_input_path,
                _,
                _,
                _,
            ) = prepare_batch_payload(
                samples=chunk,
                model=args.model,
                output_dir=output_dir,
                dataset_slug=dataset_slug,
                run_id_override=chunk_run_id,
                timestamp_override=timestamp,
            )
            output_path = run_live_classification(
                client=client,
                requests=requests,
                output_dir=output_dir,
                batch_input_path=batch_input_path,
                model=args.model,
                run_id=chunk_run_id,
                dataset_name=args.dataset_name,
                dataset_split=args.dataset_split,
            )
            year_map, usage_stats = parse_batch_predictions(output_path=output_path)
            chunk_grouped = attach_years(samples=chunk, year_map=year_map)
            merge_grouped(aggregated, chunk_grouped)
            usage_totals.merge(usage_stats)
            chunk_infos.append({"run_id": chunk_run_id, "batch_id": None, "subset_size": len(chunk)})

    write_combined_metadata(
        output_dir=output_dir,
        final_run_id=final_run_id,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        total_samples=total,
        chunk_infos=chunk_infos,
    )
    return aggregated, final_run_id, usage_totals


def merge_grouped(
    aggregated: Dict[int, List[Dict[str, Any]]],
    chunk_grouped: Dict[int, List[Dict[str, Any]]],
) -> None:
    for year, records in chunk_grouped.items():
        aggregated[year].extend(records)


def prepare_batch_payload(
    samples: List[SampleRecord],
    model: str,
    output_dir: Path,
    dataset_slug: str,
    run_id_override: str | None = None,
    timestamp_override: str | None = None,
) -> Tuple[List[Dict[str, Any]], Path, int, str, str]:
    requests = [build_batch_request(sample, model=model) for sample in samples]
    if not requests:
        raise ValueError("No valid samples with extracted question/answer pairs.")
    timestamp = timestamp_override or datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%MZ")
    sample_count = len(requests)
    run_id = run_id_override or f"{dataset_slug}_{timestamp}_n{sample_count}"
    batch_input_path = output_dir / f"batch_input_{run_id}.jsonl"
    write_jsonl(requests, batch_input_path)
    print(f"Wrote {sample_count} batch requests to {batch_input_path}")
    return requests, batch_input_path, sample_count, timestamp, run_id


def maybe_submit_batch(
    client: OpenAI,
    batch_input_path: Path,
    model: str,
    sample_count: int,
    output_dir: Path,
    completion_window: str,
    resume_batch_id: str | None,
    run_id: str,
    dataset_name: str,
    dataset_split: str,
):
    if resume_batch_id:
        print(f"Resuming existing batch {resume_batch_id}")
        return client.batches.retrieve(resume_batch_id)

    with batch_input_path.open("rb") as handle:
        input_file = client.files.create(file=handle, purpose="batch")

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata={"description": f"year classification subset ({dataset_name})"},
    )
    metadata = {
        "mode": "batch",
        "run_id": run_id,
        "batch_id": batch.id,
        "input_file_id": input_file.id,
        "input_file_path": str(batch_input_path),
        "model": model,
        "subset_size": sample_count,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = output_dir / f"batch_metadata_{run_id}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    (output_dir / "batch_metadata_latest.json").write_text(json.dumps(metadata, indent=2))
    print(f"Submitted batch {batch.id}; metadata saved to {metadata_path}")
    return batch


def write_combined_metadata(
    output_dir: Path,
    final_run_id: str,
    dataset_name: str,
    dataset_split: str,
    total_samples: int,
    chunk_infos: List[Dict[str, Any]],
) -> None:
    metadata = {
        "mode": "batch",
        "run_id": final_run_id,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "subset_size": total_samples,
        "chunk_batches": chunk_infos,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = output_dir / f"batch_metadata_{final_run_id}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    (output_dir / "batch_metadata_latest.json").write_text(json.dumps(metadata, indent=2))


def build_batch_request(sample: SampleRecord, model: str) -> Dict[str, Any]:
    system_prompt = (
        "You label the minimum calendar year (between 2001 and 2025) required "
        "to answer a question without temporal leakage. The label must never precede "
        "any fact mentioned in the sample; when uncertain, err toward the later year "
        "so that no future knowledge sneaks into earlier buckets."
    )
    variant_note = VARIANT_NOTES.get(sample.dataset_variant, VARIANT_NOTES[VARIANT_SFT])
    user_prompt = (
        "You receive a dataset-specific question plus an answer bundle (which may contain multiple sections).\n"
        f"{variant_note}\n"
        "Pick the smallest year Y in [2001, 2025] so that a model with knowledge "
        "through year Y could answer confidently, considering EVERYTHING in both the question "
        "and the answer bundle. If no specific time-dependent knowledge is required, output 2001.\n"
        "Rules:\n"
        "- Consider publication dates, statistics, laws, releases, and events.\n"
        "- Output the smallest year that still contains every fact mentioned.\n"
        "- If the bundle includes multiple responses (e.g., preferred/rejected answers, constraints, rationales), "
        "the chosen year must satisfy the most recent reference anywhere in the bundle.\n"
        "- If multiple explicit years are referenced, return the most recent explicit year.\n"
        "- If only a range or uncertainty is provided (e.g., 'released between 2008 and 2015'), "
        "answer with the latest year in that range so no future facts are included.\n"
        "- If information is older than 2001, still respond with 2001.\n"
        "- Do not hallucinate years that are not grounded in the text.\n"
        "- Additionally, assign the question to one category from this list: "
        f"{', '.join(CATEGORIES[:-1])}, or {CATEGORIES[-1]} if nothing fits.\n"
        "\n"
        "Illustrative example:\n"
        "Question:\n"
        "\"Teacher: In this task, you are given a text from tweets and a boolean question whether this tweet "
        "has positive sentiment or negative sentiment. Your task is to generate answer \"yes\" when the tweet has that "
        "particular sentiment, otherwise generate answer \"no\".\\nTeacher: Now, understand the problem? If you are still "
        "confused, see the following example:\\nTweet: @justinchuan Awww! I was thinking about you lot up there! Glad you enjoyed "
        "it Question: is it a positive tweet?\\nSolution: yes\\nReason: There is an expression of happiness in this tweet text, hence, "
        "we can say it's positive. So answer is 'yes'.\\n\\nNow, solve this instance: Tweet: Goddamn my back hurts this morning.  "
        "Question: is it a positive tweet?\\nStudent:\"\n"
        "Answer JSON:\n"
        '{"year": 2006, "confidence": "high", "category": "general_knowledge", "justification": "Answer references tweets, a concept only available after Twitter launched in 2006, so 2006 is the earliest safe year.", "evidence_years": [2006]}\n'
        "\n"
        "Use the same reasoning style for the sample below and respond with compact JSON only.\n"
        "\n"
        f"<question>\n{sample.question}\n</question>\n"
        f"<answer_bundle>\n{sample.answer}\n</answer_bundle>\n"
        "Return JSON exactly in this schema:\n"
        '{"year": 2001, "confidence": "low|medium|high", '
        '"category": "one of the allowed categories", '
        '"justification": "why year is required", "evidence_years": [2008]}\n'
    )
    body = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    return {"custom_id": sample.custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}


def write_jsonl(items: Iterable[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def run_live_classification(
    client: OpenAI,
    requests: List[Dict[str, Any]],
    output_dir: Path,
    batch_input_path: Path,
    model: str,
    run_id: str,
    dataset_name: str,
    dataset_split: str,
) -> Path:
    sample_count = len(requests)
    output_path = output_dir / f"batch_output_local_{run_id}.jsonl"
    print(f"Running {sample_count} live completions with {model}")
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, request in enumerate(requests, start=1):
            completion = client.chat.completions.create(**request["body"])
            payload = {
                "custom_id": request["custom_id"],
                "response": {
                    "status_code": 200,
                    "body": completion.model_dump(),
                },
            }
            handle.write(json.dumps(payload) + "\n")
            print(f"[live] Processed {idx}/{sample_count}")
    metadata = {
        "mode": "live",
        "run_id": run_id,
        "batch_id": None,
        "input_file_path": str(batch_input_path),
        "output_file_path": str(output_path),
        "model": model,
        "subset_size": sample_count,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = output_dir / f"batch_metadata_{run_id}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    (output_dir / "batch_metadata_latest.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote live output to {output_path}")
    return output_path


def wait_for_batch_completion(client: OpenAI, batch_id: str, poll_interval: int):
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in POLLABLE_STATUSES:
            time.sleep(poll_interval)
            continue
        if batch.status == "completed" and not getattr(batch, "output_file_id", None):
            time.sleep(poll_interval)
            continue
        return batch


def download_batch_output(client: OpenAI, file_id: str, output_dir: Path, run_id: str) -> Path:
    response = client.files.content(file_id)
    destination = output_dir / f"batch_output_{run_id}_{file_id}.jsonl"
    with open(destination, "wb") as handle:
        payload = response.read() if hasattr(response, "read") else response
        if isinstance(payload, bytes):
            handle.write(payload)
        else:
            handle.write(str(payload).encode("utf-8"))
    print(f"Downloaded batch output to {destination}")
    return destination


def parse_batch_predictions(output_path: Path) -> Tuple[Dict[str, Dict[str, Any]], TokenUsage]:
    year_map: Dict[str, Dict[str, Any]] = {}
    usage_totals = TokenUsage()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            custom_id = payload.get("custom_id")
            body = payload.get("response", {}).get("body", {})
            usage = body.get("usage") or {}
            usage_totals.add(
                prompt=usage.get("prompt_tokens", 0),
                completion=usage.get("completion_tokens", 0),
                total=usage.get("total_tokens"),
            )
            choices = body.get("choices") or []
            if not choices:
                continue
            message = choices[0]["message"]["content"]
            try:
                parsed = json.loads(message)
            except json.JSONDecodeError:
                parsed = {"year": 2001, "confidence": "low", "justification": message, "evidence_years": []}
            year = parsed.get("year", 2001)
            if not isinstance(year, int):
                try:
                    year = int(year)
                except (ValueError, TypeError):
                    year = 2001
            year = min(max(year, YEARS[0]), YEARS[-1])
            parsed["year"] = year
            category = parsed.get("category", "other")
            if not isinstance(category, str):
                category = "other"
            category = category.strip().lower()
            if category not in CATEGORIES:
                category = "other"
            parsed["category"] = category
            year_map[custom_id] = parsed
    return year_map, usage_totals


def attach_years(samples: List[SampleRecord], year_map: Dict[str, Dict[str, Any]]):
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        prediction = year_map.get(sample.custom_id)
        if not prediction:
            continue
        record = {
            "sample_index": sample.sample_index,
            "dataset_name": sample.dataset_name,
            "year": prediction["year"],
            "confidence": prediction.get("confidence"),
            "category": prediction.get("category", "other"),
            "justification": prediction.get("justification"),
            "evidence_years": prediction.get("evidence_years", []),
            "question": sample.question,
            "answer": sample.answer,
            "messages": sample.dataset_row.get("messages"),
            "source": sample.dataset_row.get("source"),
        }
        grouped[prediction["year"]].append(record)
    return grouped


def save_year_shards(
    grouped_records: Dict[int, List[Dict[str, Any]]],
    output_dir: Path,
    run_id: str,
    timestamp: str,
) -> Path:
    shard_dir = output_dir / f"year_shards_{run_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    category_counts = defaultdict(int)
    for year, records in grouped_records.items():
        shard_path = shard_dir / f"year={year}_{run_id}.jsonl"
        write_jsonl(records, shard_path)
        for record in records:
            category = record.get("category", "other")
            if category not in CATEGORIES:
                category = "other"
            category_counts[category] += 1
    manifest_path = shard_dir / "manifest.json"
    manifest = {
        "run_id": run_id,
        "generated_at": int(time.time()),
        "timestamp": timestamp,
        "years": sorted(grouped_records.keys()),
        "total_records": sum(len(records) for records in grouped_records.values()),
        "shard_dir": str(shard_dir),
        "category_counts": dict(sorted(category_counts.items())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return shard_dir


def print_summary(grouped_records: Dict[int, List[Dict[str, Any]]], shard_dir: Path) -> None:
    total = sum(len(records) for records in grouped_records.values())
    print(f"Saved {total} labeled samples to {shard_dir}")
    for year in sorted(grouped_records):
        print(f"Year {year}: {len(grouped_records[year])} samples")


def effective_pricing(model: str, batch_discount: bool) -> Dict[str, float] | None:
    pricing = MODEL_PRICING_PER_MILLION.get(model)
    if not pricing:
        return None
    factor = BATCH_DISCOUNT if batch_discount else 1.0
    prompt_rate = pricing.get("prompt")
    completion_rate = pricing.get("completion")
    adjusted = {}
    if prompt_rate is not None:
        adjusted["prompt"] = prompt_rate * factor
    if completion_rate is not None:
        adjusted["completion"] = completion_rate * factor
    return adjusted


def estimate_cost_usd(model: str, usage: TokenUsage, batch_discount_applied: bool) -> float | None:
    rates = effective_pricing(model, batch_discount_applied)
    if not rates:
        return None
    prompt_rate = rates.get("prompt")
    completion_rate = rates.get("completion")
    cost = 0.0
    has_rate = False
    if prompt_rate is not None:
        cost += (usage.prompt_tokens / 1_000_000) * prompt_rate
        has_rate = True
    if completion_rate is not None:
        cost += (usage.completion_tokens / 1_000_000) * completion_rate
        has_rate = True
    return round(cost, 4) if has_rate else None


def log_usage_summary(
    usage: TokenUsage,
    model: str,
    output_dir: Path,
    run_id: str,
    batch_discount_applied: bool,
) -> None:
    rates = effective_pricing(model=model, batch_discount=batch_discount_applied)
    summary = {
        "run_id": run_id,
        "model": model,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "batch_discount_applied": batch_discount_applied,
        "prompt_price_per_million_tokens_usd": rates.get("prompt") if rates else None,
        "completion_price_per_million_tokens_usd": rates.get("completion") if rates else None,
    }
    cost = estimate_cost_usd(
        model=model,
        usage=usage,
        batch_discount_applied=batch_discount_applied,
    )
    summary["estimated_cost_usd"] = cost
    usage_path = output_dir / f"token_usage_{run_id}.json"
    usage_path.write_text(json.dumps(summary, indent=2))
    latest_path = output_dir / "token_usage_latest.json"
    latest_path.write_text(json.dumps(summary, indent=2))
    printable_cost = f"${cost:.4f}" if cost is not None else "n/a"
    print(
        "Token usage summary - "
        f"prompt: {usage.prompt_tokens:,}, completion: {usage.completion_tokens:,}, "
        f"total: {usage.total_tokens:,}, estimated cost: {printable_cost}. "
        f"Saved to {usage_path}"
    )


class YearBoundedTuluLoader:
    """
    Utility for merging and shuffling shards saved under year_shards_<run_id>.
    """

    def __init__(self, shard_dir: Path):
        self.shard_dir = Path(shard_dir)

    def load(
        self,
        max_year: int,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> Dataset:
        if max_year < YEARS[0]:
            raise ValueError(f"max_year must be >= {YEARS[0]}")
        max_year = min(max_year, YEARS[-1])
        records: List[Dict[str, Any]] = []
        for year in YEARS:
            if year > max_year:
                break
            shard_paths = self._find_shard_paths(year)
            if not shard_paths:
                continue
            for shard_path in shard_paths:
                with shard_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        records.append(record)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(records)
        if not records:
            raise FileNotFoundError(f"No shards found up to year {max_year} in {self.shard_dir}")
        return Dataset.from_list(records)

    def _find_shard_paths(self, year: int) -> List[Path]:
        exact = self.shard_dir / f"year={year}.jsonl"
        if exact.exists():
            return [exact]
        matches = sorted(self.shard_dir.glob(f"year={year}_*.jsonl"))
        return matches


if __name__ == "__main__":
    main()
