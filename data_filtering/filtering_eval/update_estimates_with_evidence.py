#!/usr/bin/env python3
"""Update entity estimates using search evidence."""
from __future__ import annotations

import argparse
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from openai import OpenAI


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_samples(row: dict) -> List[Tuple[str, dict]]:
    sample_keys = [k for k in row.keys() if k.startswith("sample_") and isinstance(row.get(k), dict)]
    if sample_keys:
        return [(key, row[key]) for key in sorted(sample_keys)]
    return [("sample_1", row)]


def to_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def normalize_interval(interval: Any, best: int | None) -> List[int]:
    if isinstance(interval, list) and len(interval) == 2:
        low = to_int(interval[0])
        high = to_int(interval[1])
        if low is not None and high is not None:
            return [low, high]
    if best is not None:
        return [best, best]
    return []


def extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    if "```" in text:
        fence_start = text.find("```")
        fence_end = text.find("```", fence_start + 3)
        if fence_end != -1:
            fenced = text[fence_start + 3 : fence_end].strip()
            if fenced.lower().startswith("json"):
                fenced = fenced[4:].strip()
            try:
                return json.loads(fenced)
            except Exception:
                pass
    end = text.rfind("}")
    if end == -1:
        return {}
    start = text.rfind("{", 0, end)
    if start != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return {}
    return {}


def build_prompt(sample: dict) -> str:
    entities = sample.get("entities", {})
    return (
        "You are updating year estimates using evidence from web search.\n"
        "Use the prior justification and evidence snippets to update each entity's estimate.\n"
        "Return JSON only.\n"
        "\n"
        "Input:\n"
        f"Justification: {sample.get('justification', '')}\n"
        f"Entities: {json.dumps(entities, ensure_ascii=False)}\n"
        "\n"
        "Output JSON schema:\n"
        '{'
        '"entities": {"Entity Name": {"updated_best_estimate": 2019, "updated_confidence_interval": [2019, 2020]}}'
        '}\n'
        "\n"
        "Rules:\n"
        "- Use evidence (url/evidence/estimate) when present; otherwise keep prior best_estimate/interval.\n"
        "- Confidence interval must include updated_best_estimate.\n"
        "- Never output a year lower than 2001; cap any lower evidence to 2001.\n"
    )


def apply_sample_updates(sample: dict, updates: dict) -> None:
    entities = sample.get("entities", {})
    if not isinstance(entities, dict):
        return
    updates_entities = updates.get("entities", {})
    if not isinstance(updates_entities, dict):
        updates_entities = {}

    for name, ent in entities.items():
        if not isinstance(ent, dict):
            continue
        update = updates_entities.get(name, {}) if isinstance(updates_entities, dict) else {}
        updated_best = to_int(update.get("updated_best_estimate"))
        updated_ci = update.get("updated_confidence_interval")
        updated_ci = normalize_interval(updated_ci, updated_best)

        if updated_best is None:
            prior_best = to_int(ent.get("best_estimate"))
            updated_best = prior_best
            updated_ci = normalize_interval(ent.get("confidence_interval_95"), prior_best)

        ent["updated_best_estimate"] = updated_best if updated_best is not None else ""
        ent["updated_confidence_interval"] = updated_ci


def sample_year(sample: dict) -> int | None:
    entities = sample.get("entities", {})
    upper_bounds: List[int] = []
    if isinstance(entities, dict):
        for ent in entities.values():
            if not isinstance(ent, dict):
                continue
            ci = ent.get("updated_confidence_interval") or ent.get("confidence_interval_95")
            if isinstance(ci, list) and len(ci) == 2:
                upper = to_int(ci[1])
                if upper is not None:
                    upper_bounds.append(upper)
    if upper_bounds:
        return max(2001, max(upper_bounds))
    fallback = to_int(sample.get("year", sample.get("pred_year")))
    if fallback is not None:
        return max(2001, fallback)
    return None


def aggregate_years(years: List[int], method: str) -> int:
    years = [max(2001, y) for y in years]
    if not years:
        return 2001
    if method == "median":
        years_sorted = sorted(years)
        return years_sorted[len(years_sorted) // 2]
    if method == "majority":
        counts: Dict[int, int] = {}
        for y in years:
            counts[y] = counts.get(y, 0) + 1
        max_count = max(counts.values())
        winners = [y for y, c in counts.items() if c == max_count]
        return max(winners)
    return max(years)


def build_aggregate_prompt(sample_payload: Dict[str, dict]) -> str:
    return (
        "You are aggregating multiple independently grounded samples.\n"
        "Use the evidence-backed entities to choose a single final year.\n"
        "Return JSON only.\n"
        "\n"
        "Input samples:\n"
        f"{json.dumps(sample_payload, ensure_ascii=False)}\n"
        "\n"
        "Output JSON schema:\n"
        '{"year": 2019, "justification": "why this year is safest across samples"}\n'
        "\n"
        "Rules:\n"
        "- Use the maximum upper bound implied by evidence when in doubt.\n"
        "- Never output a year lower than 2001; cap any lower evidence to 2001.\n"
    )


def aggregate_rank(years: List[int], rank: int) -> int:
    years = sorted(max(2001, y) for y in years)
    if not years:
        return 2001
    rank = max(1, min(rank, len(years)))
    return years[rank - 1]


def format_progress_bar(completed: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[{}] 0/0".format(" " * width)
    filled = int(width * completed / total)
    return "[{}{}] {}/{}".format("#" * filled, "-" * (width - filled), completed, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update estimates using grounded evidence.")
    parser.add_argument("--preds", required=True, help="Input grounded predictions JSONL")
    parser.add_argument("--out", required=True, help="Output JSONL with updated estimates")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--model", default="gpt-5-mini", help="LLM used to update estimates")
    parser.add_argument(
        "--aggregation",
        choices=["max", "median", "majority", "llm"],
        default="max",
        help="How to aggregate sample years into updated_year",
    )
    parser.add_argument(
        "--aggregate-model",
        default="gemini-3-flash-preview",
        help="LLM used to aggregate across samples when --aggregation=llm",
    )
    parser.add_argument(
        "--aggregate-max-workers",
        type=int,
        default=4,
        help="Parallel workers for LLM aggregation across rows",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between completions")
    args = parser.parse_args()

    rows = load_jsonl(args.preds)
    updated_rows: List[dict] = [None] * len(rows)  # type: ignore[list-item]

    def run_update(sample: dict) -> dict:
        client = OpenAI()
        prompt = build_prompt(sample)
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "Return JSON only, no extra text."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or ""
        payload = extract_json_object(text)
        return payload

    def run_aggregate(sample_payload: Dict[str, dict]) -> dict:
        prompt = build_aggregate_prompt(sample_payload)
        if args.aggregate_model.startswith("gemini-"):
            import google.generativeai as genai

            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY for Gemini aggregation.")
            genai.configure(api_key=api_key)
            gm = genai.GenerativeModel(
                model_name=args.aggregate_model,
                system_instruction="Return JSON only, no extra text.",
            )
            resp = gm.generate_content(prompt)
            text = resp.text or ""
            return extract_json_object(text)

        client = OpenAI()
        resp = client.chat.completions.create(
            model=args.aggregate_model,
            messages=[
                {"role": "system", "content": "Return JSON only, no extra text."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or ""
        return extract_json_object(text)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for row_idx, row in enumerate(rows):
            for sample_key, sample in iter_samples(row):
                futures[executor.submit(run_update, sample)] = (row_idx, sample_key)
        completed = 0
        total = len(futures)
        for future in as_completed(futures):
            row_idx, sample_key = futures[future]
            payload = future.result()
            if updated_rows[row_idx] is None:
                updated_rows[row_idx] = rows[row_idx]
            for key, sample in iter_samples(updated_rows[row_idx]):
                if key == sample_key:
                    apply_sample_updates(sample, payload)
                    break
            completed += 1
            print(f"Completed {completed}/{total}")
            if args.sleep:
                time.sleep(args.sleep)

    aggregation_inputs: Dict[int, Dict[str, dict]] = {}
    for idx, row in enumerate(updated_rows):
        if row is None:
            updated_rows[idx] = rows[idx]
            continue
        sample_payload: Dict[str, dict] = {}
        sample_years: List[int] = []
        for key, sample in iter_samples(row):
            sample_payload[key] = {
                "justification": sample.get("justification", ""),
                "entities": sample.get("entities", {}),
            }
            year = sample_year(sample)
            if year is not None:
                sample_years.append(year)

        row["updated_year_rule_max"] = aggregate_years(sample_years, "max")
        row["updated_year_rule_median"] = aggregate_years(sample_years, "median")
        row["updated_year_rule_majority"] = aggregate_years(sample_years, "majority")
        row["updated_year_rank1"] = aggregate_rank(sample_years, 1)
        row["updated_year_rank2"] = aggregate_rank(sample_years, 2)
        row["updated_year_rank3"] = aggregate_rank(sample_years, 3)
        row["updated_year_rank4"] = aggregate_rank(sample_years, 4)
        row["updated_year_rank5"] = aggregate_rank(sample_years, 5)

        if args.aggregation == "llm":
            aggregation_inputs[idx] = sample_payload
        else:
            row["updated_year"] = aggregate_years(sample_years, args.aggregation)

    if args.aggregation == "llm":
        with ThreadPoolExecutor(max_workers=args.aggregate_max_workers) as executor:
            futures = {executor.submit(run_aggregate, payload): idx for idx, payload in aggregation_inputs.items()}
            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                idx = futures[future]
                row = updated_rows[idx]
                if row is None:
                    continue
                aggregate_payload = future.result()
                agg_year = to_int(aggregate_payload.get("year"))
                if agg_year is None:
                    agg_year = row.get("updated_year_rule_max", 2001)
                row["updated_year_llm"] = max(2001, int(agg_year))
                row["updated_year"] = row["updated_year_llm"]
                completed += 1
                bar = format_progress_bar(completed, total)
                print(f"\rAggregating {bar}", end="", flush=True)
                if args.sleep:
                    time.sleep(args.sleep)
            if total:
                print()

    write_jsonl(args.out, updated_rows)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
