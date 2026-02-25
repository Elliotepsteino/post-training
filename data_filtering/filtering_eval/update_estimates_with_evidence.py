#!/usr/bin/env python3
"""Update entity estimates using search evidence and aggregate multi-sample outputs."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from temporal_schema import (
    MAX_YEAR,
    MIN_YEAR,
    aggregate_row_samples,
    apply_temporal_fields,
    iter_samples,
    merge_sample_temporal_fields,
    normalize_entities,
    normalize_entity,
    sample_year,
    to_int,
)


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


def extract_json_object(text: str) -> dict:
    text = (text or "").strip()
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
        "You are updating temporal entity estimates using web evidence.\n"
        "Return JSON only.\n"
        "\n"
        "For each entity, output one of:\n"
        "- explicit: updated_temporal_type='explicit' and updated_explicit_year=YYYY\n"
        "- implicit: updated_temporal_type='implicit', updated_implicit_interval=[a,b], updated_implicit_probabilities=[...]\n"
        "\n"
        "Rules:\n"
        "- Use evidence fields (url/evidence/estimate) when available.\n"
        "- For implicit probabilities, provide one value per year in [a,b], sum to 1.\n"
        "- Keep years in [2001, 2025] at output.\n"
        "\n"
        f"Justification: {sample.get('justification', '')}\n"
        f"Entities: {json.dumps(entities, ensure_ascii=False)}\n"
        "\n"
        "Output schema:\n"
        '{"entities": {"Entity Name": {"updated_temporal_type": "explicit|implicit", "updated_explicit_year": 2019, "updated_implicit_interval": [2017, 2020], "updated_implicit_probabilities": [0.2, 0.4, 0.25, 0.15]}}}'
    )


def _update_entity_from_payload(name: str, ent: dict, update: dict) -> dict:
    if not isinstance(ent, dict):
        ent = {}
    if not isinstance(update, dict):
        update = {}

    merged = dict(ent)
    updated_type = str(update.get("updated_temporal_type", update.get("temporal_type", ""))).strip().lower()

    if updated_type == "explicit":
        year = to_int(update.get("updated_explicit_year"))
        if year is None:
            year = to_int(update.get("updated_best_estimate"))
        if year is None:
            year = to_int(ent.get("explicit_year", ent.get("best_estimate")))
        if year is None:
            year = MIN_YEAR
        year = min(max(year, MIN_YEAR), MAX_YEAR)
        merged["temporal_type"] = "explicit"
        merged["explicit_year"] = year
        merged["implicit_interval"] = []
        merged["implicit_probabilities"] = []
    elif updated_type == "implicit":
        merged["temporal_type"] = "implicit"
        if isinstance(update.get("updated_implicit_interval"), list):
            merged["implicit_interval"] = update.get("updated_implicit_interval")
        elif isinstance(update.get("updated_confidence_interval"), list):
            merged["implicit_interval"] = update.get("updated_confidence_interval")
        if isinstance(update.get("updated_implicit_probabilities"), list):
            merged["implicit_probabilities"] = update.get("updated_implicit_probabilities")
    else:
        legacy_best = to_int(update.get("updated_best_estimate"))
        legacy_ci = update.get("updated_confidence_interval")
        if legacy_best is not None:
            merged["explicit_year"] = min(max(legacy_best, MIN_YEAR), MAX_YEAR)
            merged["temporal_type"] = merged.get("temporal_type", "explicit")
        if isinstance(legacy_ci, list) and len(legacy_ci) == 2:
            merged["confidence_interval_95"] = legacy_ci
            if merged.get("temporal_type") == "implicit":
                merged["implicit_interval"] = legacy_ci

    normalized = normalize_entity(name, merged)
    normalized["updated_best_estimate"] = normalized.get("best_estimate", "")
    normalized["updated_confidence_interval"] = normalized.get("confidence_interval_95", [])
    return normalized


def apply_sample_updates(sample: dict, updates: dict) -> None:
    entities = normalize_entities(sample.get("entities", {}))
    updates_entities = updates.get("entities", {}) if isinstance(updates, dict) else {}
    if not isinstance(updates_entities, dict):
        updates_entities = {}

    updated_entities: Dict[str, dict] = {}
    for name, ent in entities.items():
        updated_entities[name] = _update_entity_from_payload(name, ent, updates_entities.get(name, {}))

    sample["entities"] = updated_entities
    merged = merge_sample_temporal_fields(updated_entities, fallback_year=to_int(sample.get("year")) or MIN_YEAR)
    apply_temporal_fields(sample, merged)


def aggregate_years(years: List[int], method: str) -> int:
    years = [min(max(y, MIN_YEAR), MAX_YEAR) for y in years]
    if not years:
        return MIN_YEAR
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
        "Use evidence-backed entities to choose a final year.\n"
        "Return JSON only.\n"
        f"Input: {json.dumps(sample_payload, ensure_ascii=False)}\n"
        "Output schema: {\"year\": 2019, \"justification\": \"...\"}"
    )


def aggregate_rank(years: List[int], rank: int) -> int:
    years = sorted(min(max(y, MIN_YEAR), MAX_YEAR) for y in years)
    if not years:
        return MIN_YEAR
    rank = max(1, min(rank, len(years)))
    return years[rank - 1]


def format_progress_bar(completed: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[{}] 0/0".format(" " * width)
    filled = int(width * completed / total)
    return "[{}{}] {}/{}".format("#" * filled, "-" * (width - filled), completed, total)


def _build_reconcile_prompt(sample_a: dict, sample_b: dict) -> str:
    payload = {
        "sample_a": {
            "latest_explicit_year": sample_a.get("latest_explicit_year"),
            "sample_temporal_type": sample_a.get("sample_temporal_type"),
            "possible_years": sample_a.get("possible_years"),
            "possible_years_probabilities": sample_a.get("possible_years_probabilities"),
            "justification": sample_a.get("justification", ""),
            "entities": sample_a.get("entities", {}),
        },
        "sample_b": {
            "latest_explicit_year": sample_b.get("latest_explicit_year"),
            "sample_temporal_type": sample_b.get("sample_temporal_type"),
            "possible_years": sample_b.get("possible_years"),
            "possible_years_probabilities": sample_b.get("possible_years_probabilities"),
            "justification": sample_b.get("justification", ""),
            "entities": sample_b.get("entities", {}),
        },
    }
    return (
        "You arbitrate conflicting explicit-year estimates from two temporal-labeling agents.\n"
        "Step 1: Write argument_a where Agent A seriously considers Agent B's view and argues for the better explicit year.\n"
        "Step 2: Write argument_b where Agent B seriously considers Agent A's view and argues for the better explicit year.\n"
        "Step 3: Judge and output judge_explicit_year.\n"
        "Return JSON only.\n"
        f"Input: {json.dumps(payload, ensure_ascii=False)}\n"
        "Output schema:\n"
        '{"argument_a": "...", "argument_b": "...", "judge_explicit_year": 2018, "judge_reason": "..."}'
    )


def _make_reconcile_fn(model: str):
    if model.startswith("gemini-"):
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY for Gemini reconciliation.")
        genai.configure(api_key=api_key)

        def _fn(sample_a_key: str, sample_a: dict, sample_b_key: str, sample_b: dict) -> Tuple[int | None, Dict[str, Any]]:
            details: Dict[str, Any] = {
                "reconcile_model": model,
                "sample_a_key": sample_a_key,
                "sample_b_key": sample_b_key,
            }
            prompt = _build_reconcile_prompt(sample_a, sample_b)
            try:
                gm = genai.GenerativeModel(model_name=model, system_instruction="Return JSON only, no markdown.")
                resp = gm.generate_content(prompt)
                payload = extract_json_object(resp.text or "")
                details["argument_a"] = payload.get("argument_a", "")
                details["argument_b"] = payload.get("argument_b", "")
                details["judge_reason"] = payload.get("judge_reason", "")
                year = to_int(payload.get("judge_explicit_year"))
                if year is None:
                    year = max(
                        to_int(sample_a.get("latest_explicit_year")) or MIN_YEAR,
                        to_int(sample_b.get("latest_explicit_year")) or MIN_YEAR,
                    )
                return min(max(year, MIN_YEAR), MAX_YEAR), details
            except Exception as exc:
                details["error"] = str(exc)
                fallback = max(
                    to_int(sample_a.get("latest_explicit_year")) or MIN_YEAR,
                    to_int(sample_b.get("latest_explicit_year")) or MIN_YEAR,
                )
                return fallback, details

        return _fn

    client = OpenAI()

    def _fn(sample_a_key: str, sample_a: dict, sample_b_key: str, sample_b: dict) -> Tuple[int | None, Dict[str, Any]]:
        details: Dict[str, Any] = {
            "reconcile_model": model,
            "sample_a_key": sample_a_key,
            "sample_b_key": sample_b_key,
        }
        prompt = _build_reconcile_prompt(sample_a, sample_b)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Return JSON only, no markdown."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            payload = extract_json_object(resp.choices[0].message.content or "")
            details["argument_a"] = payload.get("argument_a", "")
            details["argument_b"] = payload.get("argument_b", "")
            details["judge_reason"] = payload.get("judge_reason", "")
            year = to_int(payload.get("judge_explicit_year"))
            if year is None:
                year = max(
                    to_int(sample_a.get("latest_explicit_year")) or MIN_YEAR,
                    to_int(sample_b.get("latest_explicit_year")) or MIN_YEAR,
                )
            return min(max(year, MIN_YEAR), MAX_YEAR), details
        except Exception as exc:
            details["error"] = str(exc)
            fallback = max(
                to_int(sample_a.get("latest_explicit_year")) or MIN_YEAR,
                to_int(sample_b.get("latest_explicit_year")) or MIN_YEAR,
            )
            return fallback, details

    return _fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Update estimates using grounded evidence.")
    parser.add_argument("--preds", required=True, help="Input grounded predictions JSONL")
    parser.add_argument("--out", required=True, help="Output JSONL with updated estimates")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--model", default="gpt-5-mini", help="LLM used to update estimates")
    parser.add_argument(
        "--aggregation",
        choices=["temporal_merge", "max", "median", "majority", "llm"],
        default="temporal_merge",
        help="How to set updated_year after temporal merge",
    )
    parser.add_argument(
        "--aggregate-model",
        default="gemini-3-flash-preview",
        help="LLM used for llm aggregation and explicit-year reconciliation",
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
        return extract_json_object(text)

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
    reconcile_fn = _make_reconcile_fn(args.aggregate_model)

    for idx, row in enumerate(updated_rows):
        if row is None:
            updated_rows[idx] = rows[idx]
            row = updated_rows[idx]

        sample_payload: Dict[str, dict] = {}
        sample_years: List[int] = []
        samples_map: Dict[str, dict] = {}

        for key, sample in iter_samples(row):
            samples_map[key] = sample
            sample_payload[key] = {
                "justification": sample.get("justification", ""),
                "sample_temporal_type": sample.get("sample_temporal_type", "timeless"),
                "latest_explicit_year": sample.get("latest_explicit_year"),
                "possible_years": sample.get("possible_years"),
                "possible_years_probabilities": sample.get("possible_years_probabilities"),
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

        merged = aggregate_row_samples(samples_map, reconcile_explicit_fn=reconcile_fn)
        apply_temporal_fields(row, merged)
        row["updated_year_temporal_merge"] = row["year"]

        if args.aggregation == "llm":
            aggregation_inputs[idx] = sample_payload
        elif args.aggregation == "max":
            row["updated_year"] = row["updated_year_rule_max"]
        elif args.aggregation == "median":
            row["updated_year"] = row["updated_year_rule_median"]
        elif args.aggregation == "majority":
            row["updated_year"] = row["updated_year_rule_majority"]
        else:
            row["updated_year"] = row["updated_year_temporal_merge"]

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
                    agg_year = row.get("updated_year_temporal_merge", MIN_YEAR)
                row["updated_year_llm"] = min(max(int(agg_year), MIN_YEAR), MAX_YEAR)
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
