#!/usr/bin/env python3
"""Add explicit/implicit temporal structure to gold dev/test datasets."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from temporal_schema import MIN_YEAR, merge_sample_temporal_fields, normalize_entity, to_int


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_predictions_map(path: str) -> Dict[str, dict]:
    if not path:
        return {}
    rows = load_jsonl(path)
    out: Dict[str, dict] = {}
    for row in rows:
        rid = str(row.get("id", ""))
        if rid:
            out[rid] = row
    return out


def _build_temporal_entities(row: dict, pred_row: dict | None = None) -> Dict[str, dict]:
    if pred_row:
        pred_sample = pred_row.get("sample_1", pred_row)
        pred_entities = pred_sample.get("entities", {})
        if isinstance(pred_entities, dict) and pred_entities:
            from_preds: Dict[str, dict] = {}
            for name, payload in pred_entities.items():
                from_preds[name] = normalize_entity(name, payload)
            return from_preds

    entities = row.get("entities", {})
    if not isinstance(entities, dict):
        entities = {}

    out: Dict[str, dict] = {}
    for name, ent in entities.items():
        if not isinstance(ent, dict):
            continue
        payload = dict(ent)
        if "temporal_type" not in payload:
            year = to_int(payload.get("year"))
            if year is None:
                year = to_int(row.get("gold_year")) or MIN_YEAR
            payload["temporal_type"] = "explicit"
            payload["explicit_year"] = year
            payload["best_estimate"] = year
            payload["confidence_interval_95"] = [year, year]
            source = payload.get("source")
            if source and "search_query" not in payload:
                payload["search_query"] = str(source)
        out[name] = normalize_entity(name, payload)

    if not out:
        fallback_year = to_int(row.get("gold_year")) or MIN_YEAR
        anchor = {
            "temporal_type": "explicit",
            "explicit_year": fallback_year,
            "best_estimate": fallback_year,
            "confidence_interval_95": [fallback_year, fallback_year],
            "search_query": "gold label anchor",
        }
        out["gold_label_anchor"] = normalize_entity("gold_label_anchor", anchor)
    return out


def categorize_rows(rows: List[dict], preds_map: Dict[str, dict] | None = None) -> List[dict]:
    out_rows: List[dict] = []
    for row in rows:
        pred_row = (preds_map or {}).get(str(row.get("id", "")))
        temporal_entities = _build_temporal_entities(row, pred_row=pred_row)
        merged = merge_sample_temporal_fields(temporal_entities, fallback_year=MIN_YEAR)
        updated = dict(row)
        updated["entities_categorized"] = merged["entities"]
        updated["sample_temporal_type"] = merged["sample_temporal_type"]
        updated["latest_explicit_year"] = merged["latest_explicit_year"]
        updated["possible_years"] = merged["possible_years"]
        updated["possible_years_probabilities"] = merged["possible_years_probabilities"]
        updated["gold_temporal_type"] = merged["sample_temporal_type"]
        out_rows.append(updated)
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create categorized gold dev/test JSONL files.")
    parser.add_argument(
        "--dev-in",
        default="/home/epsteine/post-training/data_filtering/filtering_eval/data/gold_dataset_dev.jsonl",
        help="Input gold dev JSONL.",
    )
    parser.add_argument(
        "--test-in",
        default="/home/epsteine/post-training/data_filtering/filtering_eval/data/gold_dataset_test.jsonl",
        help="Input gold test JSONL.",
    )
    parser.add_argument(
        "--dev-out",
        default="/home/epsteine/post-training/data_filtering/filtering_eval/data/gold_dataset_dev_categorized.jsonl",
        help="Output categorized gold dev JSONL.",
    )
    parser.add_argument(
        "--test-out",
        default="/home/epsteine/post-training/data_filtering/filtering_eval/data/gold_dataset_test_categorized.jsonl",
        help="Output categorized gold test JSONL.",
    )
    parser.add_argument(
        "--dev-preds",
        default="",
        help="Optional predictions JSONL aligned with dev IDs; if set, pulls temporal entities from predictions.",
    )
    parser.add_argument(
        "--test-preds",
        default="",
        help="Optional predictions JSONL aligned with test IDs; if set, pulls temporal entities from predictions.",
    )
    args = parser.parse_args()

    dev_preds = load_predictions_map(args.dev_preds)
    test_preds = load_predictions_map(args.test_preds)
    dev_rows = categorize_rows(load_jsonl(args.dev_in), preds_map=dev_preds)
    test_rows = categorize_rows(load_jsonl(args.test_in), preds_map=test_preds)

    write_jsonl(args.dev_out, dev_rows)
    write_jsonl(args.test_out, test_rows)
    print(f"Wrote {args.dev_out} ({len(dev_rows)})")
    print(f"Wrote {args.test_out} ({len(test_rows)})")


if __name__ == "__main__":
    main()
