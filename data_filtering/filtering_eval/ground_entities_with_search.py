#!/usr/bin/env python3
"""Augment entity fields with search evidence."""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")


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


def extract_years(text: str) -> List[int]:
    years: List[int] = []
    for match in YEAR_RE.findall(text or ""):
        try:
            years.append(int(match))
        except Exception:
            continue
    return years


def pick_estimate(candidates: List[int], best_estimate: Any) -> int | None:
    if not candidates:
        return None
    if isinstance(best_estimate, int):
        candidates = sorted(candidates, key=lambda y: (abs(y - best_estimate), -y))
        return candidates[0]
    return max(candidates)


def response_to_dict(resp: Any) -> Dict[str, Any]:
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    if hasattr(resp, "to_dict"):
        return resp.to_dict()
    if isinstance(resp, dict):
        return resp
    return {}


def extract_top_result(data: Dict[str, Any]) -> Tuple[str, str]:
    results: List[dict] = []
    output = data.get("output", [])
    for item in output:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("results"), list):
            results.extend(item.get("results", []))
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("results"), list):
                results.extend(content.get("results", []))

    if results:
        top = results[0]
        url = top.get("url", "") or top.get("link", "")
        snippet = top.get("snippet", "") or top.get("content", "") or top.get("title", "")
        return url, snippet

    # Fallback: use output text + first annotation URL if present.
    text = data.get("output_text", "")
    annotations: List[dict] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text":
                text = text or content.get("text", "")
                annotations.extend(content.get("annotations", []))

    url = ""
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        url = ann.get("url", "") or ann.get("source_url", "") or ann.get("link", "")
        if url:
            break
    return url, text


def search_once(client: Any, query: str, model: str) -> Tuple[str, str]:
    resp = client.responses.create(
        model=model,
        tools=[{"type": "web_search"}],
        tool_choice="required",
        input=query,
    )
    #breakpoint()
    data = response_to_dict(resp)
    return extract_top_result(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add search evidence to entity fields.")
    parser.add_argument("--preds", required=True, help="Path to predictions JSONL")
    parser.add_argument("--out", required=True, help="Output path for augmented JSONL")
    parser.add_argument("--model", default="gpt-4o-mini", help="Search-capable OpenAI model")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between query completions")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing url/evidence/estimate")
    parser.add_argument("--max-workers", type=int, default=10, help="Parallel search workers")
    args = parser.parse_args()

    from openai import OpenAI

    rows = load_jsonl(args.preds)
    tasks: List[Tuple[int, str, str, dict, str]] = []
    for row_idx, row in enumerate(rows):
        for sample_key, sample in iter_samples(row):
            entities = sample.get("entities", {})
            if not isinstance(entities, dict):
                continue
            for name, ent in entities.items():
                if not isinstance(ent, dict):
                    continue
                query = ent.get("search_query")
                if not query:
                    continue
                if not args.overwrite and any(k in ent for k in ("url", "evidence", "estimate")):
                    continue
                tasks.append((row_idx, sample_key, name, ent, str(query)))

    def run_task(
        row_idx: int, sample_key: str, name: str, ent: dict, query: str
    ) -> Tuple[int, str, str, dict, str, str, int | None]:
        client = OpenAI()
        url, snippet = search_once(client, query, args.model)
        years = extract_years(snippet)
        estimate = pick_estimate(years, ent.get("best_estimate"))
        return row_idx, sample_key, name, ent, url, snippet, estimate

    completed = 0
    total = len(tasks)
    if total:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(run_task, row_idx, sample_key, name, ent, query): (row_idx, name)
                for row_idx, sample_key, name, ent, query in tasks
            }
            for future in as_completed(futures):
                row_idx, sample_key, name, ent, url, snippet, estimate = future.result()
                ent["url"] = url
                ent["evidence"] = snippet
                ent["estimate"] = estimate if estimate is not None else ""
                completed += 1
                print(f"Completed {completed}/{total}: {name} ({sample_key}, row {row_idx + 1})")
                if args.sleep:
                    time.sleep(args.sleep)

    write_jsonl(args.out, rows)


if __name__ == "__main__":
    main()
