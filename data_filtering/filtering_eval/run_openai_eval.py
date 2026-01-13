#!/usr/bin/env python3
"""Run OpenAI model predictions for year labeling (batch or live)."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List

from prompt_templates import SYSTEM_PROMPT, build_user_prompt

YEAR_RE = re.compile(r"\b(20\d{2})\b")


def load_samples(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def extract_year(text: str) -> int | None:
    try:
        payload = json.loads(text)
        year = int(payload.get("year"))
        if 2001 <= year <= 2025:
            return year
    except Exception:
        pass
    match = YEAR_RE.search(text)
    if match:
        year = int(match.group(1))
        if 2001 <= year <= 2025:
            return year
    return None


def _extract_json_object(text: str) -> tuple[dict, bool]:
    text = text.strip()
    try:
        return json.loads(text), True
    except Exception:
        pass
    # Try to recover last JSON object from free-form text.
    end = text.rfind("}")
    if end == -1:
        return {}, False
    start = text.rfind("{", 0, end)
    if start != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet), True
        except Exception:
            return {}, False
    return {}, False


def _compute_year_from_entities(entities: dict) -> int | None:
    upper_bounds = []
    for val in entities.values():
        if not isinstance(val, dict):
            continue
        ci = val.get("confidence_interval_95")
        if isinstance(ci, list) and len(ci) == 2:
            try:
                upper_bounds.append(int(ci[1]))
            except Exception:
                continue
    if upper_bounds:
        return max(upper_bounds)
    return None


def parse_response(text: str) -> dict:
    payload, parsed_ok = _extract_json_object(text)

    confidence = payload.get("confidence", "")
    category = payload.get("category", "")
    justification = payload.get("justification", "")
    entities = payload.get("entities", {})
    if not isinstance(entities, dict):
        entities = {}

    year_from_entities = _compute_year_from_entities(entities)
    year = year_from_entities or extract_year(text) or 2001

    result = {
        "pred_year": year,
        "confidence": confidence,
        "category": category,
        "justification": justification,
        "entities": entities,
    }
    if not parsed_ok:
        result["raw_response"] = text
    return result


def write_predictions(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_requests(samples: List[dict], model: str) -> List[dict]:
    requests = []
    for row in samples:
        user_prompt = build_user_prompt(row.get("question", ""), row.get("answer", ""))
        req = {
            "custom_id": row["id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            },
        }
        requests.append(req)
    return requests


def run_batch(samples: List[dict], model: str, out_dir: str, completion_window: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    os.makedirs(out_dir, exist_ok=True)
    requests_path = os.path.join(out_dir, f"requests_{model}.jsonl")
    with open(requests_path, "w", encoding="utf-8") as f:
        for req in build_requests(samples, model):
            f.write(json.dumps(req, ensure_ascii=True) + "\n")

    file_obj = client.files.create(file=open(requests_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata={"model": model},
    )
    print(f"Submitted batch {batch.id} for model {model}")
    print("Use --fetch-batch to retrieve results when complete.")
    return batch.id


def fetch_batch(batch_id: str, model: str, out_path: str) -> None:
    from openai import OpenAI

    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    if batch.status not in {"completed", "failed", "expired", "cancelled"}:
        print(f"Batch {batch_id} status: {batch.status}")
        return
    if not batch.output_file_id:
        raise RuntimeError(f"Batch {batch_id} completed without output file.")

    content = client.files.content(batch.output_file_id)
    lines = content.text.splitlines()
    preds = []
    for line in lines:
        record = json.loads(line)
        response = record.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [])
        text = ""
        if choices:
            text = choices[0].get("message", {}).get("content", "")
        parsed = parse_response(text)
        preds.append({"id": record.get("custom_id"), "model": model, **parsed})

    write_predictions(out_path, preds)
    print(f"Wrote predictions to {out_path}")


def run_one_live(row: dict, model: str) -> dict:
    from openai import OpenAI

    client = OpenAI()
    user_prompt = build_user_prompt(row.get("question", ""), row.get("answer", ""))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or ""
    parsed = parse_response(text)
    return {"id": row["id"], "model": model, **parsed}


def run_live(samples: List[dict], model: str, out_path: str, sleep_s: float) -> None:
    preds = []
    for idx, row in enumerate(samples, start=1):
        preds.append(run_one_live(row, model))
        print(f"Completed {idx}/{len(samples)}")
        if sleep_s:
            time.sleep(sleep_s)
    write_predictions(out_path, preds)
    print(f"Wrote predictions to {out_path}")


def run_live_parallel(samples: List[dict], model: str, out_path: str, max_workers: int) -> None:
    preds_by_id: Dict[str, dict] = {}
    total = len(samples)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one_live, row, model): row for row in samples}
        completed = 0
        for future in as_completed(futures):
            row = futures[future]
            result = future.result()
            preds_by_id[str(row["id"])] = result
            completed += 1
            print(f"Completed {completed}/{total}")

    preds = [preds_by_id[str(row["id"])] for row in samples if str(row["id"]) in preds_by_id]
    write_predictions(out_path, preds)
    print(f"Wrote predictions to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI labeling for year prediction.")
    parser.add_argument("--samples", default="/home/epsteine/post-training/data_filtering/filtering_eval/data/samples.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="/home/epsteine/post-training/data_filtering/filtering_eval/predictions/preds.jsonl")
    parser.add_argument("--batch", action="store_true", help="Use OpenAI batch API")
    parser.add_argument("--completion-window", default="24h")
    parser.add_argument("--batch-out-dir", default="/home/epsteine/post-training/data_filtering/filtering_eval/requests")
    parser.add_argument("--fetch-batch", default="", help="Batch ID to fetch results")
    parser.add_argument("--wait", action="store_true", help="Wait for batch completion and fetch results")
    parser.add_argument("--poll-interval", type=float, default=30.0, help="Seconds between batch status checks")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between requests in live mode")
    parser.add_argument("--parallel", action="store_true", help="Run live requests in parallel")
    parser.add_argument("--max-workers", type=int, default=35, help="Max workers for parallel live mode")
    args = parser.parse_args()

    samples = load_samples(args.samples)
    if args.fetch_batch:
        fetch_batch(args.fetch_batch, args.model, args.out)
        return
    if args.batch:
        batch_id = run_batch(samples, args.model, args.batch_out_dir, args.completion_window)
        if args.wait:
            from openai import OpenAI

            client = OpenAI()
            while True:
                batch = client.batches.retrieve(batch_id)
                status = batch.status
                if status in {"completed", "failed", "expired", "cancelled"}:
                    break
                print(f"Batch {batch_id} status: {status}")
                time.sleep(args.poll_interval)
            if status != "completed":
                raise RuntimeError(f"Batch {batch_id} finished with status: {status}")
            fetch_batch(batch_id, args.model, args.out)
    else:
        if args.parallel:
            run_live_parallel(samples, args.model, args.out, args.max_workers)
        else:
            run_live(samples, args.model, args.out, args.sleep)


if __name__ == "__main__":
    main()
