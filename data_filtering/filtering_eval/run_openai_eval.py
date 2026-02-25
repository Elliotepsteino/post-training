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
from temporal_schema import MIN_YEAR, normalize_sample_prediction, to_int

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
    text = (text or "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, True
        return {}, False
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
                parsed = json.loads(fenced)
                if isinstance(parsed, dict):
                    return parsed, True
            except Exception:
                pass

    end = text.rfind("}")
    if end == -1:
        return {}, False
    start = text.rfind("{", 0, end)
    if start != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed, True
        except Exception:
            return {}, False
    return {}, False


def parse_response(text: str) -> dict:
    payload, parsed_ok = _extract_json_object(text)
    sample = normalize_sample_prediction(payload, fallback_year=MIN_YEAR)
    if not parsed_ok:
        sample["raw_response"] = text
    if "confidence" not in sample:
        sample["confidence"] = "low"
    if "category" not in sample:
        sample["category"] = "other"
    if "justification" not in sample:
        sample["justification"] = ""
    return sample


def parse_response_samples(text: str, num_samples: int) -> Dict[str, dict]:
    payload, parsed_ok = _extract_json_object(text)
    samples: Dict[str, dict] = {}

    sample_payloads: Dict[str, dict] | None = None
    if isinstance(payload, dict):
        if isinstance(payload.get("samples"), dict):
            sample_payloads = payload.get("samples")
        else:
            sample_keys = [k for k in payload.keys() if re.fullmatch(r"sample_\d+", k)]
            if sample_keys:
                sample_payloads = {k: payload.get(k) for k in sample_keys}
    elif isinstance(payload, list):
        sample_payloads = {f"sample_{idx + 1}": item for idx, item in enumerate(payload)}

    if sample_payloads is None:
        single = parse_response(text)
        if not parsed_ok:
            single["raw_response"] = text
        samples["sample_1"] = single
        return samples

    if num_samples > 0:
        target_keys = [f"sample_{i}" for i in range(1, num_samples + 1)]
    else:
        target_keys = sorted(sample_payloads.keys())

    for key in target_keys:
        item = sample_payloads.get(key)
        if isinstance(item, dict):
            sample = normalize_sample_prediction(item, fallback_year=MIN_YEAR)
            if not parsed_ok:
                sample["raw_response"] = text
            samples[key] = sample
        else:
            fallback = normalize_sample_prediction({}, fallback_year=MIN_YEAR)
            fallback["raw_response"] = text
            samples[key] = fallback
    if not samples and num_samples == 0:
        single = parse_response(text)
        samples["sample_1"] = single
    return samples


def parse_single_response(text: str) -> dict:
    return parse_response_samples(text, 1).get("sample_1", normalize_sample_prediction({}, fallback_year=MIN_YEAR))


def build_user_prompt_with_samples(question: str, answer: str, num_samples: int) -> str:
    _ = num_samples
    return build_user_prompt(question, answer)


def write_predictions(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_requests(samples: List[dict], model: str, num_samples: int) -> List[dict]:
    requests = []
    for row in samples:
        user_prompt = build_user_prompt_with_samples(row.get("question", ""), row.get("answer", ""), num_samples)
        count = max(1, num_samples)
        for idx in range(count):
            custom_id = row["id"] if count == 1 else f"{row['id']}::sample_{idx + 1}"
            req = {
                "custom_id": custom_id,
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


def run_batch(samples: List[dict], model: str, out_dir: str, completion_window: str, num_samples: int) -> str:
    from openai import OpenAI

    client = OpenAI()
    os.makedirs(out_dir, exist_ok=True)
    requests_path = os.path.join(out_dir, f"requests_{model}.jsonl")
    with open(requests_path, "w", encoding="utf-8") as f:
        for req in build_requests(samples, model, num_samples):
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


def _reconcile_explicit_year_openai(client: "OpenAI", model: str, key_a: str, sample_a: dict, key_b: str, sample_b: dict) -> dict:
    prompt = (
        "Two agent samples disagree on latest explicit year.\n"
        "Produce reconciliation JSON only.\n"
        f"{key_a}: {json.dumps({'latest_explicit_year': sample_a.get('latest_explicit_year'), 'justification': sample_a.get('justification', ''), 'entities': sample_a.get('entities', {})}, ensure_ascii=False)}\n"
        f"{key_b}: {json.dumps({'latest_explicit_year': sample_b.get('latest_explicit_year'), 'justification': sample_b.get('justification', ''), 'entities': sample_b.get('entities', {})}, ensure_ascii=False)}\n"
        "Output schema:\n"
        '{"agent_1_argument": "...", "agent_2_argument": "...", "judge_explicit_year": 2018, "judge_reason": "..."}\n'
        "Rules:\n"
        "- agent_1_argument: take sample A, consider B's evidence, argue best explicit year.\n"
        "- agent_2_argument: take sample B, consider A's evidence, argue best explicit year.\n"
        "- judge_explicit_year: single integer between 2001 and 2025.\n"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or ""
        payload, ok = _extract_json_object(text)
        if not ok:
            payload = {}
    except Exception:
        payload = {}

    judged = to_int(payload.get("judge_explicit_year")) if isinstance(payload, dict) else None
    if judged is None:
        years = [to_int(sample_a.get("latest_explicit_year")), to_int(sample_b.get("latest_explicit_year"))]
        years = [y for y in years if y is not None]
        judged = max(years) if years else MIN_YEAR
    judged = max(2001, min(2025, judged))

    return {
        "agent_1_argument": payload.get("agent_1_argument", "") if isinstance(payload, dict) else "",
        "agent_2_argument": payload.get("agent_2_argument", "") if isinstance(payload, dict) else "",
        "judge_explicit_year": judged,
        "judge_reason": payload.get("judge_reason", "") if isinstance(payload, dict) else "",
    }


def fetch_batch(batch_id: str, model: str, out_path: str, num_samples: int) -> None:
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
    samples_by_id: Dict[str, Dict[str, dict]] = {}
    order: List[str] = []
    for line in lines:
        record = json.loads(line)
        response = record.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [])
        text = ""
        if choices:
            text = choices[0].get("message", {}).get("content", "")
        sample = parse_single_response(text)
        sample["model"] = sample.get("model", model)

        custom_id = record.get("custom_id")
        base_id = str(custom_id)
        sample_key = "sample_1"
        if isinstance(custom_id, str) and "::sample_" in custom_id:
            base_id, suffix = custom_id.split("::", 1)
            sample_key = suffix

        if base_id not in samples_by_id:
            samples_by_id[base_id] = {}
            order.append(base_id)
        samples_by_id[base_id][sample_key] = sample

    preds = []
    total_samples = max(1, num_samples)
    for base_id in order:
        samples = samples_by_id.get(base_id, {})
        for idx in range(1, total_samples + 1):
            key = f"sample_{idx}"
            if key not in samples:
                samples[key] = normalize_sample_prediction({}, fallback_year=MIN_YEAR)
                samples[key]["model"] = model

        year = max(
            (to_int(sample.get("year")) for sample in samples.values() if isinstance(sample, dict)),
            default=MIN_YEAR,
        )
        row = {"id": base_id, "model": model, "year": max(MIN_YEAR, min(2025, int(year)))}
        row.update(samples)
        preds.append(row)

    write_predictions(out_path, preds)
    print(f"Wrote predictions to {out_path}")


def run_one_live(row: dict, model: str, num_samples: int, timeout_s: float, max_retries: int) -> dict:
    from openai import OpenAI

    user_prompt = build_user_prompt_with_samples(row.get("question", ""), row.get("answer", ""), num_samples)
    for attempt in range(1, max_retries + 1):
        try:
            # Build a short-lived client each attempt so transport-level timeouts apply reliably.
            client = OpenAI(timeout=timeout_s, max_retries=0)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                timeout=timeout_s,
            )
            text = resp.choices[0].message.content or ""
            sample = parse_single_response(text)
            sample["model"] = sample.get("model", model)
            return sample
        except Exception as exc:
            if attempt >= max_retries:
                sample = normalize_sample_prediction({}, fallback_year=MIN_YEAR)
                sample["model"] = model
                sample["error"] = str(exc)
                return sample
            time.sleep(1.0)


def run_live(
    samples: List[dict],
    model: str,
    out_path: str,
    sleep_s: float,
    num_samples: int,
    timeout_s: float,
    max_retries: int,
) -> None:
    preds = []
    for idx, row in enumerate(samples, start=1):
        samples_by_key: Dict[str, dict] = {}
        count = max(1, num_samples)
        for sample_idx in range(1, count + 1):
            sample = run_one_live(row, model, num_samples, timeout_s=timeout_s, max_retries=max_retries)
            samples_by_key[f"sample_{sample_idx}"] = sample

        year = max((to_int(sample.get("year")) for sample in samples_by_key.values()), default=MIN_YEAR)
        result = {"id": row["id"], "model": model, "year": max(MIN_YEAR, min(2025, int(year)))}
        result.update(samples_by_key)
        preds.append(result)
        print(f"Completed {idx}/{len(samples)}")
        if sleep_s:
            time.sleep(sleep_s)
    write_predictions(out_path, preds)
    print(f"Wrote predictions to {out_path}")


def run_live_parallel(
    samples: List[dict],
    model: str,
    out_path: str,
    max_workers: int,
    num_samples: int,
    timeout_s: float,
    max_retries: int,
) -> None:
    preds_by_id: Dict[str, Dict[str, dict]] = {}
    count = max(1, num_samples)
    total = len(samples) * count
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for row in samples:
            for sample_idx in range(1, count + 1):
                futures[executor.submit(run_one_live, row, model, num_samples, timeout_s, max_retries)] = (
                    row,
                    sample_idx,
                )
        completed = 0
        for future in as_completed(futures):
            row, sample_idx = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = normalize_sample_prediction({}, fallback_year=MIN_YEAR)
                result["model"] = model
                result["error"] = str(exc)
                print(f"[{model}] sample failed for row {row.get('id')} sample_{sample_idx}: {exc}")
            row_id = str(row["id"])
            preds_by_id.setdefault(row_id, {})[f"sample_{sample_idx}"] = result
            completed += 1
            print(f"Completed {completed}/{total}")

    preds = []
    count = max(1, num_samples)
    for row in samples:
        row_id = str(row["id"])
        samples_for_row = preds_by_id.get(row_id, {})
        for sample_idx in range(1, count + 1):
            key = f"sample_{sample_idx}"
            if key not in samples_for_row:
                samples_for_row[key] = normalize_sample_prediction({}, fallback_year=MIN_YEAR)
                samples_for_row[key]["model"] = model

        year = max((to_int(sample.get("year")) for sample in samples_for_row.values()), default=MIN_YEAR)
        result = {"id": row_id, "model": model, "year": max(MIN_YEAR, min(2025, int(year)))}
        result.update(samples_for_row)
        preds.append(result)
    write_predictions(out_path, preds)
    print(f"Wrote predictions to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI labeling for year prediction.")
    file_storage = os.environ.get("FILE_STORAGE_ROOT", "/home/epsteine/post-training/file_storage")
    default_pred_dir = os.path.join(file_storage, "data_filtering/filtering_eval/predictions")
    default_requests_dir = os.path.join(file_storage, "data_filtering/filtering_eval/requests")
    parser.add_argument("--samples", default="/home/epsteine/post-training/data_filtering/filtering_eval/data/samples.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default=os.path.join(default_pred_dir, "preds.jsonl"))
    parser.add_argument("--batch", action="store_true", help="Use OpenAI batch API")
    parser.add_argument("--completion-window", default="24h")
    parser.add_argument("--batch-out-dir", default=default_requests_dir)
    parser.add_argument("--fetch-batch", default="", help="Batch ID to fetch results")
    parser.add_argument("--wait", action="store_true", help="Wait for batch completion and fetch results")
    parser.add_argument("--poll-interval", type=float, default=30.0, help="Seconds between batch status checks")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between requests in live mode")
    parser.add_argument("--parallel", action="store_true", help="Run live requests in parallel")
    parser.add_argument("--max-workers", type=int, default=200, help="Max workers for parallel live mode")
    parser.add_argument("--num-samples", type=int, default=2, help="Number of samples per input")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per sample call")
    args = parser.parse_args()

    samples = load_samples(args.samples)
    if args.fetch_batch:
        fetch_batch(args.fetch_batch, args.model, args.out, args.num_samples)
        return
    if args.batch:
        batch_id = run_batch(samples, args.model, args.batch_out_dir, args.completion_window, args.num_samples)
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
            fetch_batch(batch_id, args.model, args.out, args.num_samples)
    else:
        if args.parallel:
            run_live_parallel(
                samples,
                args.model,
                args.out,
                args.max_workers,
                args.num_samples,
                timeout_s=args.timeout,
                max_retries=args.max_retries,
            )
        else:
            run_live(
                samples,
                args.model,
                args.out,
                args.sleep,
                args.num_samples,
                timeout_s=args.timeout,
                max_retries=args.max_retries,
            )


if __name__ == "__main__":
    main()
