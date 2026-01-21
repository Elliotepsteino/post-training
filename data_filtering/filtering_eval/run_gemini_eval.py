#!/usr/bin/env python3
"""Run Gemini model predictions for year labeling (live only)."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from prompt_templates import SYSTEM_PROMPT, build_user_prompt
from run_openai_eval import parse_single_response


def load_samples(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_predictions(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def configure_gemini() -> None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY to run Gemini evals.")
    import google.generativeai as genai

    genai.configure(api_key=api_key)


def run_live(
    samples: List[dict],
    model: str,
    out_path: str,
    sleep_s: float,
    parallel: bool,
    max_workers: int,
    fallback_model: str,
    max_retries: int,
    retry_sleep_s: float,
    num_samples: int,
) -> None:
    import google.generativeai as genai

    def run_one(row: dict) -> dict:
        user_prompt = build_user_prompt(row.get("question", ""), row.get("answer", ""))
        attempt = 0
        while True:
            try:
                gm = genai.GenerativeModel(model_name=model, system_instruction=SYSTEM_PROMPT)
                resp = gm.generate_content(user_prompt)
                text = resp.text or ""
                sample = parse_single_response(text)
                sample["model"] = sample.get("model", model)
                if sleep_s:
                    time.sleep(sleep_s)
                return sample
            except Exception as exc:
                attempt += 1
                if attempt > max_retries:
                    if not fallback_model:
                        raise
                    gm = genai.GenerativeModel(model_name=fallback_model, system_instruction=SYSTEM_PROMPT)
                    resp = gm.generate_content(user_prompt)
                    text = resp.text or ""
                    sample = parse_single_response(text)
                    sample["model"] = sample.get("model", fallback_model)
                    return sample
                time.sleep(retry_sleep_s)

    preds: List[dict] = []
    count = max(1, num_samples)
    total = len(samples) * count
    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for row in samples:
                for sample_idx in range(1, count + 1):
                    futures[executor.submit(run_one, row)] = (row, sample_idx)
            done = 0
            for future in as_completed(futures):
                row, sample_idx = futures[future]
                sample = future.result()
                row_id = str(row["id"])
                existing = next((p for p in preds if p["id"] == row_id), None)
                if existing is None:
                    existing = {"id": row_id, "model": model}
                    preds.append(existing)
                existing[f"sample_{sample_idx}"] = sample
                done += 1
                print(f"[{model}] completed {done}/{total}")
    else:
        done = 0
        for row in samples:
            row_id = str(row["id"])
            entry = {"id": row_id, "model": model}
            for sample_idx in range(1, count + 1):
                entry[f"sample_{sample_idx}"] = run_one(row)
                done += 1
                print(f"[{model}] completed {done}/{total}")
            preds.append(entry)

    normalized_preds: List[dict] = []
    for row in samples:
        row_id = str(row["id"])
        entry = next((p for p in preds if p["id"] == row_id), None)
        if entry is None:
            entry = {"id": row_id, "model": model}
        for sample_idx in range(1, count + 1):
            key = f"sample_{sample_idx}"
            if key not in entry:
                entry[key] = parse_single_response("")
                entry[key]["model"] = model
        year = max((entry[f"sample_{idx}"].get("year", 2001) for idx in range(1, count + 1)), default=2001)
        entry["year"] = year
        normalized_preds.append(entry)

    write_predictions(out_path, normalized_preds)
    print(f"Wrote predictions to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemini labeling for year prediction.")
    parser.add_argument("--samples", default="/home/epsteine/post-training/data_filtering/filtering_eval/data/samples.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="/home/epsteine/post-training/data_filtering/filtering_eval/predictions/preds_gemini.jsonl")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between requests")
    parser.add_argument("--parallel", action="store_true", help="Run requests in parallel")
    parser.add_argument("--max-workers", type=int, default=200, help="Parallel worker count")
    parser.add_argument("--fallback-model", default="", help="Fallback Gemini model")
    parser.add_argument("--max-retries", type=int, default=4, help="Retry count before fallback")
    parser.add_argument("--retry-sleep", type=float, default=10.0, help="Seconds between retries")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples per input")
    args = parser.parse_args()

    configure_gemini()
    samples = load_samples(args.samples)
    run_live(
        samples,
        args.model,
        args.out,
        args.sleep,
        args.parallel,
        args.max_workers,
        args.fallback_model,
        args.max_retries,
        args.retry_sleep,
        args.num_samples,
    )


if __name__ == "__main__":
    main()
