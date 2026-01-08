#!/usr/bin/env python3
"""Run Gemini model predictions for year labeling (live only)."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from run_openai_eval import SYSTEM_PROMPT, build_user_prompt, extract_year


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
                year = extract_year(text) or 2001
                if sleep_s:
                    time.sleep(sleep_s)
                return {"id": row["id"], "model": model, "pred_year": year, "raw_response": text}
            except Exception as exc:
                attempt += 1
                if attempt > max_retries:
                    if not fallback_model:
                        raise
                    gm = genai.GenerativeModel(model_name=fallback_model, system_instruction=SYSTEM_PROMPT)
                    resp = gm.generate_content(user_prompt)
                    text = resp.text or ""
                    year = extract_year(text) or 2001
                    return {
                        "id": row["id"],
                        "model": fallback_model,
                        "pred_year": year,
                        "raw_response": text,
                        "fallback_used": True,
                        "error": str(exc),
                    }
                time.sleep(retry_sleep_s)

    preds: List[dict] = []
    total = len(samples)
    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_one, row): row for row in samples}
            done = 0
            for future in as_completed(futures):
                preds.append(future.result())
                done += 1
                print(f"[{model}] completed {done}/{total}")
    else:
        done = 0
        for row in samples:
            preds.append(run_one(row))
            done += 1
            print(f"[{model}] completed {done}/{total}")

    write_predictions(out_path, preds)
    print(f"Wrote predictions to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemini labeling for year prediction.")
    parser.add_argument("--samples", default="/home/epsteine/post-training/data_filtering/filtering_eval/data/samples.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="/home/epsteine/post-training/data_filtering/filtering_eval/predictions/preds_gemini.jsonl")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between requests")
    parser.add_argument("--parallel", action="store_true", help="Run requests in parallel")
    parser.add_argument("--max-workers", type=int, default=10, help="Parallel worker count")
    parser.add_argument("--fallback-model", default="", help="Fallback Gemini model")
    parser.add_argument("--max-retries", type=int, default=4, help="Retry count before fallback")
    parser.add_argument("--retry-sleep", type=float, default=10.0, help="Seconds between retries")
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
    )


if __name__ == "__main__":
    main()
