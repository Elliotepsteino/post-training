#!/usr/bin/env python3
"""Run OpenAI model predictions for year labeling (batch or live)."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Dict, Iterable, List

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

SYSTEM_PROMPT = (
    "You label the minimum calendar year (between 2001 and 2025) required "
    "to answer a question without temporal leakage. The label must never precede "
    "any fact mentioned in the sample; when uncertain, err toward the later year "
    "so that no future knowledge sneaks into earlier buckets."
)

VARIANT_NOTE = (
    "These are supervised instruction-tuning pairs: treat the question as the user prompt "
    "and the answer as the assistant's canonical reply."
)


def build_user_prompt(question: str, answer: str) -> str:
    return (
        "You receive a dataset-specific question plus an answer bundle (which may contain multiple sections).\n"
        f"{VARIANT_NOTE}\n"
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
        f"<question>\n{question}\n</question>\n"
        f"<answer_bundle>\n{answer}\n</answer_bundle>\n"
        "Return JSON exactly in this schema:\n"
        '{"year": 2001, "confidence": "low|medium|high", '
        '"category": "one of the allowed categories", '
        '"justification": "why year is required", "evidence_years": [2008]}\n'
    )

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
        year = extract_year(text) or 2001
        preds.append(
            {
                "id": record.get("custom_id"),
                "model": model,
                "pred_year": year,
                "raw_response": text,
            }
        )

    write_predictions(out_path, preds)
    print(f"Wrote predictions to {out_path}")


def run_live(samples: List[dict], model: str, out_path: str, sleep_s: float) -> None:
    from openai import OpenAI

    client = OpenAI()
    preds = []
    for row in samples:
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
        year = extract_year(text) or 2001
        preds.append({"id": row["id"], "model": model, "pred_year": year, "raw_response": text})
        if sleep_s:
            time.sleep(sleep_s)
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
        run_live(samples, args.model, args.out, args.sleep)


if __name__ == "__main__":
    main()
