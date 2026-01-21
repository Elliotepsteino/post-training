#!/usr/bin/env python3
"""Classify non-conservative prediction failures using an LLM."""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

from openai import OpenAI


BUCKETS = {
    1: "Fail to extract the right entities",
    2: "Fail to specify a good search query",
    3: "Search fails to retrieve the correct answer to the query",
    4: "Gold answer is wrong",
    5: "Other",
}


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


def get_pred_year(row: dict) -> int | None:
    for key in ("updated_year", "year", "pred_year"):
        if key in row and row[key] is not None:
            return int(row[key])
    return None


def build_prompt(gold_row: dict, pred_row: dict) -> str:
    return (
        "Classify why the model is non-conservative (predicted year < gold year).\n"
        "Choose exactly one bucket from the list and return JSON only.\n"
        "\n"
        "Buckets:\n"
        "1. Fail to extract the right entities\n"
        "2. Fail to specify a good search query\n"
        "3. Search fails to retrieve the correct answer to the query\n"
        "4. Gold answer is wrong\n"
        "5. Other\n"
        "\n"
        f"Gold row:\n{json.dumps(gold_row, ensure_ascii=False)}\n"
        f"Prediction row:\n{json.dumps(pred_row, ensure_ascii=False)}\n"
        "\n"
        'Return JSON schema: {"bucket": 1, "label": "Fail to extract the right entities", "rationale": "..." }\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify non-conservative failures.")
    parser.add_argument("--preds", required=True, help="Updated predictions JSONL")
    parser.add_argument("--gold-path", required=True, help="Gold dataset JSONL")
    parser.add_argument("--out", required=True, help="Output JSONL for classifications")
    parser.add_argument("--model", default="gpt-5-mini", help="LLM used to classify failures")
    args = parser.parse_args()

    gold_rows = {row["id"]: row for row in load_jsonl(args.gold_path)}
    preds = load_jsonl(args.preds)

    client = OpenAI()
    results: List[dict] = []
    for row in preds:
        gid = row.get("id")
        if gid not in gold_rows:
            continue
        gold_year = int(gold_rows[gid].get("gold_year", 2001))
        pred_year = get_pred_year(row)
        if pred_year is None or pred_year >= gold_year:
            continue

        prompt = build_prompt(gold_rows[gid], row)
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
        bucket = payload.get("bucket", 5)
        try:
            bucket = int(bucket)
        except Exception:
            bucket = 5
        if bucket not in BUCKETS:
            bucket = 5
        label = payload.get("label", BUCKETS[bucket])
        rationale = payload.get("rationale", "")
        results.append(
            {
                "id": gid,
                "model": row.get("model", ""),
                "pred_year": pred_year,
                "gold_year": gold_year,
                "bucket": bucket,
                "label": label,
                "rationale": rationale,
            }
        )

    write_jsonl(args.out, results)


if __name__ == "__main__":
    main()
