#!/usr/bin/env python3
"""Run a 3-agent temporal inference pipeline on synthetic sentences."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import time
from collections import Counter
from statistics import median
from typing import Dict, Iterable, List, Tuple

from openai import OpenAI

_GEMINI_MODEL_ALIASES = {
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "gemini-3-pro": "gemini-3-pro-preview",
    "gemini-3-pro-preview": "gemini-3-pro-preview",
}

_CLAUDE_MODEL_ALIASES = {
    "claude-4.5-haiku": "claude-haiku-4-5-20251001",
    "claude-3.5-haiku": "claude-haiku-4-5-20251001",
}


def _is_rate_limit_error(error: str) -> bool:
    lowered = (error or "").lower()
    return (
        "rate_limit_error" in lowered
        or "rate limit" in lowered
        or "rate-limit" in lowered
        or "429" in lowered
    )


def _model_worker_limit(model: str, global_limit: int, claude_limit: int) -> int:
    provider = _provider_for_model(_normalize_model(model))
    if provider == "claude":
        return max(0, min(global_limit, claude_limit))
    return max(1, global_limit)


def _parse_models(raw: str) -> List[str]:
    models = [item.strip() for item in raw.split(",") if item.strip()]
    if not models:
        raise ValueError("--agent-models must contain at least one model name.")
    return models


def _normalize_model(raw_model: str) -> str:
    lowered = (raw_model or "").lower().strip()
    return _GEMINI_MODEL_ALIASES.get(lowered, _CLAUDE_MODEL_ALIASES.get(lowered, lowered))


def _provider_for_model(model: str) -> str:
    norm = model.lower()
    if norm.startswith("gemini-"):
        return "gemini"
    if norm.startswith("claude-"):
        return "claude"
    return "openai"


def _call_openai(
    model: str,
    prompt: str,
    timeout: float,
    temperature: float | None,
) -> tuple[int | None, str, List[int], List[float], str]:
    client = OpenAI()
    for _ in range(1):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a careful temporal reasoning assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "timeout": timeout,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            year, rationale, years, probs = _extract_output_payload(text)
            return year, rationale, years, probs, ""
        except Exception as exc:  # noqa: PERF203
            return None, str(exc), [], [], str(exc)
    return None, "", [], [], ""


def _call_gemini(
    model: str,
    prompt: str,
    timeout: float,
) -> tuple[int | None, str, List[int], List[float], str]:
    try:
        import google.generativeai as genai
    except ModuleNotFoundError as exc:
        return None, str(exc), [], [], "google.generativeai is required for gemini models."

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, "GOOGLE_API_KEY not set.", [], [], "GOOGLE_API_KEY not set."

    try:
        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel(_normalize_model(model))
        response = gemini.generate_content(
            (
                "You are a careful temporal reasoning assistant. "
                "Return JSON only.\n"
                f"{prompt}"
            ),
            generation_config={"response_mime_type": "application/json"},
        )
        text = response.text or ""
        year, rationale, years, probs = _extract_output_payload(text)
        return year, rationale, years, probs, ""
    except Exception as exc:
        return None, str(exc), [], [], str(exc)


def _call_claude(
    model: str,
    prompt: str,
    timeout: float,
) -> tuple[int | None, str, List[int], List[float], str]:
    try:
        import anthropic
    except ModuleNotFoundError as exc:
        return None, str(exc), [], [], "anthropic package is required for claude models."

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None, "ANTHROPIC_API_KEY not set.", [], [], "ANTHROPIC_API_KEY not set."

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a careful temporal reasoning assistant. Return JSON only.\n"
                        f"{prompt}"
                    ),
                }
            ],
        )
        text = "".join(
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", "") == "text"
        )
        year, rationale, years, probs = _extract_output_payload(text)
        return year, rationale, years, probs, ""
    except Exception as exc:
        return None, str(exc), [], [], str(exc)


def _call_agent(
    model: str,
    prompt: str,
    timeout: float,
    retries: int,
    temperature: float | None = None,
) -> tuple[int | None, str, List[int], List[float], str]:
    model = _normalize_model(model)
    last_err = ""
    for attempt in range(max(1, retries + 1)):
        provider = _provider_for_model(model)
        if provider == "gemini":
            year, rationale, plausible_years, plausible_probs, err = _call_gemini(model, prompt, timeout)
        elif provider == "claude":
            year, rationale, plausible_years, plausible_probs, err = _call_claude(model, prompt, timeout)
        else:
            year, rationale, plausible_years, plausible_probs, err = _call_openai(
                model,
                prompt,
                timeout,
                temperature,
            )
        if year is not None:
            return year, rationale, plausible_years, plausible_probs, ""
        last_err = err
        if _is_rate_limit_error(err) and attempt + 1 < max(1, retries + 1):
            backoff = min(8.0, 1.0 * (2 ** attempt))
            time.sleep(backoff + random.random())
            continue
        continue
    if last_err:
        return None, last_err, [], [], last_err
    return None, "empty", [], [], "empty"

YEAR_PATTERN = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")
IMPLICIT_LIKE_TYPES = {"implicit", "multi_implicit"}


Prediction = tuple[int | None, str, List[int], List[float]]

def _load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return None


def _extract_json(text: str) -> Tuple[dict, bool]:
    text = (text or "").strip()
    if not text:
        return {}, False
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload, True
    except Exception:
        pass

    if "```" in text:
        start = text.find("```")
        end = text.find("```", start + 3)
        if end != -1:
            fenced = text[start + 3 : end].strip()
            if fenced.lower().startswith("json"):
                fenced = fenced[4:].strip()
            try:
                payload = json.loads(fenced)
                if isinstance(payload, dict):
                    return payload, True
            except Exception:
                pass

    end = text.rfind("}")
    if end != -1:
        start = text.rfind("{", 0, end)
        if start != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                payload = json.loads(snippet)
                if isinstance(payload, dict):
                    return payload, True
            except Exception:
                pass

    return {}, False


def _extract_year(text: str, low: int = 1800, high: int = 2030) -> int | None:
    payload, ok = _extract_json(text)
    if ok:
        y = _to_int(payload.get("year"))
        if y is not None and low <= y <= high:
            return y

        nested = payload.get("best_year")
        y = _to_int(nested)
        if y is not None and low <= y <= high:
            return y

    match = YEAR_PATTERN.search(text)
    if match:
        y = int(match.group(0))
        if low <= y <= high:
            return y
    return None


def _extract_float(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _extract_output_payload(text: str) -> tuple[int | None, str, List[int], List[float]]:
    payload, ok = _extract_json(text)
    if ok:
        year = _to_int(payload.get("year"))
        if year is None:
            year = _extract_year(text)

        rationale = ""
        for key in ("rationale", "justification", "reasoning", "analysis"):
            value = payload.get(key)
            if isinstance(value, str):
                rationale = value.strip()
                break

        raw_years = payload.get("plausible_years")
        raw_probs = payload.get("plausible_years_prob") or payload.get("plausible_years_probs")
        if isinstance(raw_years, list) and isinstance(raw_probs, list):
            cleaned = []
            for y_val, p_val in zip(raw_years, raw_probs):
                y = _to_int(y_val)
                p = _extract_float(p_val)
                if y is None or p is None:
                    continue
                if p < 0:
                    continue
                cleaned.append((y, p))
            if cleaned:
                cleaned = sorted(cleaned, key=lambda x: x[0])
                years = [y for y, _ in cleaned]
                probs = [p for _, p in cleaned]
                total = sum(probs)
                if total > 0:
                    probs = [p / total for p in probs]
                else:
                    probs = [1.0 / len(cleaned)] * len(cleaned)
                return year, rationale, years, probs

    year = _extract_year(text)
    rationale = _extract_rationale(text)
    return year, rationale, [], []


def _extract_rationale(text: str) -> str:
    payload, ok = _extract_json(text)
    if ok:
        for key in ("rationale", "justification", "reasoning", "analysis"):
            value = payload.get(key)
            if isinstance(value, str):
                return value.strip()

    return text.strip()[:500]


def _read_samples(path: str) -> List[dict]:
    rows = _load_json(path)
    samples: List[dict] = []

    if isinstance(rows, list):
        first = rows[0] if rows else None
        if first and "sentences" in first:
            for row in rows:
                sentence_list = row.get("sentences", [])
                if not sentence_list:
                    continue
                raw_plausible_years = row.get("plausible_years")
                raw_plausible_probs = row.get("plausible_years_probs")
                if not isinstance(raw_plausible_years, list):
                    raw_plausible_years = []
                if not isinstance(raw_plausible_probs, list):
                    raw_plausible_probs = []
                samples.append(
                    {
                        "id": str(row.get("id", "")),
                        "entity": row.get("entity", ""),
                        "category": row.get("category", "unknown"),
                        "question_type": row.get("question_type", "explicit"),
                        "gold_year": _to_int(row.get("ground_truth_year")) or _to_int(row.get("year")) or 2001,
                        "plausible_years": [
                            _to_int(v)
                            for v in raw_plausible_years
                            if _to_int(v) is not None
                        ],
                        "plausible_years_probs": [
                            p
                            for p in [
                                _extract_float(v) for v in raw_plausible_probs
                            ]
                            if p is not None
                        ],
                        "sentence": sentence_list[0],
                    }
                )
            return samples

        for row in rows:
            if "sentence" not in row:
                continue
            raw_plausible_years = row.get("plausible_years")
            raw_plausible_probs = row.get("plausible_years_probs")
            if not isinstance(raw_plausible_years, list):
                raw_plausible_years = []
            if not isinstance(raw_plausible_probs, list):
                raw_plausible_probs = []
            samples.append(
                {
                    "id": str(row.get("id", "")),
                    "entity": row.get("entity", ""),
                    "category": row.get("category", "unknown"),
                    "question_type": row.get("question_type", "explicit"),
                    "gold_year": _to_int(row.get("ground_truth_year")) or _to_int(row.get("year")) or 2001,
                    "plausible_years": [
                        _to_int(v) for v in raw_plausible_years if _to_int(v) is not None
                    ],
                    "plausible_years_probs": [
                        p
                        for p in [_extract_float(v) for v in raw_plausible_probs]
                        if p is not None
                    ],
                    "sentence": row.get("sentence", ""),
                }
            )
        return [s for s in samples if s["sentence"]]

    raise RuntimeError(f"Unsupported input format: {path}")


def _build_initial_prompt(sentence: str, agent_label: str, question_type: str) -> str:
    if question_type == "explicit_multi":
        return (
            "Infer the year from this synthetic sentence.\n"
            "This is explicit_multi: the sentence references two explicit facts/entities, "
            "and the target is the LATER (LATEST) of the two fact years.\n"
            "Return JSON only.\n"
            '{"year": 2001, "rationale": "..."}\n\n'
            f"Sentence: {sentence}\n"
            f"Agent: {agent_label}. Use independent reasoning and output one integer year."
        )

    if question_type not in IMPLICIT_LIKE_TYPES:
        return (
            "Infer the most likely publication/release year from this synthetic sentence.\n"
            "Return JSON only.\n"
            '{"year": 2001, "rationale": "..."}\n\n'
            f"Sentence: {sentence}\n"
            f"Agent: {agent_label}. Use independent reasoning and output one integer year."
        )

    return (
        "Infer the earliest year this statement could reasonably have been said.\n"
        "For implicit statements, return a short candidate-year distribution.\n"
        "Return JSON only.\n"
        '{"year": 2001, "plausible_years": [2001, 2002], "plausible_years_prob": [0.7, 0.3], "rationale": "..."}\n\n'
        "Rules:\n"
        "- year is the earliest plausible year and must equal the first value in plausible_years.\n"
        "- plausible_years must be ascending years from earliest plausible year onward.\n"
        "- plausible_years_prob aligns with years and must sum to 1.\n"
        "- If deterministic, provide one year with probability 1.\n"
        f"Sentence: {sentence}\n"
        f"Agent: {agent_label}. Use independent reasoning and output one integer year."
    )


def _build_revision_prompt(
    sentence: str,
    agent_label: str,
    own_year: int | None,
    own_rationale: str,
    other_evidence: List[tuple[int | None, str]],
    question_type: str,
    own_distribution: tuple[List[int], List[float]],
    other_distributions: List[tuple[int | None, str, List[int], List[float]]],
) -> str:
    formatted_others = []
    for y, r, p_yrs, p_probs in other_distributions:
        dist = ""
        if p_yrs:
            dist = "; distribution=" + ", ".join(
                f"{yy}:{prob:.3f}" for yy, prob in zip(p_yrs, p_probs)
            )
        formatted_others.append(
            f"year={y if y is not None else 'missing'}; rationale={r or 'no rationale provided'}{dist}"
        )

    if question_type in IMPLICIT_LIKE_TYPES:
        own_dist = ""
        if own_distribution[0]:
            own_dist = "; distribution=" + ", ".join(
                f"{yy}:{prob:.3f}" for yy, prob in zip(own_distribution[0], own_distribution[1])
            )
        return (
            "You are revising your earlier year estimate after seeing other agents' predictions and rationale.\n"
            "Return JSON only.\n"
            '{"year": 2001, "plausible_years": [2001, 2002], "plausible_years_prob": [0.7, 0.3], "rationale": "..."}\n\n'
            "Rules:\n"
            "- Output the earliest plausible year and keep it as the first item in plausible_years.\n"
            f"Sentence: {sentence}\n"
            f"Your earlier year: {own_year if own_year is not None else 'missing'}\n"
            f"Your earlier rationale: {own_rationale or 'no rationale provided'}\n"
            f"Your earlier distribution: {own_dist or 'none'}\n"
            f"Other agents' evidence: {formatted_others}\n"
            "Update both your earliest year and plausible-year distribution."
        )

    if question_type == "explicit_multi":
        return (
            "You are revising your earlier year estimate after seeing other agents' predictions and rationale.\n"
            "Task reminder: explicit_multi means the sentence references two explicit facts/entities, "
            "and the target is the LATER (LATEST) of those two fact years.\n"
            "Return JSON only.\n"
            '{"year": 2001, "rationale": "..."}\n\n'
            f"Sentence: {sentence}\n"
            f"Your earlier year: {own_year if own_year is not None else 'missing'}\n"
            f"Your earlier rationale: {own_rationale or 'no rationale provided'}\n"
            f"Other agents' evidence: {formatted_others}\n"
            "Re-evaluate the strongest evidence and output your best single year."
        )

    return (
        "You are revising your earlier year estimate after seeing other agents' predictions and rationale.\n"
        "Return JSON only.\n"
        '{"year": 2001, "rationale": "..."}\n\n'
        f"Sentence: {sentence}\n"
        f"Your earlier year: {own_year if own_year is not None else 'missing'}\n"
        f"Your earlier rationale: {own_rationale or 'no rationale provided'}\n"
        f"Other agents' evidence: {formatted_others}\n"
        "Re-evaluate the strongest evidence and output your best single year."
    )


def _build_judge_prompt(sentence: str, revisions: List[tuple[int | None, str, List[int], List[float]]], question_type: str) -> str:
    if question_type == "explicit_multi":
        years = [item[0] for item in revisions]
        return (
            "Choose the final best single year from three revised agent outputs.\n"
            "Task reminder: explicit_multi means the sentence references two explicit facts/entities, "
            "and the target is the LATER (LATEST) of those two fact years.\n"
            "Return JSON only.\n"
            '{"year": 2001, "rationale": "..."}\n\n'
            f"Sentence: {sentence}\n"
            f"Revised agent years: {years}\n"
            "Select one year and provide a short rationale."
        )

    if question_type not in IMPLICIT_LIKE_TYPES:
        years = [item[0] for item in revisions]
        return (
            "Choose the final best single year from three revised agent outputs.\n"
            "Prioritize accuracy over confidence. Return JSON only.\n"
            '{"year": 2001, "rationale": "..."}\n\n'
            f"Sentence: {sentence}\n"
            f"Revised agent years: {years}\n"
            "Select one year and provide a short rationale."
        )

    evidence = [
        {
            "year": year,
            "rationale": rationale,
            "plausible_years": possible_years if possible_years else None,
            "plausible_years_prob": possible_probs if possible_probs else None,
        }
        for (year, rationale, possible_years, possible_probs) in revisions
    ]
    return (
        "Choose the final best single year for this implicit claim using all revised evidence.\n"
        "Return JSON only.\n"
        '{"year": 2001, "plausible_years": [2001, 2002], "plausible_years_prob": [0.7, 0.3], "rationale": "..."}\n\n'
        "Rules:\n"
        "- Return the earliest year as an integer year and keep it as the first value in plausible_years.\n"
        "- plausible_years should be ascending and start from that earliest year.\n"
        f"Sentence: {sentence}\n"
        f"Revised agent evidence: {evidence}\n"
        "Select one year and provide rationale."
    )


def _consensus_year(values: List[int | None]) -> int | None:
    nums = [v for v in values if isinstance(v, int)]
    if not nums:
        return None
    counts = Counter(nums)
    top_freq = counts.most_common(1)[0][1]
    top = [year for year, freq in counts.items() if freq == top_freq]
    if len(top) == 1:
        return top[0]
    return int(median(sorted(nums)))


def _run_agent_jobs(
    jobs: List[tuple[str, str]],
    model: str,
    workers: int,
    timeout: float,
    retries: int,
    temperature: float | None,
    stage_label: str,
) -> Dict[str, tuple[int | None, str, List[int], List[float], str]]:
    results: Dict[str, tuple[int | None, str, List[int], List[float], str]] = {}
    total_jobs = len(jobs)

    if workers <= 0:
        for i, (sample_id, prompt) in enumerate(jobs, start=1):
            try:
                year, rationale, p_years, p_probs, err = _call_agent(
                    model,
                    prompt,
                    timeout,
                    retries,
                    temperature,
                )
            except Exception as exc:  # noqa: BLE001
                year, rationale, p_years, p_probs, err = None, str(exc), [], [], str(exc)
            results[sample_id] = (year, rationale, p_years, p_probs, err)
            if i % 25 == 0:
                print(f"[{stage_label}] {i}/{total_jobs}")
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_call_agent, model, prompt, timeout, retries, temperature): sample_id
            for sample_id, prompt in jobs
        }

        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            sample_id = futures[future]
            try:
                year, rationale, p_years, p_probs, err = future.result()
            except Exception as exc:  # noqa: BLE001
                year, rationale, p_years, p_probs, err = None, str(exc), [], [], str(exc)
            results[sample_id] = (year, rationale, p_years, p_probs, err)
            if i % 25 == 0:
                print(f"[{stage_label}] {i}/{len(futures)}")

    return results


def _run_stage(
    samples: List[dict],
    models: List[str],
    max_workers: int,
    claude_max_workers: int,
    timeout: float,
    retries: int,
    temperature: float | None,
    fallback_workers: int,
) -> Dict[str, List[Prediction]]:
    out: Dict[str, List[Prediction]] = {}

    def _run_single_agent(agent_idx: int, agent_model: str) -> tuple[int, Dict[str, Prediction]]:
        workers = _model_worker_limit(agent_model, max_workers, claude_max_workers)
        jobs: List[tuple[str, str]] = []
        jobs_by_id: Dict[str, str] = {}
        for sample in samples:
            sample_id = sample["id"]
            prompt = _build_initial_prompt(
                sample["sentence"],
                f"agent_{agent_idx}",
                sample.get("question_type", "explicit"),
            )
            jobs.append((sample_id, prompt))
            jobs_by_id[sample_id] = prompt

        results = _run_agent_jobs(
            jobs,
            agent_model,
            workers,
            timeout,
            retries,
            temperature,
            "stage1",
        )

        if (
            _provider_for_model(_normalize_model(agent_model)) == "claude"
            and fallback_workers != workers
        ):
            limited_jobs = [
                (sid, jobs_by_id[sid])
                for sid, (_, _, _, _, err) in list(results.items())
                if _is_rate_limit_error(err)
            ]
            if limited_jobs:
                print(
                    f"[stage1] Claude rate limit detected for {len(limited_jobs)} samples; "
                    f"retrying with fallback worker count {fallback_workers}."
                )
                fallback_results = _run_agent_jobs(
                    limited_jobs,
                    agent_model,
                    fallback_workers,
                    timeout,
                    retries,
                    temperature,
                    "stage1-fallback",
                )
                results.update(fallback_results)

        model_outputs: Dict[str, Prediction] = {}
        for sample_id in jobs_by_id:
            year, rationale, p_years, p_probs, _ = results.get(
                sample_id,
                (None, "", [], [], "missing"),
            )
            model_outputs[sample_id] = (year, rationale, p_years, p_probs)
        return agent_idx, model_outputs

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as ex:
        futures = {
            ex.submit(_run_single_agent, agent_idx, agent_model): agent_idx
            for agent_idx, agent_model in enumerate(models, start=1)
        }
        for future in concurrent.futures.as_completed(futures):
            agent_idx, model_outputs = future.result()
            for sample_id, prediction in model_outputs.items():
                bucket = out.setdefault(sample_id, [(None, "", [], [])] * 3)
                bucket[agent_idx - 1] = prediction

    return out


def _run_revisions(
    samples: List[dict],
    stage1: Dict[str, List[Prediction]],
    models: List[str],
    max_workers: int,
    claude_max_workers: int,
    timeout: float,
    retries: int,
    temperature: float | None,
    fallback_workers: int,
) -> Dict[str, List[Prediction]]:
    out: Dict[str, List[Prediction]] = {}
    def _run_single_revision(agent_idx: int, agent_model: str) -> tuple[int, Dict[str, Prediction]]:
        workers = _model_worker_limit(agent_model, max_workers, claude_max_workers)
        jobs: List[tuple[str, str]] = []
        jobs_by_id: Dict[str, str] = {}
        for sample in samples:
            sample_id = sample["id"]
            first = stage1.get(sample_id, [(None, "", [], [])] * 3)
            years = [item[0] for item in first]
            rationales = [item[1] for item in first]
            own_year = years[agent_idx - 1]
            own_rationale = rationales[agent_idx - 1]
            others = [
                (y, rationales[j])
                for j, y in enumerate(years)
                if j != agent_idx - 1
            ]
            prompt = _build_revision_prompt(
                sample["sentence"],
                f"agent_{agent_idx}",
                own_year,
                own_rationale,
                others,
                sample.get("question_type", "explicit"),
                (first[agent_idx - 1][2], first[agent_idx - 1][3]),
                [(y, r, p_y, p_p) for y, r, p_y, p_p in first],
            )
            jobs.append((sample_id, prompt))
            jobs_by_id[sample_id] = prompt

        results = _run_agent_jobs(
            jobs,
            agent_model,
            workers,
            timeout,
            retries,
            temperature,
            "stage2",
        )

        if (
            _provider_for_model(_normalize_model(agent_model)) == "claude"
            and fallback_workers != workers
        ):
            limited_jobs = [
                (sid, jobs_by_id[sid])
                for sid, (_, _, _, _, err) in list(results.items())
                if _is_rate_limit_error(err)
            ]
            if limited_jobs:
                print(
                    f"[stage2] Claude rate limit detected for {len(limited_jobs)} samples; "
                    f"retrying with fallback worker count {fallback_workers}."
                )
                fallback_results = _run_agent_jobs(
                    limited_jobs,
                    agent_model,
                    fallback_workers,
                    timeout,
                    retries,
                    temperature,
                    "stage2-fallback",
                )
                results.update(fallback_results)

        model_outputs: Dict[str, Prediction] = {}
        for sample_id in jobs_by_id:
            year, rationale, p_years, p_probs, _ = results.get(
                sample_id,
                (None, "", [], [], "missing"),
            )
            model_outputs[sample_id] = (year, rationale, p_years, p_probs)
        return agent_idx, model_outputs

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as ex:
        futures = {
            ex.submit(_run_single_revision, agent_idx, agent_model): agent_idx
            for agent_idx, agent_model in enumerate(models, start=1)
        }
        for future in concurrent.futures.as_completed(futures):
            agent_idx, model_outputs = future.result()
            for sample_id, prediction in model_outputs.items():
                bucket = out.setdefault(sample_id, [(None, "", [], [])] * 3)
                bucket[agent_idx - 1] = prediction

    return out


def _run_judges(
    samples: List[dict],
    revisions: Dict[str, List[Prediction]],
    model: str,
    max_workers: int,
    claude_max_workers: int,
    timeout: float,
    retries: int,
    temperature: float | None,
    fallback_workers: int,
) -> Dict[str, tuple[int | None, str, List[int], List[float]]]:
    jobs: List[tuple[str, str]] = []
    for sample in samples:
        sample_id = sample["id"]
        prompt = _build_judge_prompt(
            sample["sentence"],
            revisions.get(sample_id, [(None, "", [], [])] * 3),
            sample.get("question_type", "explicit"),
        )
        jobs.append((sample_id, prompt))

    out: Dict[str, tuple[int | None, str, List[int], List[float]]] = {}
    workers = _model_worker_limit(model, max_workers, claude_max_workers)
    results = _run_agent_jobs(
        jobs,
        model,
        workers,
        timeout,
        retries,
        temperature,
        "stage3",
    )

    if (
        _provider_for_model(_normalize_model(model)) == "claude"
        and fallback_workers != workers
    ):
        jobs_by_id = {sample_id: prompt for sample_id, prompt in jobs}
        limited_jobs = [
            (sid, jobs_by_id[sid])
            for sid, (_, _, _, _, err) in list(results.items())
            if _is_rate_limit_error(err)
        ]
        if limited_jobs:
            print(
                f"[stage3] Claude rate limit detected for {len(limited_jobs)} samples; "
                f"retrying with fallback worker count {fallback_workers}."
            )
            fallback_results = _run_agent_jobs(
                limited_jobs,
                model,
                fallback_workers,
                timeout,
                retries,
                temperature,
                "stage3-fallback",
            )
            results.update(fallback_results)

    for sample_id, value in results.items():
        year, rationale, p_years, p_probs, _ = value
        out[sample_id] = (year, rationale, p_years, p_probs)

    return out


def _build_rows(
    samples: List[dict],
    stage1: Dict[str, List[Prediction]],
    stage2: Dict[str, List[Prediction]],
    judges: Dict[str, tuple[int | None, str, List[int], List[float]]],
) -> List[dict]:
    out_rows: List[dict] = []
    for sample in samples:
        sample_id = sample["id"]
        s1 = stage1.get(sample_id, [(None, "", [], [])] * 3)
        s2 = stage2.get(sample_id, [(None, "", [], [])] * 3)
        judge, judge_rationale, judge_years, judge_probs = judges.get(
            sample_id,
            (None, "", [], []),
        )

        s1_years = [item[0] for item in s1]
        s2_years = [item[0] for item in s2]
        out_rows.append(
            {
                "id": sample_id,
                "entity": sample.get("entity", ""),
                "category": sample.get("category", "unknown"),
                "question_type": sample.get("question_type", "explicit"),
                "sentence": sample.get("sentence", ""),
                "gold_year": sample.get("gold_year", 2001),
                "stage_1": {
                    "agent_1": {
                        "year": s1[0][0],
                        "rationale": s1[0][1],
                        "plausible_years": s1[0][2],
                        "plausible_years_prob": s1[0][3],
                    },
                    "agent_2": {
                        "year": s1[1][0],
                        "rationale": s1[1][1],
                        "plausible_years": s1[1][2],
                        "plausible_years_prob": s1[1][3],
                    },
                    "agent_3": {
                        "year": s1[2][0],
                        "rationale": s1[2][1],
                        "plausible_years": s1[2][2],
                        "plausible_years_prob": s1[2][3],
                    },
                    "consensus_year": _consensus_year(s1_years),
                },
                "stage_2": {
                    "agent_1": {
                        "year": s2[0][0],
                        "rationale": s2[0][1],
                        "plausible_years": s2[0][2],
                        "plausible_years_prob": s2[0][3],
                    },
                    "agent_2": {
                        "year": s2[1][0],
                        "rationale": s2[1][1],
                        "plausible_years": s2[1][2],
                        "plausible_years_prob": s2[1][3],
                    },
                    "agent_3": {
                        "year": s2[2][0],
                        "rationale": s2[2][1],
                        "plausible_years": s2[2][2],
                        "plausible_years_prob": s2[2][3],
                    },
                    "consensus_year": _consensus_year(s2_years),
                },
                "stage_3": {
                    "judge_year": judge,
                    "judge_rationale": judge_rationale,
                    "judge_plausible_years": judge_years,
                    "judge_plausible_years_prob": judge_probs,
                },
            }
        )
    return out_rows


def run_pipeline(args: argparse.Namespace) -> List[dict]:
    samples = _read_samples(args.input)
    if not samples:
        raise RuntimeError(f"No samples read from {args.input}")

    if len(args.agent_models) != 3:
        raise ValueError("Exactly three agent models are required (agent_1, agent_2, agent_3).")

    # randomize to reduce correlated latency patterns
    random.shuffle(samples)
    judge_model = args.judge_model or args.agent_models[0]

    # randomize to reduce correlated latency patterns
    stage1 = _run_stage(
        samples,
        args.agent_models,
        args.max_workers,
        args.claude_max_workers,
        args.request_timeout,
        args.max_retries,
        args.temperature,
        args.claude_fallback_workers,
    )
    print("Completed stage 1")
    stage2 = _run_revisions(
        samples,
        stage1,
        args.agent_models,
        args.max_workers,
        args.claude_max_workers,
        args.request_timeout,
        args.max_retries,
        args.temperature,
        args.claude_fallback_workers,
    )
    print("Completed stage 2")
    judges = _run_judges(
        samples,
        stage2,
        judge_model,
        args.max_workers,
        args.claude_max_workers,
        args.request_timeout,
        args.max_retries,
        args.temperature,
        args.claude_fallback_workers,
    )
    print("Completed stage 3")

    rows = _build_rows(samples, stage1, stage2, judges)
    out_dir = os.path.dirname(args.out_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out_jsonl}")
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3-agent synthetic date pipeline.")
    parser.add_argument(
        "--input",
        default="/home/epsteine/post-training/data_filtering/synthetic_filtering_eval/synthetic_sentences_100_by_entity.json",
        help="Input JSON array file with generated synthetic samples.",
    )
    parser.add_argument(
        "--out-jsonl",
        default="/home/epsteine/post-training/data_filtering/synthetic_filtering_eval/results/synthetic_three_agent_predictions.jsonl",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Deprecated: use --agent-models for 3-agent runs. "
            "If provided, this model will be used for all three agents."
        ),
    )
    parser.add_argument(
        "--agent-models",
        default="gpt-5-mini,gemini-3-flash,claude-4.5-haiku",
        help="Comma-separated list of three agent model names (agent_1,agent_2,agent_3).",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="Optional model used by the judge in stage 3. Defaults to agent_1.",
    )
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--claude-max-workers", type=int, default=5)
    parser.add_argument("--claude-fallback-workers", type=int, default=0)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature. Default omits parameter for provider-default behavior.",
    )
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.model and args.model.strip():
        args.agent_models = _parse_models(f"{args.model},{args.model},{args.model}")
    else:
        args.agent_models = _parse_models(args.agent_models)
    if len(args.agent_models) == 1:
        args.agent_models = args.agent_models * 3
    elif len(args.agent_models) != 3:
        raise ValueError("Exactly three models are required. Use --agent-models='a,b,c'.")
    run_pipeline(args)


if __name__ == "__main__":
    main()
