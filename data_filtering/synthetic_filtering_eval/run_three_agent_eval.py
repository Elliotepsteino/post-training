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

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return (
            None,
            "GOOGLE_API_KEY or GEMINI_API_KEY not set.",
            [],
            [],
            "GOOGLE_API_KEY or GEMINI_API_KEY not set.",
        )

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
            request_options={"timeout": max(1, int(timeout))},
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
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
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
VALID_YEAR_MIN = 1800
VALID_YEAR_MAX = 2030

MOA_AGGREGATE_AND_SYNTHESIZE_PREAMBLE = (
    "You have been provided with a set of responses from various open-source models to the latest user query. "
    "Your task is to synthesize these responses into a single, high-quality response. "
    "It is crucial to critically evaluate the information provided in these responses, recognizing that some of it "
    "may be biased or incorrect. Your response should not simply replicate the given answers but should offer a "
    "refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, "
    "and adheres to the highest standards of accuracy and reliability."
)


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
        if year is not None and not (VALID_YEAR_MIN <= year <= VALID_YEAR_MAX):
            year = None
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
                if not (VALID_YEAR_MIN <= y <= VALID_YEAR_MAX):
                    continue
                cleaned.append((y, p))
            if cleaned:
                cleaned = sorted(cleaned, key=lambda x: x[0])
                years = [y for y, _ in cleaned]
                probs = [p for _, p in cleaned]
                if year is None:
                    # For implicit outputs, models often provide only distribution; use earliest plausible year.
                    year = years[0]
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
            "Infer the year from this sentence.\n"
            "This is explicit_multi: the sentence references two explicit facts/entities, "
            "and the target is the LATER (LATEST) of the two fact years.\n"
            "Return JSON only.\n"
            '{"year": 2001, "rationale": "..."}\n\n'
            f"Sentence: {sentence}\n"
            f"Agent: {agent_label}. Use independent reasoning and output one integer year."
        )

    if question_type not in IMPLICIT_LIKE_TYPES:
        return (
            "Infer the most likely publication/release year from this sentence.\n"
            "Return JSON only.\n"
            '{"year": 2001, "rationale": "..."}\n\n'
            f"Sentence: {sentence}\n"
            f"Agent: {agent_label}. Use independent reasoning and output one integer year."
        )

    # Keep implicit and multi_implicit aligned with the same inference rubric.
    return (
        "Infer the earliest plausible year this statement could reasonably have been said.\n"
        "For implicit and multi_implicit statements, output an earliest-year distribution.\n"
        "Return JSON only.\n"
        '{"year": 2001, "plausible_years": [2001, 2002], "plausible_years_prob": [0.7, 0.3], "rationale": "..."}\n\n'
        "Rules:\n"
        "- year is the earliest plausible year and must equal the first value in plausible_years.\n"
        "- plausible_years must be ascending years from earliest plausible year onward.\n"
        "- plausible_years_prob aligns with years and must sum to 1.\n"
        "- Choose the earliest year where the statement would be reasonable, not the later year of peak adoption/popularity.\n"
        "- If uncertain, include multiple adjacent plausible years and keep probability mass near the earliest years.\n"
        "- If deterministic, provide one year with probability 1.\n"
        f"Sentence: {sentence}\n"
        f"Agent: {agent_label}."
    )


def _format_layer_response(
    year: int | None,
    rationale: str,
    plausible_years: List[int],
    plausible_probs: List[float],
) -> str:
    parts = [f"year={year if year is not None else 'missing'}"]
    if plausible_years:
        dist = ", ".join(
            f"{yy}:{prob:.3f}" for yy, prob in zip(plausible_years, plausible_probs)
        )
        parts.append(f"distribution={dist}")
    if rationale:
        parts.append(f"rationale={rationale}")
    return "; ".join(parts)


def _build_moa_instruction(
    sentence: str,
    question_type: str,
    responses: List[str],
) -> str:
    response_lines = "\n".join(
        f"{idx + 1}. [Model Response from A_i,{idx + 1}] {text}"
        for idx, text in enumerate(responses)
    )

    if question_type == "explicit_multi":
        task_block = (
            "Task reminder: explicit_multi means the sentence references two explicit facts/entities, "
            "and the target is the LATER (LATEST) of the two fact years.\n"
            "Return JSON only.\n"
            '{"year": 2001, "rationale": "..."}'
        )
    elif question_type not in IMPLICIT_LIKE_TYPES:
        task_block = (
            "Infer the most likely publication/release year from this sentence.\n"
            "Return JSON only.\n"
            '{"year": 2001, "rationale": "..."}'
        )
    else:
        task_block = (
            "Infer the earliest plausible year from this implicit or multi_implicit statement and provide a distribution.\n"
            "Return JSON only.\n"
            '{"year": 2001, "plausible_years": [2001, 2002], "plausible_years_prob": [0.7, 0.3], "rationale": "..."}\n'
            "Rules:\n"
            "- year must equal the first value in plausible_years.\n"
            "- plausible_years must be ascending.\n"
            "- plausible_years_prob must align with plausible_years and sum to 1.\n"
            "- Prefer the earliest reasonable year, not later years that only reflect maturity or mainstream uptake."
        )

    return (
        f"{MOA_AGGREGATE_AND_SYNTHESIZE_PREAMBLE}\n\n"
        f"Responses from models:\n{response_lines}\n\n"
        "Latest user query:\n"
        f"Sentence: {sentence}\n"
        f"{task_block}"
    )


def _build_second_proposal_prompt(
    sentence: str,
    question_type: str,
    previous_layer_responses: List[str],
) -> str:
    return _build_moa_instruction(
        sentence=sentence,
        question_type=question_type,
        responses=previous_layer_responses,
    )


def _build_aggregation_prompt(
    sentence: str,
    proposals: List[tuple[int | None, str, List[int], List[float]]],
    question_type: str,
) -> str:
    responses = [
        _format_layer_response(year, rationale, possible_years, possible_probs)
        for (year, rationale, possible_years, possible_probs) in proposals
    ]
    return _build_moa_instruction(
        sentence=sentence,
        question_type=question_type,
        responses=responses,
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


def _run_second_proposals(
    samples: List[dict],
    first_aggregation: Dict[str, tuple[int | None, str, List[int], List[float]]],
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
            agg_year, agg_rationale, agg_years, agg_probs = first_aggregation.get(
                sample_id,
                (None, "", [], []),
            )
            previous_layer_responses = [
                _format_layer_response(agg_year, agg_rationale, agg_years, agg_probs)
            ]
            prompt = _build_second_proposal_prompt(
                sample["sentence"],
                sample.get("question_type", "explicit"),
                previous_layer_responses,
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
            "stage3",
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
                    f"[stage3] Claude rate limit detected for {len(limited_jobs)} samples; "
                    f"retrying with fallback worker count {fallback_workers}."
                )
                fallback_results = _run_agent_jobs(
                    limited_jobs,
                    agent_model,
                    fallback_workers,
                    timeout,
                    retries,
                    temperature,
                    "stage3-fallback",
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


def _run_aggregation(
    samples: List[dict],
    proposals: Dict[str, List[Prediction]],
    model: str,
    max_workers: int,
    claude_max_workers: int,
    timeout: float,
    retries: int,
    temperature: float | None,
    fallback_workers: int,
    stage_label: str,
) -> Dict[str, tuple[int | None, str, List[int], List[float]]]:
    jobs: List[tuple[str, str]] = []
    for sample in samples:
        sample_id = sample["id"]
        prompt = _build_aggregation_prompt(
            sample["sentence"],
            proposals.get(sample_id, [(None, "", [], [])] * 3),
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
        stage_label,
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
                f"[{stage_label}] Claude rate limit detected for {len(limited_jobs)} samples; "
                f"retrying with fallback worker count {fallback_workers}."
            )
            fallback_results = _run_agent_jobs(
                limited_jobs,
                model,
                fallback_workers,
                timeout,
                retries,
                temperature,
                f"{stage_label}-fallback",
            )
            results.update(fallback_results)

    for sample_id, value in results.items():
        year, rationale, p_years, p_probs, _ = value
        out[sample_id] = (year, rationale, p_years, p_probs)

    return out


def _build_rows(
    samples: List[dict],
    stage1: Dict[str, List[Prediction]],
    stage2: Dict[str, tuple[int | None, str, List[int], List[float]]],
    stage3: Dict[str, List[Prediction]],
    stage4: Dict[str, tuple[int | None, str, List[int], List[float]]],
) -> List[dict]:
    out_rows: List[dict] = []
    for sample in samples:
        sample_id = sample["id"]
        s1 = stage1.get(sample_id, [(None, "", [], [])] * 3)
        s2 = stage2.get(sample_id, (None, "", [], []))
        s3 = stage3.get(sample_id, [(None, "", [], [])] * 3)
        s4 = stage4.get(
            sample_id,
            (None, "", [], []),
        )
        agg1_year, agg1_rationale, agg1_years, agg1_probs = s2
        agg2_year, agg2_rationale, agg2_years, agg2_probs = s4

        s1_years = [item[0] for item in s1]
        s3_years = [item[0] for item in s3]
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
                    "aggregated_year": agg1_year,
                    "aggregated_rationale": agg1_rationale,
                    "aggregated_plausible_years": agg1_years,
                    "aggregated_plausible_years_prob": agg1_probs,
                },
                "stage_3": {
                    "agent_1": {
                        "year": s3[0][0],
                        "rationale": s3[0][1],
                        "plausible_years": s3[0][2],
                        "plausible_years_prob": s3[0][3],
                    },
                    "agent_2": {
                        "year": s3[1][0],
                        "rationale": s3[1][1],
                        "plausible_years": s3[1][2],
                        "plausible_years_prob": s3[1][3],
                    },
                    "agent_3": {
                        "year": s3[2][0],
                        "rationale": s3[2][1],
                        "plausible_years": s3[2][2],
                        "plausible_years_prob": s3[2][3],
                    },
                    "consensus_year": _consensus_year(s3_years),
                },
                "stage_4": {
                    "aggregated_year": agg2_year,
                    "aggregated_rationale": agg2_rationale,
                    "aggregated_plausible_years": agg2_years,
                    "aggregated_plausible_years_prob": agg2_probs,
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

    random.shuffle(samples)
    aggregator_model = args.aggregator_model or args.agent_models[0]

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
    print("Completed stage 1 (proposal round 1)")
    stage2 = _run_aggregation(
        samples,
        stage1,
        aggregator_model,
        args.max_workers,
        args.claude_max_workers,
        args.request_timeout,
        args.max_retries,
        args.temperature,
        args.claude_fallback_workers,
        "stage2",
    )
    print("Completed stage 2 (aggregation round 1)")
    stage3 = _run_second_proposals(
        samples,
        stage2,
        args.agent_models,
        args.max_workers,
        args.claude_max_workers,
        args.request_timeout,
        args.max_retries,
        args.temperature,
        args.claude_fallback_workers,
    )
    print("Completed stage 3 (proposal round 2)")
    stage4 = _run_aggregation(
        samples,
        stage3,
        aggregator_model,
        args.max_workers,
        args.claude_max_workers,
        args.request_timeout,
        args.max_retries,
        args.temperature,
        args.claude_fallback_workers,
        "stage4",
    )
    print("Completed stage 4 (aggregation round 2)")

    rows = _build_rows(samples, stage1, stage2, stage3, stage4)
    out_dir = os.path.dirname(args.out_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out_jsonl}")
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3-agent synthetic date pipeline (proposal -> aggregation -> proposal -> aggregation)."
    )
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
        default="gemini-3-flash,claude-4.5-haiku,gpt-5-mini",
        help="Comma-separated list of three agent model names (agent_1,agent_2,agent_3).",
    )
    parser.add_argument(
        "--aggregator-model",
        default="gemini-3-flash",
        help="Model used by the aggregator in stages 2 and 4.",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help=argparse.SUPPRESS,
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
    if getattr(args, "judge_model", "") and not getattr(args, "aggregator_model", ""):
        args.aggregator_model = args.judge_model
    if not getattr(args, "aggregator_model", ""):
        args.aggregator_model = args.agent_models[0]
    run_pipeline(args)


if __name__ == "__main__":
    main()
