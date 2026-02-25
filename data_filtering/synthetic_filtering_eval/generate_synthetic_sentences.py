#!/usr/bin/env python3
"""Generate synthetic sentence data with fixed ground-truth years."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
DECADE_RE = re.compile(r"\b(?:[12][0-9]{3}s|'?[0-9]{2}s)\b", flags=re.IGNORECASE)

_GEMINI_MODEL_ALIASES = {
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "gemini-3-pro": "gemini-3-pro-preview",
    "gemini-3-pro-preview": "gemini-3-pro-preview",
}

QUESTION_TYPES = ("explicit", "explicit_multi", "implicit")
IMPLICIT_LIKE_TYPES = {"implicit", "multi_implicit"}

# Paper-aligned temporal framing:
# explicit: single point-like fact that determines a single earliest admissible year
# explicit_multi: a sentence references two distinct entities; answer asks for the later (latest) release year
# implicit: non-event, general claim statements with a single hard-coded ground-truth year
# multi_implicit: one sentence with two implicit claims; answer is earliest year both can hold jointly

SYSTEM_PROMPT = "You generate high-quality factual synthetic evaluation text.\nReturn valid JSON only."


def load_entities(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON list in {path}")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Entry {i} is not an object")
        entity = str(row.get("entity", "")).strip()
        category = str(row.get("category", "")).strip() or "other"
        year_raw = row.get("year")
        if not entity:
            raise ValueError(f"Entry {i} missing entity")
        try:
            year = int(year_raw)
        except Exception as exc:
            raise ValueError(f"Entry {i} has invalid year: {year_raw!r}") from exc
        out.append({"entity": entity, "category": category, "year": year})
    return out


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
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
                parsed = json.loads(fenced)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

    last_end = text.rfind("}")
    if last_end != -1:
        first_start = text.find("{")
        if first_start != -1 and first_start < last_end:
            snippet = text[first_start : last_end + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
    return {}


def _validate_sentences(sentences: list[str], year: int, expected_count: int) -> tuple[bool, str]:
    if len(sentences) != expected_count:
        return False, f"Expected exactly {expected_count} sentences"
    cleaned = [s.strip() for s in sentences if isinstance(s, str) and s.strip()]
    if len(cleaned) != expected_count:
        return False, "Found empty sentence(s)"
    if len(set(cleaned)) != expected_count:
        return False, "Sentences are not unique"
    for s in cleaned:
        if len(s.split()) < 8:
            return False, f"Sentence too short/non-triviality risk: {s}"
        if YEAR_RE.search(s):
            return False, f"Sentence must not contain explicit year: {s}"
        if DECADE_RE.search(s):
            return False, f"Sentence must not contain decade hints: {s}"
    return True, ""


def _to_int_or_none(value: object) -> int | None:
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


def _fallback_sentences(entity: str, year: int, question_type: str) -> list[str]:
    _ = (entity, year)
    if question_type in IMPLICIT_LIKE_TYPES:
        return [
            f"{entity} became widely recognized as a major concern across public discourse.",
        ]
    return [
        f"The debut of {entity} marked its first public availability.",
        f"{entity} became widely recognized as a major topic after its launch.",
        f"{entity} established itself as a major offering in its domain.",
    ]


def _sanitize_implicit_distribution(
    year: int, plausible_years_raw: list[int], plausible_probs_raw: list[float]
) -> tuple[list[int], list[float]]:
    filtered_years: list[int] = []
    filtered_probs: list[float] = []
    for py_val, prob in zip(plausible_years_raw, plausible_probs_raw):
        if py_val >= year:
            filtered_years.append(py_val)
            filtered_probs.append(prob)

    if not filtered_years:
        return [year], [1.0]

    if filtered_years[0] != year:
        filtered_years = [year] + filtered_years
        filtered_probs = [0.0] + filtered_probs

    total_prob = sum(filtered_probs)
    if total_prob <= 0:
        return [year], [1.0]
    return filtered_years, [p / total_prob for p in filtered_probs]


def _confidence_ok(value: object, min_confidence: str) -> bool:
    rank = {"low": 0, "medium": 1, "high": 2}
    got = rank.get(str(value or "").strip().lower(), -1)
    need = rank.get(min_confidence, 1)
    return got >= need


def _build_prompt(
    entity: str,
    category: str,
    year: int,
    question_type: str,
    secondary_entity: str | None = None,
    secondary_category: str | None = None,
    secondary_year: int | None = None,
) -> str:
    type_spec = {
        "explicit": (
            "Generate concise temporal statements, not Q&A format, for a synthetic temporal grounding dataset.\n"
            "Each sentence should describe a factual statement about the entity whose first-public introduction year is recoverable as a single target year.\n"
            "For each sentence, the year you infer should correspond to the earliest year the statement could credibly be said as true."
        ),
        "explicit_multi": (
            "Generate concise temporal statements (not Q&A) that reference TWO DISTINCT ENTITIES in one sentence. "
            "The target year must be the release/introduction year of the LATER (LATEST) entity among the two."
        ),
        "implicit": (
            "Generate concise temporal statements about a general fact/claim, not an event launch.\n"
            "The statement should describe a persistent social/scientific understanding (for example: 'smoking is clearly harmful to the lungs'), "
            "and should be something that could be plausibly stated across a range of years rather than tied to one unique factual event.\n"
            "Avoid specific event-based or person-specific timeline anchors. Bad example: \n"
            "\"Alexander Fleming delivers a prophetic warning about the dangers of antibiotic resistance during his Nobel Prize acceptance speech.\"\n"
            "Use this style instead: ambiguous claim-style wording that can be attributed to a community over time."
        ),
        "multi_implicit": (
            "Generate one concise non-event statement that combines TWO DISTINCT implicit claims in the same sentence.\n"
            "Each claim should describe a persistent social/scientific understanding (not a one-off event).\n"
            "The target year is the earliest year when BOTH claims could reasonably be stated together as true."
        ),
    }[question_type]

    if question_type in IMPLICIT_LIKE_TYPES:
        extra_entities = ""
        if question_type == "multi_implicit":
            if not secondary_entity or secondary_year is None:
                raise ValueError("multi_implicit requires secondary entity/year")
            extra_entities = (
                f"Primary implicit concept: {entity}\n"
                f"Primary concept year: {year}\n"
                f"Secondary implicit concept: {secondary_entity}\n"
                f"Secondary concept year: {secondary_year}\n"
                f"Joint target year (earliest year both concepts can hold): {year}\n"
            )
        else:
            extra_entities = (
                f"Entity: {entity}\n"
                f"Category: {category}\n"
                f"Ground-truth year: {year}\n"
            )

        return (
            "Generate exactly 1 single-sentence statement for a synthetic temporal grounding dataset.\n"
            f"{extra_entities}"
            f"Question type: {question_type}\n\n"
            f"{type_spec}\n\n"
            "Constraints:\n"
            "- The sentence must be grammatical and natural.\n"
            "- Prefer declarative format (no question format like 'When was ...?').\n"
            "- Keep the sentence self-contained and focused on the concept(s).\n"
            "- Do NOT include any explicit year, full date, or decade mention.\n"
            "- Do NOT include phrases like 'in <year>' or '<year>s'.\n\n"
            "For plausible year support, include a full probability distribution over multiple years the statement could have been said.\n"
            "Use years from the GT year onward (no years before the GT year).\n"
            "After generation, the evaluator will normalize this distribution to sum to 1.0,\n"
            "and if any mass remains before the GT year it should be ignored.\n\n"
            "Also perform a self-check by inferring year(s) and confidence.\n"
            "Return JSON with this exact schema:\n"
            '{"sentences": ["..."], "inferred_years": [1991], "inferred_confidences": ["high"], '
            '"plausible_years": [1991, 1992, 1993, 1994], "plausible_years_probs": [0.6, 0.2, 0.1, 0.1]}'
        )

    return (
        "Generate exactly 3 different single-sentence statements for a synthetic temporal grounding dataset.\n"
        f"Entity: {entity}\n"
        f"Category: {category}\n"
        f"Ground-truth year: {year}\n"
        f"Question type: {question_type}\n\n"
        f"{type_spec}\n\n"
        "Constraints:\n"
        "- Each sentence must be grammatical and natural.\n"
        "- Prefer declarative statements (no question format like 'When was ...?').\n"
        "- Keep each sentence self-contained and focused on the entity.\n"
        "- Do NOT include any explicit year, full date, or decade mention.\n"
        "- Do NOT include phrases like 'in <year>' or '<year>s'.\n\n"
        "Additional constraint for explicit_multi only:\n"
        "- Each sentence must mention two distinct entities.\n"
        "- The ground-truth year must correspond to the later (latest) entity's release/introduction year.\n"
        "- The second entity should be a valid real-world anchor that does NOT produce a later year than the target.\n\n"
        "Also perform a self-check for each sentence by inferring year(s) and confidence.\n"
        "Return JSON with this exact schema:\n"
        '{"sentences": ["...", "...", "..."], "inferred_years": [1991, 1991, 1991], "inferred_confidences": ["high", "high", "high"]}'
    )


def _normalize_model(raw_model: str) -> str:
    lowered = (raw_model or "").strip().lower()
    return _GEMINI_MODEL_ALIASES.get(lowered, lowered)


def _is_gemini(model: str) -> bool:
    return model.startswith("gemini-")


def _call_openai(model: str, user_prompt: str, timeout: float) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        timeout=timeout,
    )
    return response.choices[0].message.content or ""


def _call_gemini(model: str, user_prompt: str) -> str:
    try:
        import google.generativeai as genai
    except ModuleNotFoundError as exc:
        raise RuntimeError("google.generativeai is required for Gemini models.") from exc

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set for Gemini generation.")

    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel(model)
    response = gemini.generate_content(
        (
            "You are a careful temporal reasoning assistant. Return JSON only.\n"
            f"{user_prompt}"
        ),
        generation_config={"response_mime_type": "application/json"},
    )
    return response.text or ""


def generate_for_entity(
    model: str,
    entity: str,
    category: str,
    year: int,
    question_type: str,
    secondary_entity: str | None,
    secondary_category: str | None,
    secondary_year: int | None,
    max_retries: int,
    verify_uniqueness: bool,
    min_confidence: str,
    request_timeout: float,
) -> tuple[list[str], str, list[int] | None, list[float] | None]:
    user_prompt = _build_prompt(
        entity,
        category,
        year,
        question_type,
        secondary_entity=secondary_entity,
        secondary_category=secondary_category,
        secondary_year=secondary_year,
    )
    model = _normalize_model(model)
    best_effort_sentences: list[str] = []

    for attempt in range(max_retries + 1):
        try:
            if _is_gemini(model):
                text = _call_gemini(model, user_prompt)
            else:
                text = _call_openai(model, user_prompt, request_timeout)
            payload = _extract_json_object(text)
            candidate = payload.get("sentences", []) if isinstance(payload, dict) else []
            inferred_years = payload.get("inferred_years", []) if isinstance(payload, dict) else []
            inferred_conf = payload.get("inferred_confidences", []) if isinstance(payload, dict) else []
            plausible_years_raw = payload.get("plausible_years", []) if isinstance(payload, dict) else []
            plausible_probs_raw = payload.get("plausible_years_probs", []) if isinstance(payload, dict) else []

            if isinstance(candidate, list):
                sentences = [str(x).strip() for x in candidate]
                best_effort_sentences = sentences
                expected_count = 1 if question_type in IMPLICIT_LIKE_TYPES else 3
                ok, reason = _validate_sentences(sentences, year, expected_count)
                parsed_plausible_years: list[int] | None = None
                parsed_plausible_probs: list[float] | None = None
                if ok and verify_uniqueness:
                    if not isinstance(inferred_years, list) or len(inferred_years) != expected_count:
                        ok = False
                        reason = "missing or invalid inferred_years"
                    elif not isinstance(inferred_conf, list) or len(inferred_conf) != expected_count:
                        ok = False
                        reason = "missing or invalid inferred_confidences"
                    else:
                        for i in range(expected_count):
                            inferred_year = _to_int_or_none(inferred_years[i])
                            if inferred_year is None:
                                ok = False
                                reason = f"invalid inferred_year at idx={i}: {inferred_years[i]}"
                                break
                            if question_type in {"explicit", "explicit_multi"}:
                                if inferred_year != year:
                                    ok = False
                                    reason = (
                                        f"verification mismatch at idx={i}: inferred={inferred_year}, "
                                        f"expected={year}, sentence='{sentences[i]}'"
                                    )
                                    break
                if question_type in IMPLICIT_LIKE_TYPES:
                    if not isinstance(plausible_years_raw, list) or not isinstance(plausible_probs_raw, list):
                        ok = False
                        reason = "missing or invalid plausible_years/probabilities"
                    elif len(plausible_years_raw) != len(plausible_probs_raw):
                        ok = False
                        reason = "plausible_years and plausible_years_probs length mismatch"
                    elif len(plausible_years_raw) == 0:
                        ok = False
                        reason = "plausible_years cannot be empty"
                    else:
                        parsed_candidate_years: list[int] = []
                        parsed_candidate_probs: list[float] = []
                        for i, py in enumerate(plausible_years_raw):
                            py_val = _to_int_or_none(py)
                            if py_val is None:
                                ok = False
                                reason = f"invalid plausible_year at idx={i}: {py}"
                                break
                            prob_raw = plausible_probs_raw[i]
                            if not isinstance(prob_raw, (int, float)):
                                ok = False
                                reason = f"invalid plausible_years_probs at idx={i}: {prob_raw}"
                                break
                            prob = float(prob_raw)
                            if prob < 0:
                                ok = False
                                reason = f"negative plausible_years_probs at idx={i}: {prob}"
                                break
                            parsed_candidate_years.append(py_val)
                            parsed_candidate_probs.append(prob)
                        if ok:
                            parsed_plausible_years, parsed_plausible_probs = _sanitize_implicit_distribution(
                                year, parsed_candidate_years, parsed_candidate_probs
                            )

                if ok:
                    return (
                        sentences,
                        "ok",
                        parsed_plausible_years,
                        parsed_plausible_probs,
                    )
                last_error = f"validation: {reason}"
            else:
                last_error = "response schema invalid"
        except Exception as exc:
            last_error = f"api error: {exc}"

        if attempt < max_retries:
            time.sleep(min(8.0, 1.5 * (2**attempt)))

    if question_type in IMPLICIT_LIKE_TYPES:
        print(f"[warn] implicit validation failed for entity='{entity}' year={year}: {last_error}")
        # Keep a valid implicit grounding target even on low-confidence or malformed
        # generation so downstream readers never receive null distributions.
        return (
            best_effort_sentences if best_effort_sentences else _fallback_sentences(entity, year, question_type),
            "invalid",
            [year],
            [1.0],
        )

    print(f"[warn] fallback for entity='{entity}' year={year}: {last_error}")
    sentences = _fallback_sentences(entity, year, question_type)
    return sentences, "fallback", None, None


def write_json(path: str, data: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3 synthetic sentences per entity with fixed GT year.")
    parser.add_argument("--input", required=True, help="Path to entities JSON list")
    parser.add_argument("--out-by-entity", required=True, help="Output JSON grouped by entity")
    parser.add_argument("--out-flat-jsonl", required=True, help="Output JSONL, one row per sentence")
    parser.add_argument("--model", default="gemini-3-pro-preview")
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument(
        "--verify-uniqueness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify each generated sentence resolves to the target year for explicit types, and checks parsed year/confidence for implicit samples.",
    )
    parser.add_argument(
        "--min-verify-confidence",
        choices=["low", "medium", "high"],
        default="medium",
        help="Minimum confidence required by the verifier.",
    )
    parser.add_argument("--explicit-count", type=int, default=60, help="Number of entities to assign explicit type")
    parser.add_argument(
        "--explicit-multi-count",
        type=int,
        default=20,
        help="Number of entities to assign explicit_multi type",
    )
    parser.add_argument("--implicit-count", type=int, default=20, help="Number of entities to assign implicit type")
    parser.add_argument(
        "--multi-implicit-count",
        type=int,
        default=20,
        help="Number of entities to assign multi_implicit type",
    )
    parser.add_argument(
        "--implicit-concepts",
        default=None,
        help="Optional JSON file of implicit concept rows used for implicit question generation.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for question-type assignment")
    args = parser.parse_args()

    implicit_concepts_path = args.implicit_concepts
    if not implicit_concepts_path:
        implicit_concepts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "implicit_concepts_20_evolving.json")
    implicit_hard_concepts_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "implicit_hard_concepts_20_general.json",
    )

    entities = load_entities(args.input)
    implicit_entities: list[dict[str, Any]] = []
    if args.implicit_count > 0:
        implicit_entities.extend(load_entities(implicit_concepts_path))
        # Preserve the previous implicit_hard distribution by folding those prompts into
        # the regular implicit pool and relabeling them as implicit.
        implicit_entities.extend(load_entities(implicit_hard_concepts_path))
    explicit_total = args.explicit_count + args.explicit_multi_count
    total = args.explicit_count + args.explicit_multi_count + args.implicit_count + args.multi_implicit_count
    if (
        args.explicit_count < 0
        or args.explicit_multi_count < 0
        or args.implicit_count < 0
        or args.multi_implicit_count < 0
    ):
        raise ValueError("Counts must be non-negative.")
    if explicit_total > len(entities):
        raise ValueError(
            f"Explicit + explicit_multi ({explicit_total}) cannot exceed entity count ({len(entities)})."
        )
    if total <= 0:
        raise ValueError("Total sample count must be positive.")

    created_at = datetime.now(timezone.utc).isoformat()

    rng = random.Random(args.seed)
    question_types: list[str] = (
        [QUESTION_TYPES[0]] * args.explicit_count
        + [QUESTION_TYPES[1]] * args.explicit_multi_count
        + [QUESTION_TYPES[2]] * args.implicit_count
        + ["multi_implicit"] * args.multi_implicit_count
    )
    rng.shuffle(question_types)

    implicit_rows: list[dict[str, Any]] = []
    if args.implicit_count + args.multi_implicit_count > 0:
        if len(implicit_entities) < args.implicit_count:
            raise ValueError(
                f"Implicit concept source has {len(implicit_entities)} rows but requires {args.implicit_count}."
            )
        implicit_rows = implicit_entities.copy()
        rng.shuffle(implicit_rows)

    selected_rows: list[dict[str, Any]] = []
    explicit_rows = entities.copy()
    rng.shuffle(explicit_rows)
    explicit_ptr = 0
    implicit_ptr = 0

    def _choose_multi_implicit_pair() -> tuple[dict[str, Any], dict[str, Any]]:
        if len(implicit_rows) < 2:
            raise ValueError("Need at least two implicit concept rows for multi_implicit generation.")
        first = implicit_rows[rng.randrange(len(implicit_rows))]
        second = implicit_rows[rng.randrange(len(implicit_rows))]
        guard = 0
        while second["entity"] == first["entity"] and guard < 20:
            second = implicit_rows[rng.randrange(len(implicit_rows))]
            guard += 1
        if second["entity"] == first["entity"]:
            second = implicit_rows[(implicit_rows.index(first) + 1) % len(implicit_rows)]
        return first, second

    for idx in range(total):
        qtype = question_types[idx]
        if qtype in {"explicit", "explicit_multi"}:
            if explicit_ptr >= len(explicit_rows):
                raise ValueError("Ran out of explicit entities while constructing samples")
            row = dict(explicit_rows[explicit_ptr])
            explicit_ptr += 1
            row["question_type"] = qtype
            selected_rows.append(row)
        elif qtype == "implicit":
            if implicit_ptr >= len(implicit_rows):
                raise ValueError("No implicit concepts loaded for implicit sample generation")
            row = dict(implicit_rows[implicit_ptr])
            implicit_ptr += 1
            row["question_type"] = qtype
            selected_rows.append(row)
        else:
            first, second = _choose_multi_implicit_pair()
            joint_year = max(first["year"], second["year"])
            selected_rows.append(
                {
                    "entity": f"{first['entity']} + {second['entity']}",
                    "category": f"{first['category']} + {second['category']}",
                    "year": joint_year,
                    "question_type": "multi_implicit",
                    "primary_entity": first["entity"],
                    "primary_category": first["category"],
                    "primary_year": first["year"],
                    "secondary_entity": second["entity"],
                    "secondary_category": second["category"],
                    "secondary_year": second["year"],
                }
            )

    results: dict[int, dict[str, Any]] = {}

    def worker(idx: int, row: dict[str, Any], question_type: str) -> tuple[int, dict[str, Any]]:
        entity = row["entity"]
        category = row["category"]
        year = row["year"]
        sentences, status, plausible_years, plausible_probs = generate_for_entity(
            model=args.model,
            entity=entity,
            category=category,
            year=year,
            question_type=question_type,
            secondary_entity=row.get("secondary_entity"),
            secondary_category=row.get("secondary_category"),
            secondary_year=row.get("secondary_year"),
            max_retries=args.max_retries,
            verify_uniqueness=args.verify_uniqueness,
            min_confidence=args.min_verify_confidence,
            request_timeout=args.request_timeout,
        )
        out = {
            "id": str(idx + 1),
            "entity": entity,
            "category": category,
            "question_type": question_type,
            "ground_truth_year": year,
            "sentences": sentences,
            "generation_status": status,
            "model": args.model,
            "created_at_utc": created_at,
        }
        if question_type == "multi_implicit":
            out["primary_entity"] = row.get("primary_entity")
            out["primary_category"] = row.get("primary_category")
            out["primary_year"] = row.get("primary_year")
            out["secondary_entity"] = row.get("secondary_entity")
            out["secondary_category"] = row.get("secondary_category")
            out["secondary_year"] = row.get("secondary_year")
        if question_type in IMPLICIT_LIKE_TYPES:
            if not isinstance(plausible_years, list) or not isinstance(plausible_probs, list):
                plausible_years = [year]
                plausible_probs = [1.0]
            out["plausible_years"] = plausible_years
            out["plausible_years_probs"] = plausible_probs
        return idx, out

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = [
            ex.submit(worker, idx, selected_rows[idx], question_types[idx])
            for idx in range(total)
        ]
        done = 0
        for fut in as_completed(futures):
            idx, out = fut.result()
            results[idx] = out
            done += 1
            if done % 10 == 0 or done == total:
                print(f"[progress] {done}/{total}", flush=True)

    by_entity = [results[i] for i in range(total)]
    flat_rows: list[dict[str, Any]] = []
    for entity_row in by_entity:
        base_id = entity_row["id"]
        for j, sentence in enumerate(entity_row["sentences"], start=1):
            row = {
                "id": f"{base_id}_{j}",
                "entity_id": base_id,
                "entity": entity_row["entity"],
                "category": entity_row["category"],
                "question_type": entity_row["question_type"],
                "ground_truth_year": entity_row["ground_truth_year"],
                "sentence": sentence,
                "model": entity_row["model"],
                "generation_status": entity_row["generation_status"],
                "created_at_utc": entity_row["created_at_utc"],
            }
            if entity_row["question_type"] in IMPLICIT_LIKE_TYPES:
                row["plausible_years"] = entity_row.get("plausible_years")
                row["plausible_years_probs"] = entity_row.get("plausible_years_probs")
            if entity_row["question_type"] == "multi_implicit":
                row["primary_entity"] = entity_row.get("primary_entity")
                row["primary_year"] = entity_row.get("primary_year")
                row["secondary_entity"] = entity_row.get("secondary_entity")
                row["secondary_year"] = entity_row.get("secondary_year")
            flat_rows.append(row)

    write_json(args.out_by_entity, by_entity)
    write_jsonl(args.out_flat_jsonl, flat_rows)

    fallback_count = sum(1 for row in by_entity if row["generation_status"] != "ok")
    print(f"Wrote {args.out_by_entity} ({len(by_entity)} entities)", flush=True)
    print(f"Wrote {args.out_flat_jsonl} ({len(flat_rows)} sentences)", flush=True)
    print(f"Fallback rows: {fallback_count}", flush=True)


if __name__ == "__main__":
    main()
