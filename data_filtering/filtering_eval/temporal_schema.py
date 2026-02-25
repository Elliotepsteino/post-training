#!/usr/bin/env python3
"""Shared explicit/implicit temporal schema helpers."""
from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, Iterable, List, Tuple

MIN_YEAR = 2001
MAX_YEAR = 2025
TEMPORAL_TYPES = {"explicit", "implicit", "timeless"}
YEAR_RE = re.compile(r"\b(20\d{2})\b")


def to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except Exception:
            return None
        if math.isfinite(parsed):
            return parsed
    return None


def _normalize_probability_list(values: Iterable[Any], expected_len: int) -> List[float]:
    probs = []
    for item in values:
        parsed = _to_float(item)
        probs.append(max(0.0, parsed if parsed is not None else 0.0))

    if expected_len <= 0:
        return []
    if len(probs) != expected_len or sum(probs) <= 0.0:
        return [1.0 / expected_len] * expected_len

    total = sum(probs)
    normalized = [p / total for p in probs]
    drift = 1.0 - sum(normalized)
    normalized[0] += drift
    return normalized


def normalize_interval(interval: Any, fallback_year: int | None = None) -> List[int]:
    if isinstance(interval, list) and len(interval) == 2:
        low = to_int(interval[0])
        high = to_int(interval[1])
        if low is not None and high is not None:
            if high < low:
                low, high = high, low
            return [low, high]
    if fallback_year is not None:
        return [fallback_year, fallback_year]
    return []


def _infer_temporal_type(entity: Dict[str, Any]) -> str:
    typed = str(entity.get("temporal_type", entity.get("type", ""))).strip().lower()
    if typed in {"explicit", "implicit"}:
        return typed

    if any(k in entity for k in ("implicit_interval", "implicit_probabilities", "possible_years_probabilities")):
        return "implicit"
    if any(k in entity for k in ("explicit_year", "year", "best_estimate", "updated_best_estimate")):
        return "explicit"
    return "explicit"


def normalize_entity(entity_name: str, entity: Any) -> Dict[str, Any]:
    base = dict(entity) if isinstance(entity, dict) else {}
    temporal_type = _infer_temporal_type(base)

    explicit_year = to_int(
        base.get("explicit_year", base.get("updated_best_estimate", base.get("best_estimate", base.get("year"))))
    )
    interval = normalize_interval(
        base.get(
            "implicit_interval",
            base.get(
                "updated_confidence_interval",
                base.get("confidence_interval_95", base.get("possible_years")),
            ),
        ),
        explicit_year,
    )
    if not interval and explicit_year is not None:
        interval = [explicit_year, explicit_year]

    probs_raw = base.get(
        "implicit_probabilities",
        base.get("possible_years_probabilities", base.get("probabilities", [])),
    )
    expected_len = (interval[1] - interval[0] + 1) if interval else 0
    probs = _normalize_probability_list(probs_raw if isinstance(probs_raw, list) else [], expected_len)

    if temporal_type == "explicit":
        if explicit_year is None and interval:
            explicit_year = interval[1]
        if explicit_year is None:
            explicit_year = MIN_YEAR
        interval = [explicit_year, explicit_year]
        probs = [1.0]
    else:
        if not interval:
            year = explicit_year if explicit_year is not None else MIN_YEAR
            interval = [year, year]
        expected_len = interval[1] - interval[0] + 1
        if expected_len <= 1:
            probs = [1.0]
        else:
            probs = _normalize_probability_list(probs, expected_len)

    best_estimate = explicit_year
    if temporal_type == "implicit" and interval:
        if probs and len(probs) == (interval[1] - interval[0] + 1):
            idx = max(range(len(probs)), key=lambda i: probs[i])
            best_estimate = interval[0] + idx
        else:
            best_estimate = interval[0]

    out = dict(base)
    out["entity_name"] = entity_name
    out["temporal_type"] = temporal_type
    out["explicit_year"] = explicit_year if temporal_type == "explicit" else None
    out["implicit_interval"] = interval if temporal_type == "implicit" else []
    out["implicit_probabilities"] = probs if temporal_type == "implicit" else []
    out["best_estimate"] = best_estimate if best_estimate is not None else ""
    out["confidence_interval_95"] = interval
    return out


def _clip_distribution(interval: List[int], probs: List[float], lower_year: int) -> Tuple[List[int], List[float]]:
    if not interval or len(interval) != 2:
        y = max(MIN_YEAR, lower_year)
        return [y, y], [1.0]
    low, high = interval
    if high <= lower_year:
        y = max(MIN_YEAR, lower_year)
        return [y, y], [1.0]
    if lower_year <= low:
        return [max(MIN_YEAR, low), high], probs

    start = lower_year
    offset = start - low
    head_mass = sum(probs[: offset + 1])
    tail = probs[offset + 1 :]
    clipped = [head_mass] + tail
    clipped = _normalize_probability_list(clipped, len(clipped))
    return [max(MIN_YEAR, start), high], clipped


def merge_sample_temporal_fields(entities: Dict[str, Any], fallback_year: int = MIN_YEAR) -> Dict[str, Any]:
    normalized_entities: Dict[str, Dict[str, Any]] = {}
    for name, ent in (entities or {}).items():
        normalized_entities[name] = normalize_entity(name, ent)

    explicit_years: List[int] = []
    implicit_candidates: List[Tuple[str, List[int], List[float]]] = []
    for name, ent in normalized_entities.items():
        if ent.get("temporal_type") == "explicit":
            year = to_int(ent.get("explicit_year"))
            if year is not None:
                explicit_years.append(year)
        elif ent.get("temporal_type") == "implicit":
            interval = normalize_interval(ent.get("implicit_interval"))
            if not interval:
                continue
            probs = ent.get("implicit_probabilities", [])
            probs = _normalize_probability_list(probs if isinstance(probs, list) else [], interval[1] - interval[0] + 1)
            implicit_candidates.append((name, interval, probs))

    latest_explicit_year = max(explicit_years) if explicit_years else None
    latest_explicit_floor = max(MIN_YEAR, latest_explicit_year) if latest_explicit_year is not None else None

    if not normalized_entities:
        return {
            "entities": normalized_entities,
            "sample_temporal_type": "timeless",
            "latest_explicit_year": None,
            "possible_years": [fallback_year, fallback_year],
            "possible_years_probabilities": [1.0],
            "year": fallback_year,
            "dominant_implicit_entity": None,
        }

    dominant_implicit = None
    if implicit_candidates:
        dominant_implicit = max(
            implicit_candidates,
            key=lambda item: (
                item[1][1],
                item[1][0],
                sum((item[1][0] + idx) * p for idx, p in enumerate(item[2])),
            ),
        )

    if dominant_implicit is None:
        explicit_year = latest_explicit_floor if latest_explicit_floor is not None else fallback_year
        return {
            "entities": normalized_entities,
            "sample_temporal_type": "explicit",
            "latest_explicit_year": latest_explicit_floor,
            "possible_years": [explicit_year, explicit_year],
            "possible_years_probabilities": [1.0],
            "year": explicit_year,
            "dominant_implicit_entity": None,
        }

    imp_name, imp_interval, imp_probs = dominant_implicit
    imp_low, imp_high = imp_interval
    if latest_explicit_floor is None:
        clipped_interval = [max(MIN_YEAR, imp_low), imp_high]
        clipped_probs = imp_probs
        sample_type = "implicit" if imp_high > clipped_interval[0] else "explicit"
    elif imp_high > latest_explicit_floor:
        clipped_interval, clipped_probs = _clip_distribution(imp_interval, imp_probs, latest_explicit_floor)
        sample_type = "implicit"
    else:
        clipped_interval = [latest_explicit_floor, latest_explicit_floor]
        clipped_probs = [1.0]
        sample_type = "explicit"

    clipped_probs = _normalize_probability_list(clipped_probs, clipped_interval[1] - clipped_interval[0] + 1)
    conservative_year = max(MIN_YEAR, clipped_interval[1])
    return {
        "entities": normalized_entities,
        "sample_temporal_type": sample_type,
        "latest_explicit_year": latest_explicit_floor,
        "possible_years": clipped_interval,
        "possible_years_probabilities": clipped_probs,
        "year": conservative_year,
        "dominant_implicit_entity": imp_name,
    }


def normalize_sample_prediction(payload: Dict[str, Any], fallback_year: int = MIN_YEAR) -> Dict[str, Any]:
    row = dict(payload or {})
    entities = row.get("entities", {})
    if not isinstance(entities, dict):
        entities = {}

    merged = merge_sample_temporal_fields(entities, fallback_year=fallback_year)
    row["entities"] = merged["entities"]
    row["sample_temporal_type"] = merged["sample_temporal_type"]
    row["latest_explicit_year"] = merged["latest_explicit_year"]
    row["possible_years"] = merged["possible_years"]
    row["possible_years_probabilities"] = merged["possible_years_probabilities"]
    row["dominant_implicit_entity"] = merged["dominant_implicit_entity"]

    parsed_year = to_int(row.get("year"))
    if parsed_year is None:
        parsed_year = merged["year"]
    parsed_year = max(MIN_YEAR, min(MAX_YEAR, parsed_year))
    row["year"] = max(parsed_year, merged["year"])
    return row


def sample_upper_year(sample: Dict[str, Any]) -> int:
    possible = sample.get("possible_years")
    if isinstance(possible, list) and len(possible) == 2:
        high = to_int(possible[1])
        if high is not None:
            return max(MIN_YEAR, min(MAX_YEAR, high))
    year = to_int(sample.get("year"))
    if year is not None:
        return max(MIN_YEAR, min(MAX_YEAR, year))
    return MIN_YEAR


def aggregate_sample_outputs(
    samples: Dict[str, Dict[str, Any]],
    reconcile_explicit_year: Callable[[str, Dict[str, Any], str, Dict[str, Any]], Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    if not samples:
        return {
            "year": MIN_YEAR,
            "sample_temporal_type": "timeless",
            "latest_explicit_year": None,
            "possible_years": [MIN_YEAR, MIN_YEAR],
            "possible_years_probabilities": [1.0],
            "explicit_year_reconciliation": None,
        }

    normalized = {key: normalize_sample_prediction(sample) for key, sample in samples.items()}
    explicit_candidates: List[Tuple[str, int]] = []
    implicit_candidates: List[Tuple[str, List[int], List[float]]] = []
    for key, sample in normalized.items():
        explicit = to_int(sample.get("latest_explicit_year"))
        if explicit is not None:
            explicit_candidates.append((key, explicit))

        possible = sample.get("possible_years")
        probs = sample.get("possible_years_probabilities", [])
        interval = normalize_interval(possible)
        if interval:
            if len(interval) == 2 and interval[1] > interval[0]:
                implicit_candidates.append(
                    (
                        key,
                        interval,
                        _normalize_probability_list(
                            probs if isinstance(probs, list) else [],
                            interval[1] - interval[0] + 1,
                        ),
                    )
                )

    reconciliation = None
    latest_explicit = max([year for _, year in explicit_candidates], default=None)
    if reconcile_explicit_year and len(explicit_candidates) == 2:
        (key_a, year_a), (key_b, year_b) = explicit_candidates
        if year_a != year_b:
            rec = reconcile_explicit_year(key_a, normalized[key_a], key_b, normalized[key_b])
            judged = to_int(rec.get("judge_explicit_year")) if isinstance(rec, dict) else None
            if judged is not None:
                latest_explicit = judged
            reconciliation = rec

    dominant_implicit = None
    if implicit_candidates:
        dominant_implicit = max(implicit_candidates, key=lambda item: (item[1][1], item[1][0]))

    if dominant_implicit is None:
        explicit_year = max(MIN_YEAR, latest_explicit) if latest_explicit is not None else MIN_YEAR
        return {
            "year": explicit_year,
            "sample_temporal_type": "explicit" if explicit_candidates else "timeless",
            "latest_explicit_year": explicit_year if explicit_candidates else None,
            "possible_years": [explicit_year, explicit_year],
            "possible_years_probabilities": [1.0],
            "explicit_year_reconciliation": reconciliation,
        }

    _, interval, probs = dominant_implicit
    low, high = interval
    latest_explicit_floor = max(MIN_YEAR, latest_explicit) if latest_explicit is not None else None
    if latest_explicit_floor is None:
        possible = [max(MIN_YEAR, low), high]
        weights = probs
        sample_type = "implicit"
    elif high > latest_explicit_floor:
        possible, weights = _clip_distribution(interval, probs, latest_explicit_floor)
        sample_type = "implicit"
    else:
        possible = [latest_explicit_floor, latest_explicit_floor]
        weights = [1.0]
        sample_type = "explicit"

    weights = _normalize_probability_list(weights, possible[1] - possible[0] + 1)
    return {
        "year": max(MIN_YEAR, min(MAX_YEAR, possible[1])),
        "sample_temporal_type": sample_type,
        "latest_explicit_year": latest_explicit_floor,
        "possible_years": possible,
        "possible_years_probabilities": weights,
        "explicit_year_reconciliation": reconciliation,
    }


def normalize_entities(entities: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(entities, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for name, entity in entities.items():
        key = str(name).strip() or "entity"
        out[key] = normalize_entity(key, entity)
    return out


def _extract_year_from_text(text: str) -> int | None:
    match = YEAR_RE.search(text or "")
    if not match:
        return None
    year = to_int(match.group(1))
    if year is None:
        return None
    return max(MIN_YEAR, min(MAX_YEAR, year))


def normalize_prediction_payload(payload: Any, fallback_text: str = "", parsed_ok: bool = True) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
        parsed_ok = False
    fallback_year = to_int(payload.get("year"))
    if fallback_year is None:
        fallback_year = _extract_year_from_text(fallback_text) or MIN_YEAR
    normalized = normalize_sample_prediction(payload, fallback_year=fallback_year)
    if not parsed_ok:
        normalized["raw_response"] = fallback_text
    return normalized


def aggregate_row_samples(
    samples: Dict[str, Dict[str, Any]],
    reconcile_explicit_fn: Callable[[str, Dict[str, Any], str, Dict[str, Any]], Tuple[int | None, Dict[str, Any]]] | None = None,
) -> Dict[str, Any]:
    if reconcile_explicit_fn is None:
        return aggregate_sample_outputs(samples)

    def _adapter(key_a: str, sample_a: Dict[str, Any], key_b: str, sample_b: Dict[str, Any]) -> Dict[str, Any]:
        year, details = reconcile_explicit_fn(key_a, sample_a, key_b, sample_b)
        out = dict(details or {})
        out["judge_explicit_year"] = year
        return out

    return aggregate_sample_outputs(samples, reconcile_explicit_year=_adapter)


def apply_temporal_fields(row: Dict[str, Any], fields: Dict[str, Any]) -> None:
    row["year"] = fields.get("year", row.get("year", MIN_YEAR))
    row["sample_temporal_type"] = fields.get("sample_temporal_type", row.get("sample_temporal_type", "timeless"))
    row["latest_explicit_year"] = fields.get("latest_explicit_year")
    row["possible_years"] = fields.get("possible_years", [row["year"], row["year"]])
    row["possible_years_probabilities"] = fields.get("possible_years_probabilities", [1.0])
    if "explicit_year_reconciliation" in fields and fields.get("explicit_year_reconciliation") is not None:
        row["explicit_year_reconciliation"] = fields.get("explicit_year_reconciliation")


def iter_samples(row: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    sample_keys = [k for k in row.keys() if k.startswith("sample_") and isinstance(row.get(k), dict)]
    if sample_keys:
        return [(key, row[key]) for key in sorted(sample_keys)]
    return [("sample_1", row)]


def sample_year(sample: Dict[str, Any]) -> int | None:
    return sample_upper_year(sample)
