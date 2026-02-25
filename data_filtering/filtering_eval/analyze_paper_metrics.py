#!/usr/bin/env python3
"""Compute paper-aligned metrics for three stages and plot explicit/implicit separately."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from temporal_schema import MAX_YEAR, MIN_YEAR, normalize_interval, to_int

FAILURE_BUCKETS = {
    1: "Entity extraction",
    2: "Query quality",
    3: "Search failure",
    4: "Gold label wrong",
    5: "Other",
}

FAILURE_COLORS = {
    1: "#F58518",
    2: "#E45756",
    3: "#B279A2",
    4: "#72B7B2",
    5: "#54A24B",
}

STAGES = [
    ("pre_search_sample_1", "Pre-search (sample_1)"),
    ("post_search_sample_1", "Post-search (sample_1)"),
    ("post_search_agg_2samples", "Post-search + 2-sample aggregation"),
]


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_type(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"explicit", "implicit"}:
        return text
    if text == "timeless":
        return "explicit"
    return "explicit"


def normalize_probs(values: List[float], expected_len: int) -> List[float]:
    if expected_len <= 0:
        return []
    if len(values) != expected_len:
        return [1.0 / expected_len] * expected_len
    clipped = [max(0.0, float(v)) for v in values]
    total = sum(clipped)
    if total <= 0.0:
        return [1.0 / expected_len] * expected_len
    out = [v / total for v in clipped]
    out[0] += 1.0 - sum(out)
    return out


def interval_probs_from_sample(sample: dict) -> Tuple[List[int], List[float], int]:
    possible = sample.get("possible_years")
    probs = sample.get("possible_years_probabilities", [])
    interval = normalize_interval(possible)
    if interval:
        low, high = interval
        low = max(MIN_YEAR, min(MAX_YEAR, low))
        high = max(MIN_YEAR, min(MAX_YEAR, high))
        if high < low:
            low, high = high, low
        span = high - low + 1
        probs_norm = normalize_probs([float(x) for x in probs] if isinstance(probs, list) else [], span)
        return [low, high], probs_norm, high

    year = to_int(sample.get("year", sample.get("pred_year")))
    if year is None:
        year = MIN_YEAR
    year = max(MIN_YEAR, min(MAX_YEAR, year))
    return [year, year], [1.0], year


def interval_probs_from_agg_row(row: dict) -> Tuple[List[int], List[float], int]:
    possible = row.get("possible_years")
    probs = row.get("possible_years_probabilities", [])
    interval = normalize_interval(possible)
    if interval:
        low, high = interval
        low = max(MIN_YEAR, min(MAX_YEAR, low))
        high = max(MIN_YEAR, min(MAX_YEAR, high))
        if high < low:
            low, high = high, low
        span = high - low + 1
        probs_norm = normalize_probs([float(x) for x in probs] if isinstance(probs, list) else [], span)
        year = to_int(row.get("updated_year_temporal_merge"))
        if year is None:
            year = high
        year = max(MIN_YEAR, min(MAX_YEAR, year))
        return [low, high], probs_norm, year

    year = to_int(row.get("updated_year_temporal_merge"))
    if year is None:
        year = to_int(row.get("updated_year", row.get("year", row.get("pred_year"))))
    if year is None:
        year = MIN_YEAR
    year = max(MIN_YEAR, min(MAX_YEAR, year))
    return [year, year], [1.0], year


def get_stage_prediction(pre_row: dict, post_row: dict, stage_key: str) -> Tuple[List[int], List[float], int]:
    if stage_key == "pre_search_sample_1":
        sample = pre_row.get("sample_1", pre_row)
        if not isinstance(sample, dict):
            sample = pre_row
        return interval_probs_from_sample(sample)

    if stage_key == "post_search_sample_1":
        sample = post_row.get("sample_1")
        if not isinstance(sample, dict):
            sample = pre_row.get("sample_1", pre_row)
        return interval_probs_from_sample(sample)

    return interval_probs_from_agg_row(post_row)


def pmf_vector(interval: List[int], probs: List[float]) -> List[float]:
    vec = [0.0] * (MAX_YEAR - MIN_YEAR + 1)
    if not interval or len(interval) != 2:
        vec[0] = 1.0
        return vec
    low, high = interval
    span = high - low + 1
    probs = normalize_probs(probs, span)
    for idx, year in enumerate(range(low, high + 1)):
        if MIN_YEAR <= year <= MAX_YEAR:
            vec[year - MIN_YEAR] += probs[idx]
    total = sum(vec)
    if total <= 0.0:
        vec[0] = 1.0
        return vec
    vec = [v / total for v in vec]
    vec[0] += 1.0 - sum(vec)
    return vec


def cdf(vec: List[float]) -> List[float]:
    out: List[float] = []
    running = 0.0
    for v in vec:
        running += v
        out.append(running)
    return out


def w1_discrete(q: List[float], p: List[float]) -> float:
    cq = cdf(q)
    cp = cdf(p)
    return float(sum(abs(a - b) for a, b in zip(cq, cp)))


def earliest_support_year(p: List[float]) -> int:
    for idx, mass in enumerate(p):
        if mass > 0.0:
            return MIN_YEAR + idx
    return MIN_YEAR


def leakage_mass(q: List[float], y_min: int) -> float:
    cut = max(0, min(len(q), y_min - MIN_YEAR))
    return float(sum(q[:cut]))


def load_buckets(path: str) -> Dict[str, int]:
    if not path or not os.path.exists(path):
        return {}
    out: Dict[str, int] = {}
    for row in load_jsonl(path):
        gid = str(row.get("id", "")).strip()
        if not gid:
            continue
        bucket = to_int(row.get("bucket"))
        if bucket is None or bucket not in FAILURE_BUCKETS:
            bucket = 5
        out[gid] = bucket
    return out


def build_delta_plot(
    deltas_by_id: Dict[str, int],
    bucket_by_id: Dict[str, int],
    out_path: str,
    model_name: str,
) -> None:
    if not deltas_by_id:
        return
    vals = list(deltas_by_id.values())
    min_d = min(vals)
    max_d = max(vals)
    bins = list(range(min_d, max_d + 1))

    neg_counts = {k: {b: 0 for b in bins} for k in FAILURE_BUCKETS}
    pos_counts = {b: 0 for b in bins}
    for gid, delta in deltas_by_id.items():
        if delta < 0:
            bucket = bucket_by_id.get(gid, 5)
            if bucket not in neg_counts:
                bucket = 5
            neg_counts[bucket][delta] += 1
        else:
            pos_counts[delta] += 1

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for b in bins:
        if b < 0:
            bottom = 0
            for bucket in sorted(FAILURE_BUCKETS):
                count = neg_counts[bucket][b]
                if count:
                    ax.bar(b, count, bottom=bottom, color=FAILURE_COLORS[bucket], width=0.9)
                bottom += count
        else:
            count = pos_counts[b]
            if count:
                ax.bar(b, count, color="#1f77b4", width=0.9)

    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Predicted year - Gold year")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name}: post-search + aggregation error")
    handles = [Rectangle((0, 0), 1, 1, color=FAILURE_COLORS[k]) for k in sorted(FAILURE_BUCKETS)]
    labels = [FAILURE_BUCKETS[k] for k in sorted(FAILURE_BUCKETS)]
    ax.legend(handles, labels, loc="upper left", fontsize=8, title="Negative-delta failures")
    fig.savefig(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-metric analysis for one model and three stages.")
    parser.add_argument("--preds", required=True, help="Original predictions JSONL")
    parser.add_argument("--updated", required=True, help="Post-search updated predictions JSONL")
    parser.add_argument("--gold-path", required=True, help="Gold dataset JSONL")
    parser.add_argument("--gold-year-field", default="gold_year")
    parser.add_argument("--gold-type-field", default="sample_temporal_type")
    parser.add_argument("--nonconservative", default="", help="Optional non-conservative bucket JSONL")
    parser.add_argument("--lambda-penalty", type=float, default=1.0, help="Leakage penalty lambda")
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-explicit-plot", required=True)
    parser.add_argument("--out-implicit-plot", required=True)
    parser.add_argument("--out-delta-plot", required=True)
    args = parser.parse_args()

    pre_rows = {str(r["id"]): r for r in load_jsonl(args.preds)}
    post_rows = {str(r["id"]): r for r in load_jsonl(args.updated)}
    gold_rows = {str(r["id"]): r for r in load_jsonl(args.gold_path)}
    bucket_by_id = load_buckets(args.nonconservative)

    stage_state: Dict[str, Dict[str, object]] = {}
    for stage_key, stage_label in STAGES:
        stage_state[stage_key] = {
            "label": stage_label,
            "explicit_total": 0,
            "explicit_match": 0,
            "implicit_scores": [],
            "implicit_noleak": [],
        }

    deltas_stage3: Dict[str, int] = {}
    reconciliation_count = 0

    for gid, grow in gold_rows.items():
        if args.gold_year_field not in grow:
            continue
        gold_year = int(grow[args.gold_year_field])
        g_type = normalize_type(grow.get(args.gold_type_field))

        g_interval, g_probs = [gold_year, gold_year], [1.0]
        if g_type == "implicit":
            g_possible = normalize_interval(grow.get("possible_years"))
            if g_possible:
                g_interval = g_possible
                g_probs = normalize_probs(
                    [float(x) for x in grow.get("possible_years_probabilities", [])]
                    if isinstance(grow.get("possible_years_probabilities"), list)
                    else [],
                    g_interval[1] - g_interval[0] + 1,
                )
        p_gt = pmf_vector(g_interval, g_probs)
        y_min = earliest_support_year(p_gt)

        pre_row = pre_rows.get(gid, {})
        post_row = post_rows.get(gid, {})
        if isinstance(post_row.get("explicit_year_reconciliation"), dict):
            reconciliation_count += 1

        for stage_key, _ in STAGES:
            s = stage_state[stage_key]
            interval, probs, year = get_stage_prediction(pre_row, post_row, stage_key)

            if g_type == "explicit":
                s["explicit_total"] = int(s["explicit_total"]) + 1
                if int(year) == gold_year:
                    s["explicit_match"] = int(s["explicit_match"]) + 1
            else:
                q = pmf_vector(interval, probs)
                leak = leakage_mass(q, y_min)
                score = w1_discrete(q, p_gt) + args.lambda_penalty * leak
                scores = s["implicit_scores"]
                noleaks = s["implicit_noleak"]
                assert isinstance(scores, list) and isinstance(noleaks, list)
                scores.append(float(score))
                noleaks.append(1.0 if leak == 0.0 else 0.0)

            if stage_key == "post_search_agg_2samples":
                deltas_stage3[gid] = int(year) - gold_year

    explicit_values: List[float] = []
    implicit_values: List[float] = []
    implicit_noleak_values: List[float | None] = []
    stage_labels: List[str] = []
    stages_json: Dict[str, dict] = {}

    for stage_key, stage_label in STAGES:
        s = stage_state[stage_key]
        exp_total = int(s["explicit_total"])
        exp_match = int(s["explicit_match"])
        exp_acc = (exp_match / exp_total) if exp_total else 0.0

        imp_scores = s["implicit_scores"]
        imp_noleak = s["implicit_noleak"]
        assert isinstance(imp_scores, list) and isinstance(imp_noleak, list)
        imp_score_mean = (sum(imp_scores) / len(imp_scores)) if imp_scores else None
        imp_noleak_mean = (sum(imp_noleak) / len(imp_noleak)) if imp_noleak else None

        stage_labels.append(stage_label)
        explicit_values.append(exp_acc)
        implicit_values.append(float(imp_score_mean) if imp_score_mean is not None else 0.0)
        implicit_noleak_values.append(imp_noleak_mean)

        stages_json[stage_key] = {
            "label": stage_label,
            "explicit_n": exp_total,
            "explicit_exact": exp_acc,
            "implicit_n": len(imp_scores),
            "implicit_score_lambda": imp_score_mean,
            "implicit_noleak": imp_noleak_mean,
        }

    summary = {
        "model": args.model_name,
        "lambda_penalty": args.lambda_penalty,
        "stages": stages_json,
        "reconciliation_rows_stage3": reconciliation_count,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Explicit-only figure
    fig_exp, ax_exp = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax_exp.bar(stage_labels, explicit_values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax_exp.set_ylim(0, 1)
    ax_exp.set_ylabel("Exact-year accuracy")
    ax_exp.set_title(f"{args.model_name}: explicit metric (higher is better)")
    ax_exp.tick_params(axis="x", labelrotation=15)
    fig_exp.savefig(args.out_explicit_plot)

    # Implicit-only figure with separate subplots for Score_lambda and NoLeak.
    fig_imp, (ax_imp_score, ax_imp_noleak) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
        constrained_layout=True,
    )
    has_implicit = any(stages_json[k]["implicit_n"] > 0 for k, _ in STAGES)
    if not has_implicit:
        ax_imp_score.axis("off")
        ax_imp_noleak.axis("off")
        ax_imp_score.text(0.5, 0.5, "No implicit gold samples", ha="center", va="center", fontsize=12)
    else:
        x = list(range(len(stage_labels)))
        ax_imp_score.bar(x, implicit_values, color=["#4C78A8", "#F58518", "#54A24B"])
        ax_imp_score.set_ylabel(f"Score_lambda (lambda={args.lambda_penalty:g})")
        ax_imp_score.set_title(f"{args.model_name}: implicit metrics")
        ax_imp_score.set_xticks(x, stage_labels, rotation=15, ha="right")

        noleak_vals = [
            float(stages_json[stage_key]["implicit_noleak"])
            if stages_json[stage_key]["implicit_noleak"] is not None
            else 0.0
            for stage_key, _ in STAGES
        ]
        ax_imp_noleak.bar(x, noleak_vals, color="#2E7D32")
        ax_imp_noleak.set_ylim(0, 1)
        ax_imp_noleak.set_ylabel("NoLeak")
        ax_imp_noleak.set_xticks(x, stage_labels, rotation=15, ha="right")
        ax_imp_noleak.set_xlabel("Stage")
        for idx, val in enumerate(noleak_vals):
            ax_imp_noleak.text(idx, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    fig_imp.savefig(args.out_implicit_plot)

    build_delta_plot(deltas_stage3, bucket_by_id, args.out_delta_plot, args.model_name)

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_explicit_plot}")
    print(f"Wrote {args.out_implicit_plot}")
    print(f"Wrote {args.out_delta_plot}")


if __name__ == "__main__":
    main()
