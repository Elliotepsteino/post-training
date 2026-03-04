#!/usr/bin/env python3
"""Create charts from three-agent synthetic evaluation results."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from statistics import median
from typing import Dict, List

import matplotlib.pyplot as plt


def _load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _to_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _is_correct(pred: int | None, gt: int | None) -> bool:
    return pred is not None and gt is not None and int(pred) == int(gt)


def _ratio(pair: List[float]) -> float:
    return 100.0 * pair[0] / pair[1] if pair[1] else 0.0


def _mean_ci_half_width(values: List[float], z: float = 1.96) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(max(0.0, var / n))
    return z * se


def _clip_ci(err: float, accuracy: float) -> tuple[float, float]:
    """Clip asymmetric CI whiskers so they stay within [0, 100]."""
    capped = max(0.0, min(float(err), 100.0))
    lower = min(capped, max(0.0, accuracy))
    upper = min(capped, max(0.0, 100.0 - accuracy))
    return lower, upper


def _slugify_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", (model or "model").lower()).strip("-")


def _empty_stage_metrics(model_keys: List[str] | None = None) -> dict[str, list[float]]:
    keys = list(model_keys or ["agent-1", "agent-2", "agent-3"])
    return {
        "proposal1_ensemble": [0.0, 0],
        "aggregation1": [0.0, 0],
        "proposal2_ensemble": [0.0, 0],
        "aggregation2": [0.0, 0],
        **{f"proposal1_{k}": [0.0, 0] for k in keys},
        **{f"proposal2_{k}": [0.0, 0] for k in keys},
    }


def _aggregated_prediction(stage: dict) -> tuple[int | None, List[int], List[float]]:
    year = _to_int(stage.get("aggregated_year"))
    years = stage.get("aggregated_plausible_years") or []
    probs = stage.get("aggregated_plausible_years_prob") or []
    if year is not None or years or probs:
        return year, years, probs

    # Backward compatibility with judge output naming.
    year = _to_int(stage.get("judge_year"))
    years = stage.get("judge_plausible_years") or []
    probs = stage.get("judge_plausible_years_prob") or []
    return year, years, probs


def _extract_distribution(
    fallback_year: int | None,
    years: List[int] | None,
    probs: List[float] | None,
) -> tuple[List[int], List[float]]:
    if years and probs and len(years) == len(probs):
        buckets: dict[int, float] = defaultdict(float)
        for y, p in zip(years, probs):
            yi = _to_int(y)
            try:
                pf = float(p)
            except (TypeError, ValueError):
                continue
            if yi is None or pf < 0:
                continue
            buckets[yi] += pf
        if buckets:
            sorted_items = sorted(buckets.items())
            ys = [yi for yi, _ in sorted_items]
            ps = [float(p) for _, p in sorted_items]
            total = sum(ps)
            if total > 0:
                ps = [p / total for p in ps]
                return ys, ps

    if fallback_year is None:
        return [2001], [1.0]
    return [fallback_year], [1.0]


def _wasserstein_distance(
    gt: int | None,
    years: List[int] | None,
    probs: List[float] | None,
    fallback_year: int | None,
) -> float:
    if gt is None:
        return 0.0
    ys, ps = _extract_distribution(fallback_year, years, probs)
    return float(sum(p * abs(y - gt) for y, p in zip(ys, ps)))


def compute_metrics(
    rows: List[dict],
    model_names: List[str] | None = None,
) -> tuple[dict, dict, dict, List[str], List[str]]:
    model_keys = [_slugify_model_name(m) for m in (model_names or ["1", "2", "3"])]
    if len(model_keys) != 3:
        model_keys = ["agent-1", "agent-2", "agent-3"][:3]

    model_display = [
        m.strip() if isinstance(m, str) and m.strip() else f"Model {i + 1}"
        for i, m in enumerate((model_names or [])[:3])
    ]
    while len(model_display) < 3:
        model_display.append(f"Model {len(model_display) + 1}")
    if len(model_display) > 3:
        model_display = model_display[:3]

    proposal1_keys = {f"agent_{idx + 1}": model_keys[idx] for idx in range(3)}
    proposal2_keys = {f"agent_{idx + 1}": model_keys[idx] for idx in range(3)}

    totals = _empty_stage_metrics(model_keys)
    by_category = defaultdict(lambda: _empty_stage_metrics(model_keys))
    by_question_type = defaultdict(lambda: _empty_stage_metrics(model_keys))

    totals_samples: dict[str, List[float]] = {k: [] for k in _empty_stage_metrics(model_keys)}
    cat_samples: defaultdict[str, dict[str, List[float]]] = defaultdict(
        lambda: {k: [] for k in _empty_stage_metrics(model_keys)}
    )
    type_samples: defaultdict[str, dict[str, List[float]]] = defaultdict(
        lambda: {k: [] for k in _empty_stage_metrics(model_keys)}
    )

    for row in rows:
        gt = _to_int(row.get("gold_year"))
        s1 = row.get("stage_1", {})
        s2 = row.get("stage_2", {})
        s3 = row.get("stage_3", {})
        s4 = row.get("stage_4", {})
        cat = row.get("category", "unknown")
        qtype = row.get("question_type", "unknown")

        proposal1_corrects = []
        for key, model_key in proposal1_keys.items():
            p = s1.get(key, {}).get("year")
            stage_key = f"proposal1_{model_key}"
            totals[stage_key][1] += 1
            by_category[cat][stage_key][1] += 1
            by_question_type[qtype][stage_key][1] += 1
            c = 1.0 if _is_correct(_to_int(p), gt) else 0.0
            if c:
                totals[stage_key][0] += 1
                by_category[cat][stage_key][0] += 1
                by_question_type[qtype][stage_key][0] += 1
            proposal1_corrects.append(c)
            totals_samples[stage_key].append(c)
            cat_samples[cat][stage_key].append(c)
            type_samples[qtype][stage_key].append(c)

        proposal1_ensemble = sum(proposal1_corrects) / 3
        totals["proposal1_ensemble"][0] += proposal1_ensemble
        totals["proposal1_ensemble"][1] += 1
        by_category[cat]["proposal1_ensemble"][1] += 1
        by_category[cat]["proposal1_ensemble"][0] += proposal1_ensemble
        by_question_type[qtype]["proposal1_ensemble"][1] += 1
        by_question_type[qtype]["proposal1_ensemble"][0] += proposal1_ensemble
        totals_samples["proposal1_ensemble"].append(proposal1_ensemble)
        cat_samples[cat]["proposal1_ensemble"].append(proposal1_ensemble)
        type_samples[qtype]["proposal1_ensemble"].append(proposal1_ensemble)

        agg1_year, _, _ = _aggregated_prediction(s2)
        totals["aggregation1"][1] += 1
        by_category[cat]["aggregation1"][1] += 1
        by_question_type[qtype]["aggregation1"][1] += 1
        a1 = 1.0 if _is_correct(agg1_year, gt) else 0.0
        if a1:
            totals["aggregation1"][0] += 1
            by_category[cat]["aggregation1"][0] += 1
            by_question_type[qtype]["aggregation1"][0] += 1
        totals_samples["aggregation1"].append(a1)
        cat_samples[cat]["aggregation1"].append(a1)
        type_samples[qtype]["aggregation1"].append(a1)

        proposal2_stage = s3 if "agent_1" in s3 else s2
        proposal2_corrects = []
        for key, model_key in proposal2_keys.items():
            p = proposal2_stage.get(key, {}).get("year")
            stage_key = f"proposal2_{model_key}"
            totals[stage_key][1] += 1
            by_category[cat][stage_key][1] += 1
            by_question_type[qtype][stage_key][1] += 1
            c = 1.0 if _is_correct(_to_int(p), gt) else 0.0
            if c:
                totals[stage_key][0] += 1
                by_category[cat][stage_key][0] += 1
                by_question_type[qtype][stage_key][0] += 1
            proposal2_corrects.append(c)
            totals_samples[stage_key].append(c)
            cat_samples[cat][stage_key].append(c)
            type_samples[qtype][stage_key].append(c)

        proposal2_ensemble = sum(proposal2_corrects) / 3
        totals["proposal2_ensemble"][0] += proposal2_ensemble
        totals["proposal2_ensemble"][1] += 1
        by_category[cat]["proposal2_ensemble"][1] += 1
        by_category[cat]["proposal2_ensemble"][0] += proposal2_ensemble
        by_question_type[qtype]["proposal2_ensemble"][1] += 1
        by_question_type[qtype]["proposal2_ensemble"][0] += proposal2_ensemble
        totals_samples["proposal2_ensemble"].append(proposal2_ensemble)
        cat_samples[cat]["proposal2_ensemble"].append(proposal2_ensemble)
        type_samples[qtype]["proposal2_ensemble"].append(proposal2_ensemble)

        agg2_source = s4 if s4 else s3
        agg2_year, _, _ = _aggregated_prediction(agg2_source)
        totals["aggregation2"][1] += 1
        by_category[cat]["aggregation2"][1] += 1
        by_question_type[qtype]["aggregation2"][1] += 1
        a2 = 1.0 if _is_correct(agg2_year, gt) else 0.0
        if a2:
            totals["aggregation2"][0] += 1
            by_category[cat]["aggregation2"][0] += 1
            by_question_type[qtype]["aggregation2"][0] += 1
        totals_samples["aggregation2"].append(a2)
        cat_samples[cat]["aggregation2"].append(a2)
        type_samples[qtype]["aggregation2"].append(a2)

    overall = {}
    for key, val in totals.items():
        overall[key] = {
            "correct": float(val[0]),
            "total": int(val[1]),
            "accuracy": _ratio(val),
            "accuracy_stderr": _mean_ci_half_width(totals_samples[key]) * 100.0,
        }

    by_cat = {}
    for cat, vals in by_category.items():
        by_cat[cat] = {
            key: {
                "correct": float(val[0]),
                "total": int(val[1]),
                "accuracy": _ratio(val),
                "accuracy_stderr": _mean_ci_half_width(cat_samples[cat][key]) * 100.0,
            }
            for key, val in vals.items()
        }

    by_type = {}
    for qtype, vals in by_question_type.items():
        by_type[qtype] = {
            key: {
                "correct": float(val[0]),
                "total": int(val[1]),
                "accuracy": _ratio(val),
                "accuracy_stderr": _mean_ci_half_width(type_samples[qtype][key]) * 100.0,
            }
            for key, val in vals.items()
        }

    return overall, by_cat, by_type, model_keys, model_display


def _compute_wasserstein_for_question_types(
    rows: List[dict],
    question_types: set[str],
) -> dict[str, dict[str, float]]:
    target_rows = [row for row in rows if row.get("question_type") in question_types]
    if not target_rows:
        return {}

    stage_keys = [
        "proposal_round_1",
        "aggregation_round_1",
        "proposal_round_2",
        "aggregation_round_2",
    ]
    samples: dict[str, List[float]] = {k: [] for k in stage_keys}

    for row in target_rows:
        gt = _to_int(row.get("gold_year"))
        s1 = row.get("stage_1", {})
        s2 = row.get("stage_2", {})
        s3 = row.get("stage_3", {})
        s4 = row.get("stage_4", {})

        s1_dist = []
        for key in ("agent_1", "agent_2", "agent_3"):
            pred = s1.get(key, {})
            s1_dist.append(
                _wasserstein_distance(
                    gt,
                    pred.get("plausible_years", []),
                    pred.get("plausible_years_prob", []),
                    _to_int(pred.get("year")),
                )
            )
        if s1_dist:
            samples["proposal_round_1"].append(sum(s1_dist) / 3)

        agg1_year, agg1_years, agg1_probs = _aggregated_prediction(s2)
        samples["aggregation_round_1"].append(
            _wasserstein_distance(gt, agg1_years, agg1_probs, agg1_year)
        )

        proposal2_stage = s3 if "agent_1" in s3 else s2
        s3_dist = []
        for key in ("agent_1", "agent_2", "agent_3"):
            pred = proposal2_stage.get(key, {})
            s3_dist.append(
                _wasserstein_distance(
                    gt,
                    pred.get("plausible_years", []),
                    pred.get("plausible_years_prob", []),
                    _to_int(pred.get("year")),
                )
            )
        if s3_dist:
            samples["proposal_round_2"].append(sum(s3_dist) / 3)

        agg2_source = s4 if s4 else s3
        agg2_year, agg2_years, agg2_probs = _aggregated_prediction(agg2_source)
        samples["aggregation_round_2"].append(
            _wasserstein_distance(gt, agg2_years, agg2_probs, agg2_year)
        )

    return {
        stage: {
            "median_distance": float(median(values)),
            "mean_distance": float(sum(values) / len(values)),
            "distance_stderr": float(_mean_ci_half_width(values)),
            "count": len(values),
        }
        for stage, values in samples.items()
        if values
    }


def compute_implicit_wasserstein(rows: List[dict]) -> dict[str, dict[str, float]]:
    return _compute_wasserstein_for_question_types(rows, {"implicit"})


def compute_multi_implicit_wasserstein(rows: List[dict]) -> dict[str, dict[str, float]]:
    return _compute_wasserstein_for_question_types(rows, {"multi_implicit"})


def _stage_label(key: str) -> str:
    names = {
        "proposal1_ensemble": "Proposal round 1 (ensemble)",
        "aggregation1": "Aggregation round 1",
        "proposal2_ensemble": "Proposal round 2 (ensemble)",
        "aggregation2": "Aggregation round 2",
        "proposal_round_1": "Proposal round 1",
        "aggregation_round_1": "Aggregation round 1",
        "proposal_round_2": "Proposal round 2",
        "aggregation_round_2": "Aggregation round 2",
    }
    return names.get(key, key)


def _label(key: str) -> str:
    return _stage_label(key)


def _plot_model_pre_post_accuracy(
    out_dir: str,
    overall: dict,
    model_keys: List[str],
    model_labels: List[str],
) -> None:
    if len(model_keys) != 3:
        return

    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    x = list(range(len(model_keys)))
    bar_width = 0.32

    pre_acc, post_acc = [], []
    pre_err, post_err = [], []
    for model_key in model_keys[:3]:
        pre = overall.get(f"proposal1_{model_key}", {})
        post = overall.get(f"proposal2_{model_key}", {})
        pre_acc.append(pre.get("accuracy", 0.0))
        post_acc.append(post.get("accuracy", 0.0))

        pre_ci = _clip_ci(
            pre.get("accuracy_stderr", 0.0),
            pre.get("accuracy", 0.0),
        )
        post_ci = _clip_ci(
            post.get("accuracy_stderr", 0.0),
            post.get("accuracy", 0.0),
        )
        pre_err.append(pre_ci)
        post_err.append(post_ci)

    pre_low = [v[0] for v in pre_err]
    pre_high = [v[1] for v in pre_err]
    post_low = [v[0] for v in post_err]
    post_high = [v[1] for v in post_err]

    ax.bar(
        [xi - bar_width / 2 for xi in x],
        pre_acc,
        width=bar_width,
        yerr=[pre_low, pre_high],
        capsize=4,
        label="Proposal round 1",
    )
    ax.bar(
        [xi + bar_width / 2 for xi in x],
        post_acc,
        width=bar_width,
        yerr=[post_low, post_high],
        capsize=4,
        label="Proposal round 2",
    )
    ax.set_title("Per-model exact-match accuracy: proposal round 1 vs proposal round 2")
    ax.set_ylabel("Exact match accuracy (%)")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels[:3], rotation=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(os.path.join(out_dir, "three_agent_model_pre_post_accuracy.pdf"))
    plt.close(fig)


def write_plots(
    out_dir: str,
    overall: dict,
    by_type: dict,
    implicit_wasserstein: dict,
    multi_implicit_wasserstein: dict,
    model_keys: List[str],
    model_labels: List[str],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plot_types = ["overall", "explicit", "explicit_multi", "implicit", "multi_implicit"]
    plot_types = [t for t in plot_types if t == "overall" or t in by_type]

    stage_keys = ["proposal1_ensemble", "aggregation1", "proposal2_ensemble", "aggregation2"]
    type_to_metrics = {
        "overall": overall,
        "explicit": by_type.get("explicit", {}),
        "explicit_multi": by_type.get("explicit_multi", {}),
        "implicit": by_type.get("implicit", {}),
        "multi_implicit": by_type.get("multi_implicit", {}),
    }

    fig, ax = plt.subplots(figsize=(11.2, 4.8), constrained_layout=True)
    x = list(range(len(plot_types)))
    bar_width = 0.18
    group_offset = -(len(stage_keys) - 1) * bar_width / 2
    for i, key in enumerate(stage_keys):
        acc = [type_to_metrics.get(t, {}).get(key, {}).get("accuracy", 0.0) for t in plot_types]
        err = [
            _clip_ci(
                type_to_metrics.get(t, {}).get(key, {}).get("accuracy_stderr", 0.0),
                type_to_metrics.get(t, {}).get(key, {}).get("accuracy", 0.0),
            )
            for t in plot_types
        ]
        low_err = [v[0] for v in err]
        high_err = [v[1] for v in err]
        yerr = [low_err, high_err]
        ax.bar(
            [xi + group_offset + i * bar_width for xi in x],
            acc,
            width=bar_width,
            yerr=yerr,
            capsize=4,
            label=_label(key),
        )
    ax.set_title("MoA synthetic temporal accuracy by question type")
    ax.set_ylabel("Exact match accuracy (%)")
    ax.set_xlabel("Question type")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "-") for t in plot_types], rotation=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(os.path.join(out_dir, "three_agent_accuracy_by_question_type.pdf"))
    plt.close(fig)

    if implicit_wasserstein:
        stage_order = [
            "proposal_round_1",
            "aggregation_round_1",
            "proposal_round_2",
            "aggregation_round_2",
        ]
        medians = [implicit_wasserstein.get(k, {}).get("median_distance", 0.0) for k in stage_order]

        fig2, ax2 = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        x2 = list(range(len(stage_order)))
        ax2.bar(x2, medians, width=0.45)
        ax2.set_title("Implicit-only Wasserstein distance by MoA stage")
        ax2.set_ylabel("Median Earth mover distance (years)")
        ax2.set_xlabel("Pipeline stage")
        ax2.set_xticks(x2)
        ax2.set_xticklabels([_stage_label(k) for k in stage_order], rotation=12)
        ax2.grid(axis="y", alpha=0.2)
        fig2.savefig(os.path.join(out_dir, "three_agent_implicit_wasserstein.pdf"))
        plt.close(fig2)

    if multi_implicit_wasserstein:
        stage_order = [
            "proposal_round_1",
            "aggregation_round_1",
            "proposal_round_2",
            "aggregation_round_2",
        ]
        medians = [multi_implicit_wasserstein.get(k, {}).get("median_distance", 0.0) for k in stage_order]

        fig3, ax3 = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        x3 = list(range(len(stage_order)))
        ax3.bar(x3, medians, width=0.45)
        ax3.set_title("Multi-implicit Wasserstein distance by MoA stage")
        ax3.set_ylabel("Median Earth mover distance (years)")
        ax3.set_xlabel("MoA stage")
        ax3.set_xticks(x3)
        ax3.set_xticklabels([_stage_label(k) for k in stage_order], rotation=12)
        ax3.grid(axis="y", alpha=0.2)
        fig3.savefig(os.path.join(out_dir, "three_agent_multi_implicit_wasserstein.pdf"))
        plt.close(fig3)
    _plot_model_pre_post_accuracy(
        out_dir=out_dir,
        overall=overall,
        model_keys=model_keys,
        model_labels=model_labels,
    )


def write_wasserstein_table(
    out_dir: str,
    implicit_wasserstein: dict,
    multi_implicit_wasserstein: dict,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    table_path = os.path.join(out_dir, "three_agent_wasserstein_table.csv")
    stage_order = [
        "proposal_round_1",
        "aggregation_round_1",
        "proposal_round_2",
        "aggregation_round_2",
    ]
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("question_type,pipeline_component,median_distance,mean_distance,distance_stderr,count\n")
        for qtype, metrics in (
            ("implicit", implicit_wasserstein),
            ("multi_implicit", multi_implicit_wasserstein),
        ):
            if not metrics:
                continue
            for stage in stage_order:
                row = metrics.get(stage)
                if not row:
                    continue
                f.write(
                    f"{qtype},{stage},{row.get('median_distance', 0.0):.6f},"
                    f"{row.get('mean_distance', 0.0):.6f},"
                    f"{row.get('distance_stderr', 0.0):.6f},{int(row.get('count', 0))}\n"
                )
    return table_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot three-agent evaluation metrics")
    parser.add_argument(
        "--preds",
        default="/home/epsteine/post-training/data_filtering/synthetic_filtering_eval/results/synthetic_three_agent_predictions.jsonl",
        help="Prediction JSONL from run_three_agent_eval.py",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/epsteine/post-training/data_filtering/synthetic_filtering_eval/results/plots",
    )
    parser.add_argument(
        "--out-metrics",
        default="/home/epsteine/post-training/data_filtering/synthetic_filtering_eval/results/synthetic_three_agent_metrics.json",
    )
    parser.add_argument(
        "--model-names",
        default="",
        help="Optional comma-separated model names used for proposal rounds per agent.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = _load_jsonl(args.preds)
    model_names = [m.strip() for m in args.model_names.split(",") if m.strip()]
    overall, by_cat, by_type, model_keys, model_labels = compute_metrics(
        rows, model_names or None
    )
    implicit_wasserstein = compute_implicit_wasserstein(rows)
    multi_implicit_wasserstein = compute_multi_implicit_wasserstein(rows)

    metrics_dir = os.path.dirname(args.out_metrics)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall": overall,
                "by_category": by_cat,
                "by_question_type": by_type,
                "implicit_wasserstein": implicit_wasserstein,
                "multi_implicit_wasserstein": multi_implicit_wasserstein,
                "model_keys": model_keys,
                "model_labels": model_labels,
            },
            f,
            indent=2,
        )

    table_path = write_wasserstein_table(
        out_dir=args.out_dir,
        implicit_wasserstein=implicit_wasserstein,
        multi_implicit_wasserstein=multi_implicit_wasserstein,
    )

    write_plots(
        out_dir=args.out_dir,
        overall=overall,
        by_type=by_type,
        implicit_wasserstein=implicit_wasserstein,
        multi_implicit_wasserstein=multi_implicit_wasserstein,
        model_keys=model_keys,
        model_labels=model_labels,
    )
    print(f"Wrote {args.out_metrics}")
    print(f"Wrote plots to {args.out_dir}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
