#!/usr/bin/env python3
"""Summarize grounding impact and plot metrics."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

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

DISPLAY_NAMES = {
    "gpt-5-mini": "GPT-5-mini",
    "gpt-5.2": "GPT-5.2",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3-pro-preview": "Gemini 3 Pro",
    "max(flash, gpt-5-mini)": "Max (Gemini 3 Flash, GPT-5-mini)",
}

def iter_samples(row: dict) -> List[Tuple[str, dict]]:
    sample_keys = [k for k in row.keys() if k.startswith("sample_") and isinstance(row.get(k), dict)]
    if sample_keys:
        return [(key, row[key]) for key in sorted(sample_keys)]
    return [("sample_1", row)]


def display_name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model)

def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_nonconservative_buckets(pred_dir: str, model: str) -> Dict[str, int]:
    path = os.path.join(pred_dir, f"preds_{model}_nonconservative.jsonl")
    if not os.path.exists(path):
        return {}
    rows = load_jsonl(path)
    buckets: Dict[str, int] = {}
    for row in rows:
        gid = row.get("id")
        if not gid:
            continue
        bucket = row.get("bucket", 5)
        try:
            bucket = int(bucket)
        except Exception:
            bucket = 5
        if bucket not in FAILURE_BUCKETS:
            bucket = 5
        buckets[str(gid)] = bucket
    return buckets


def load_gold(path: str, field: str) -> Dict[str, int]:
    gold: Dict[str, int] = {}
    for row in load_jsonl(path):
        if field in row:
            gold[row["id"]] = int(row[field])
    return gold


def to_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def sample_year(sample: dict) -> int | None:
    entities = sample.get("entities", {})
    upper_bounds: List[int] = []
    if isinstance(entities, dict):
        for ent in entities.values():
            if not isinstance(ent, dict):
                continue
            ci = ent.get("updated_confidence_interval") or ent.get("confidence_interval_95")
            if isinstance(ci, list) and len(ci) == 2:
                upper = to_int(ci[1])
                if upper is not None:
                    upper_bounds.append(upper)
    if upper_bounds:
        return max(2001, max(upper_bounds))
    fallback = to_int(sample.get("year", sample.get("pred_year")))
    if fallback is not None:
        return max(2001, fallback)
    return None


def compute_icc(groups: Dict[str, List[int]]) -> float:
    all_values = [v for vals in groups.values() for v in vals]
    n = len(groups)
    if n < 2 or len(all_values) < 2:
        return 0.0
    grand_mean = sum(all_values) / len(all_values)
    k_bar = sum(len(vals) for vals in groups.values()) / n

    ss_between = 0.0
    ss_within = 0.0
    total_n = 0
    for vals in groups.values():
        if not vals:
            continue
        mean_i = sum(vals) / len(vals)
        ss_between += len(vals) * (mean_i - grand_mean) ** 2
        ss_within += sum((v - mean_i) ** 2 for v in vals)
        total_n += len(vals)

    if n <= 1 or total_n <= n:
        return 0.0
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (total_n - n)
    denom = ms_between + (k_bar - 1) * ms_within
    if denom <= 0:
        return 0.0
    return (ms_between - ms_within) / denom


def score(preds: Dict[str, int], gold: Dict[str, int]) -> Tuple[float, float, float]:
    total = 0
    exact = 0
    conservative = 0
    for k, gold_year in gold.items():
        if k not in preds:
            continue
        total += 1
        pred_year = preds[k]
        if pred_year == gold_year:
            exact += 1
        if pred_year >= gold_year:
            conservative += 1
    if total == 0:
        return 0.0, 0.0, 0.0
    exact_acc = exact / total
    conservative_acc = conservative / total
    weighted = 0.5 * (exact_acc + conservative_acc)
    return exact_acc, conservative_acc, weighted


def entity_change_stats(rows: List[dict]) -> Tuple[float, float]:
    total_est = 0
    changed_est = 0
    total_ci = 0
    changed_ci = 0
    for row in rows:
        entities = row.get("entities", {})
        if not isinstance(entities, dict):
            continue
        for ent in entities.values():
            if not isinstance(ent, dict):
                continue
            orig_best = ent.get("best_estimate")
            updated_best = ent.get("updated_best_estimate")
            if isinstance(orig_best, int) and isinstance(updated_best, int):
                total_est += 1
                if orig_best != updated_best:
                    changed_est += 1
            orig_ci = ent.get("confidence_interval_95")
            updated_ci = ent.get("updated_confidence_interval")
            if isinstance(orig_ci, list) and isinstance(updated_ci, list) and len(orig_ci) == 2 and len(updated_ci) == 2:
                total_ci += 1
                if orig_ci != updated_ci:
                    changed_ci += 1
    est_frac = (changed_est / total_est) if total_est else 0.0
    ci_frac = (changed_ci / total_ci) if total_ci else 0.0
    return est_frac, ci_frac


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze search grounding impact.")
    parser.add_argument("--pred-dir", required=True, help="Directory with original preds_*.jsonl")
    parser.add_argument("--updated-dir", required=True, help="Directory with updated preds_*.jsonl")
    parser.add_argument("--updated-suffix", default="_grounded_updated.jsonl", help="Suffix for updated files")
    parser.add_argument(
        "--updated-year-field",
        default="updated_year",
        help="Field name to use for updated years",
    )
    parser.add_argument("--gold-path", required=True, help="Path to gold dataset JSONL")
    parser.add_argument("--gold-field", default="gold_year", help="Gold year field")
    parser.add_argument("--out-json", required=True, help="Path to write summary JSON")
    parser.add_argument(
        "--out-plot",
        required=True,
        help="Output base path for PDF plots (extension optional).",
    )
    args = parser.parse_args()

    gold = load_gold(args.gold_path, args.gold_field)

    summary: List[dict] = []
    models_for_changes: List[str] = []
    est_fracs: List[float] = []
    ci_fracs: List[float] = []
    models_for_accuracy: List[str] = []
    metrics_orig: List[Tuple[float, float, float]] = []
    metrics_updated: List[Tuple[float, float, float]] = []
    preds_original_by_model: Dict[str, Dict[str, int]] = {}
    preds_updated_by_model: Dict[str, Dict[str, int]] = {}
    deltas_by_model: Dict[str, Dict[str, List[int]]] = {}

    for fname in sorted(os.listdir(args.updated_dir)):
        if not fname.startswith("preds_") or not fname.endswith(args.updated_suffix):
            continue
        model = fname.replace("preds_", "").replace(args.updated_suffix, "")
        updated_path = os.path.join(args.updated_dir, fname)
        original_path = os.path.join(args.pred_dir, f"preds_{model}.jsonl")
        if not os.path.exists(original_path):
            continue

        updated_rows = load_jsonl(updated_path)
        original_rows = load_jsonl(original_path)

        preds_original = {}
        for row in original_rows:
            year = to_int(row.get("year", row.get("pred_year")))
            if year is not None:
                preds_original[row["id"]] = year
        preds_updated = {}
        for row in updated_rows:
            year = to_int(row.get(args.updated_year_field))
            if year is None:
                year = to_int(row.get("year", row.get("pred_year")))
            if year is not None:
                preds_updated[row["id"]] = year

        exact_o, cons_o, weighted_o = score(preds_original, gold)
        exact_u, cons_u, weighted_u = score(preds_updated, gold)
        est_frac, ci_frac = entity_change_stats(updated_rows)

        sample_groups: Dict[str, List[int]] = {}
        all_same_count = 0
        all_same_total = 0
        for row in updated_rows:
            row_id = str(row.get("id"))
            values: List[int] = []
            for _, sample in iter_samples(row):
                year = sample_year(sample)
                if year is not None:
                    values.append(year)
            if values:
                sample_groups[row_id] = values
                all_same_total += 1
                if len(set(values)) == 1:
                    all_same_count += 1
        sample_icc = compute_icc(sample_groups) if sample_groups else 0.0
        all_same_frac = (all_same_count / all_same_total) if all_same_total else 0.0

        summary.append(
            {
                "model": model,
                "estimate_changed_frac": est_frac,
                "ci_changed_frac": ci_frac,
                "exact_original": exact_o,
                "conservative_original": cons_o,
                "weighted_original": weighted_o,
                "exact_updated": exact_u,
                "conservative_updated": cons_u,
                "weighted_updated": weighted_u,
                "sample_consistency_icc": sample_icc,
                "sample_all_same_frac": all_same_frac,
            }
        )
        models_for_changes.append(model)
        est_fracs.append(est_frac)
        ci_fracs.append(ci_frac)
        models_for_accuracy.append(model)
        metrics_orig.append((exact_o, cons_o, weighted_o))
        metrics_updated.append((exact_u, cons_u, weighted_u))
        preds_original_by_model[model] = preds_original
        preds_updated_by_model[model] = preds_updated

        orig_deltas: List[int] = []
        upd_deltas: List[int] = []
        for gid, gold_year in gold.items():
            if gid in preds_original:
                orig_deltas.append(preds_original[gid] - gold_year)
            if gid in preds_updated:
                upd_deltas.append(preds_updated[gid] - gold_year)
        deltas_by_model[model] = {"orig": orig_deltas, "updated": upd_deltas}

    combo_models = ("gemini-3-flash-preview", "gpt-5-mini")
    if all(m in preds_original_by_model for m in combo_models) and all(
        m in preds_updated_by_model for m in combo_models
    ):
        combo_name = "max(flash, gpt-5-mini)"
        preds_a = preds_original_by_model[combo_models[0]]
        preds_b = preds_original_by_model[combo_models[1]]
        preds_combined = {k: max(preds_a.get(k, 0), preds_b.get(k, 0)) for k in gold.keys()}
        preds_a_u = preds_updated_by_model[combo_models[0]]
        preds_b_u = preds_updated_by_model[combo_models[1]]
        preds_combined_u = {k: max(preds_a_u.get(k, 0), preds_b_u.get(k, 0)) for k in gold.keys()}

        exact_o, cons_o, weighted_o = score(preds_combined, gold)
        exact_u, cons_u, weighted_u = score(preds_combined_u, gold)
        summary.append(
            {
                "model": combo_name,
                "estimate_changed_frac": None,
                "ci_changed_frac": None,
                "exact_original": exact_o,
                "conservative_original": cons_o,
                "weighted_original": weighted_o,
                "exact_updated": exact_u,
                "conservative_updated": cons_u,
                "weighted_updated": weighted_u,
            }
        )
        models_for_accuracy.insert(0, combo_name)
        metrics_orig.insert(0, (exact_o, cons_o, weighted_o))
        metrics_updated.insert(0, (exact_u, cons_u, weighted_u))

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not models_for_accuracy:
        return

    base = args.out_plot
    if base.lower().endswith(".pdf"):
        base = base[: -len(".pdf")]

    # Plot 1: estimate changed fraction
    fig1, ax1 = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax1.bar([display_name(m) for m in models_for_changes], est_fracs)
    ax1.set_title("Fraction of entity estimates changed (by model)")
    ax1.set_ylabel("Fraction")
    ax1.set_ylim(0, 1)
    fig1.savefig(f"{base}_estimate_changed.pdf")

    # Plot 2: CI changed fraction
    fig2, ax2 = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax2.bar([display_name(m) for m in models_for_changes], ci_fracs)
    ax2.set_title("Fraction of confidence intervals changed (by model)")
    ax2.set_ylabel("Fraction")
    ax2.set_ylim(0, 1)
    fig2.savefig(f"{base}_ci_changed.pdf")

    # Plot 3: accuracy metrics (1x3 subplots with shared legend)
    metric_labels = ["Exact", "Conservative", "Weighted"]
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True, sharey=True)
    width = 0.35
    x = list(range(len(models_for_accuracy)))
    legend_handles = []
    legend_labels = []
    for metric_idx, metric_name in enumerate(metric_labels):
        ax = axes3[metric_idx]
        for model_idx, model in enumerate(models_for_accuracy):
            orig_val = metrics_orig[model_idx][metric_idx]
            upd_val = metrics_updated[model_idx][metric_idx]
            label_base = display_name(model)
            h1 = ax.bar(model_idx - width / 2, orig_val, width, label=f"{label_base} (no search)")
            h2 = ax.bar(model_idx + width / 2, upd_val, width, label=f"{label_base} (search)")
            if metric_idx == 0:
                legend_handles.extend([h1[0], h2[0]])
                legend_labels.extend([f"{label_base} (no search)", f"{label_base} (search)"])
        ax.set_title(f"{metric_name} accuracy")
        ax.set_xticks(x, [display_name(m) for m in models_for_accuracy], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        if metric_idx == 0:
            ax.set_ylabel("Accuracy")

    fig3.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=8,
    )
    fig3.savefig(f"{base}_accuracy.pdf", bbox_inches="tight")

    # Plot 4: no-leak accuracy only
    fig4, ax4 = plt.subplots(figsize=(10, 4), constrained_layout=True)
    width = 0.35
    x = list(range(len(models_for_accuracy)))
    orig_vals = [vals[1] for vals in metrics_orig]
    upd_vals = [vals[1] for vals in metrics_updated]
    color_no_search = "#4C78A8"
    color_search = "#F58518"
    for i in range(len(models_for_accuracy)):
        ax4.bar(i - width / 2, orig_vals[i], width, color=color_no_search)
        ax4.bar(i + width / 2, upd_vals[i], width, color=color_search)
    ax4.set_title("")
    ax4.set_xticks(x, [display_name(m) for m in models_for_accuracy], rotation=20, ha="right")
    ax4.set_ylabel("No-leak accuracy")
    ax4.set_ylim(0.3, 1.05)
    legend_handles = [
        Patch(facecolor=color_no_search, label="No search"),
        Patch(facecolor=color_search, label="Search"),
    ]
    ax4.legend(
        handles=legend_handles,
        ncol=2,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.35),
    )
    fig4.savefig(f"{base}_conservative.pdf")

    # Plot 5: per-model delta (pred - gold), original vs updated
    for model in models_for_changes:
        deltas = deltas_by_model.get(model)
        if not deltas:
            continue
        orig = deltas.get("orig", [])
        upd = deltas.get("updated", [])
        if not orig and not upd:
            continue
        all_deltas = orig + upd
        min_d = min(all_deltas)
        max_d = max(all_deltas)
        bins = [x - 0.5 for x in range(min_d, max_d + 2)]

        fig5, axes5 = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
        axes5[0].hist(orig, bins=bins, color="#4C78A8", alpha=0.8)
        axes5[0].axvline(0, color="black", linewidth=1)
        axes5[0].set_title(f"{display_name(model)} (no search)")
        axes5[0].set_xlabel("Predicted year - Ground Truth year")
        axes5[0].set_ylabel("Count")

        axes5[1].hist(upd, bins=bins, color="#4C78A8", alpha=0.4)
        axes5[1].axvline(0, color="black", linewidth=1)
        axes5[1].set_title(f"{display_name(model)} (search)")
        axes5[1].set_xlabel("Predicted year - Ground Truth year")

        fig5.savefig(f"{base}_delta_{model}.pdf")

    # Plot 5b: search-only delta grid with failure-colored negative bars
    search_only_models = [
        m
        for m in models_for_accuracy
        if m in {"gpt-5-mini", "gpt-5.2", "gemini-3-flash-preview", "gemini-3-pro-preview"}
    ]
    if search_only_models:
        fig5b, axes5b = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=False, sharey=True)
        axes_flat = [ax for row in axes5b for ax in row]
        legend_patches = [
            Patch(facecolor=FAILURE_COLORS[k], label=FAILURE_BUCKETS[k]) for k in sorted(FAILURE_BUCKETS)
        ]
        for ax, model in zip(axes_flat, search_only_models):
            preds_updated = preds_updated_by_model.get(model, {})
            if not preds_updated:
                ax.axis("off")
                continue
            buckets_by_id = load_nonconservative_buckets(args.pred_dir, model)
            deltas_by_id: Dict[str, int] = {}
            for gid, gold_year in gold.items():
                if gid in preds_updated:
                    deltas_by_id[gid] = preds_updated[gid] - gold_year
            if not deltas_by_id:
                ax.axis("off")
                continue

            delta_vals = list(deltas_by_id.values())
            min_d = min(delta_vals)
            max_d = max(delta_vals)
            bin_values = list(range(min_d, max_d + 1))

            neg_counts = {k: {b: 0 for b in bin_values} for k in FAILURE_BUCKETS}
            pos_counts = {b: 0 for b in bin_values}
            for gid, delta in deltas_by_id.items():
                if delta < 0:
                    bucket = buckets_by_id.get(str(gid), 5)
                    if bucket not in neg_counts:
                        bucket = 5
                    neg_counts[bucket][delta] += 1
                else:
                    pos_counts[delta] += 1

            for bin_value in bin_values:
                if bin_value < 0:
                    bottom = 0
                    for bucket in sorted(FAILURE_BUCKETS):
                        count = neg_counts[bucket][bin_value]
                        if count:
                            ax.bar(
                                bin_value,
                                count,
                                bottom=bottom,
                                color=FAILURE_COLORS[bucket],
                                width=0.9,
                            )
                        bottom += count
                else:
                    count = pos_counts[bin_value]
                    if count:
                        ax.bar(bin_value, count, color="#1f77b4", width=0.9)

            ax.axvline(0, color="black", linewidth=1)
            ax.set_title(display_name(model), pad=6, fontsize=11)
            ax.set_xlabel("Predicted year - Ground Truth year", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis="both", labelsize=9)

        for ax in axes_flat[len(search_only_models) :]:
            ax.axis("off")
        fig5b.subplots_adjust(bottom=0.16, hspace=0.35)
        fig5b.legend(
            handles=legend_patches,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=3,
            fontsize=9,
            title="Failure type",
            title_fontsize=9,
        )
        fig5b.savefig(f"{base}_delta_search_grid.pdf")

    # Plot 6: failure case buckets per model (2x2 grid)
    failure_files = []
    for fname in sorted(os.listdir(args.pred_dir)):
        if fname.startswith("preds_") and fname.endswith("_nonconservative.jsonl"):
            failure_files.append(fname)

    if failure_files:
        fig6, axes6 = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes_flat = [ax for row in axes6 for ax in row]
        labels = [FAILURE_BUCKETS[k] for k in sorted(FAILURE_BUCKETS)]
        max_count = 1
        counts_by_model: List[List[int]] = []
        models_for_failures: List[str] = []
        for fname in failure_files[:4]:
            model = fname.replace("preds_", "").replace("_nonconservative.jsonl", "")
            rows = load_jsonl(os.path.join(args.pred_dir, fname))
            counts = {k: 0 for k in FAILURE_BUCKETS}
            for row in rows:
                bucket = row.get("bucket")
                try:
                    bucket = int(bucket)
                except Exception:
                    bucket = 5
                if bucket not in counts:
                    bucket = 5
                counts[bucket] += 1
            values = [counts[k] for k in sorted(FAILURE_BUCKETS)]
            counts_by_model.append(values)
            models_for_failures.append(model)
            max_count = max(max_count, max(values))

        for ax, model, values in zip(axes_flat, models_for_failures, counts_by_model):
            colors = [FAILURE_COLORS[k] for k in sorted(FAILURE_BUCKETS)]
            ax.bar(labels, values, color=colors)
            ax.set_title(display_name(model), fontsize=11)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_ylim(0, max_count + 1)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis="both", labelsize=9)

        for ax in axes_flat[len(failure_files[:4]) :]:
            ax.axis("off")

        legend_patches = [
            Patch(facecolor=FAILURE_COLORS[k], label=FAILURE_BUCKETS[k]) for k in sorted(FAILURE_BUCKETS)
        ]
        fig6.legend(
            handles=legend_patches,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=3,
            fontsize=9,
            title="Failure type",
            title_fontsize=9,
        )
        fig6.savefig(f"{base}_failures_grid.pdf")

    # Plot 7: search-only accuracy (2x2 grid)
    if search_only_models:
        fig7, axes7 = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharey=True)
        axes_flat = [ax for row in axes7 for ax in row]
        for ax, model in zip(axes_flat, search_only_models):
            idx = models_for_accuracy.index(model)
            vals = metrics_updated[idx]
            ax.bar(["Exact", "Conservative", "Weighted"], vals, color="#4C78A8")
            ax.set_title(display_name(model))
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
        for ax in axes_flat[len(search_only_models) :]:
            ax.axis("off")
        fig7.savefig(f"{base}_accuracy_search_only.pdf")


if __name__ == "__main__":
    main()
