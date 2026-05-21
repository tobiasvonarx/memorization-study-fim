#!/usr/bin/env python3
"""Collate attention-probe summaries across repetitions and arms."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))

from collation_utils import (
    apply_conference_style,
    ci95_from_std,
    finite_ylim,
    metric_ci95_from_row,
    set_repetition_axis,
    interval_band,
)
from direct_overlap_eval import default_results_root
from verbatim_suite import arm_root, load_manifest_for_suite, suite_arms_for_report, suite_root


DEFAULT_REPETITIONS = [1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128]
DEFAULT_NATIVE_SPLITS = "0:100,20:80,40:60,60:40,80:20,100:0"
METRICS = [
    "cooper_extractable",
    "cooper_token_geomean_p_z",
    "cooper_supported_token_rate",
    "attn_prefix_mass",
    "attn_suffix_mass",
    "attn_target_prev_mass",
    "attn_fim_marker_mass",
    "attn_prompt_mass",
    "attn_prefix_share_of_context",
    "attn_suffix_share_of_context",
    "attn_first_prefix_mass",
    "attn_first_suffix_mass",
    "attn_first_fim_marker_mass",
    "attn_later_prefix_mass",
    "attn_later_suffix_mass",
    "attn_later_target_prev_mass",
    "attn_later_fim_marker_mass",
    "attn_entropy_norm",
    "attn_max_weight",
    "attn_backward_distance",
    "attn_backward_distance_norm",
    "attn_early_prefix_mass",
    "attn_mid_prefix_mass",
    "attn_late_prefix_mass",
    "attn_early_suffix_mass",
    "attn_mid_suffix_mass",
    "attn_late_suffix_mass",
    "attn_early_target_prev_mass",
    "attn_mid_target_prev_mass",
    "attn_late_target_prev_mass",
]
ATTENTION_PARTITION_METRICS = [
    "attn_prefix_mass",
    "attn_suffix_mass",
    "attn_fim_marker_mass",
    "attn_target_prev_mass",
]
ATTENTION_PARTITION_TOLERANCE = 1e-3


@dataclass(frozen=True)
class ArmSpec:
    arm_label: str
    display_label: str
    prompt_format: str
    model_label: str
    prefix_length: int
    suffix_length: int
    suite_arm_id: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare attention-probe summaries")
    parser.add_argument("--results-root", type=Path, default=default_results_root())
    parser.add_argument("--suite", default=None, help="Read attention summaries from a unified suite store")
    parser.add_argument(
        "--suite-report",
        choices=["attention_ltr", "attention_native_geometry"],
        default="attention_ltr",
    )
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--repetitions", type=int, nargs="+", default=DEFAULT_REPETITIONS)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument(
        "--window-layout",
        choices=["matched_context", "cooper_nonoverlap", "cooper_sliding"],
        default=None,
    )
    parser.add_argument("--max-windows-per-excerpt", type=int, default=4)
    parser.add_argument("--context-budget", type=int, default=None)
    parser.add_argument("--middle-length", type=int, default=None)
    parser.add_argument("--native-splits", default="none")
    parser.add_argument("--no-ltr-arms", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def apply_suite_defaults(args: argparse.Namespace) -> None:
    if not args.suite:
        missing = [
            name
            for name in ["study_name", "window_stride", "window_layout", "context_budget", "middle_length"]
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError("Missing required legacy arguments: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
        return

    args.study_name = args.study_name or args.suite
    args.context_budget = args.context_budget or 100
    args.middle_length = args.middle_length or 20
    args.window_stride = args.window_stride or 120
    if args.suite_report == "attention_ltr":
        args.window_layout = args.window_layout or "cooper_nonoverlap"
        if args.native_splits == "none":
            args.native_splits = "none"
        args.no_ltr_arms = False
    elif args.suite_report == "attention_native_geometry":
        args.window_layout = args.window_layout or "matched_context"
        if args.native_splits == "none":
            args.native_splits = DEFAULT_NATIVE_SPLITS
        args.no_ltr_arms = True


def parse_native_splits(value: str, context_budget: int) -> list[tuple[int, int]]:
    if value.strip().lower() in {"", "none", "null", "off"}:
        return []
    splits: list[tuple[int, int]] = []
    for item in value.replace(",", " ").split():
        prefix_raw, suffix_raw = item.split(":", 1)
        prefix = int(prefix_raw)
        suffix = int(suffix_raw)
        if prefix + suffix != context_budget:
            raise ValueError(f"Split {item} does not match context budget {context_budget}")
        splits.append((prefix, suffix))
    return splits


def build_arms(args: argparse.Namespace) -> list[ArmSpec]:
    if args.suite:
        manifest = load_manifest_for_suite(args.results_root, args.suite)
        arms = []
        for arm in suite_arms_for_report(manifest, args.suite_report):
            if args.suite_report == "attention_ltr":
                arm_label = f"{arm['model_label']}_ltr"
            else:
                arm_label = f"fim_v2_native_p{arm['prefix_length']}_s{arm['suffix_length']}"
            arms.append(
                ArmSpec(
                    arm_label=arm_label,
                    display_label=arm["display_label"],
                    prompt_format=arm["prompt_format"],
                    model_label=arm["model_label"],
                    prefix_length=int(arm["prefix_length"]),
                    suffix_length=int(arm["suffix_length"]),
                    suite_arm_id=arm["arm_id"],
                )
            )
        if not arms:
            raise ValueError(f"No suite arms found for report {args.suite_report}")
        return arms

    arms: list[ArmSpec] = []
    if not args.no_ltr_arms:
        arms.extend(
            [
                ArmSpec("no_fim_ltr", f"no-FIM LTR ({args.context_budget}L)", "ltr_prefix", "no_fim", args.context_budget, 0),
                ArmSpec("fim_v2_ltr", f"FIM-v2 LTR ({args.context_budget}L)", "ltr_prefix", "fim_v2", args.context_budget, 0),
            ]
        )
    for prefix, suffix in parse_native_splits(args.native_splits, args.context_budget):
        arms.append(
            ArmSpec(
                f"fim_v2_native_p{prefix}_s{suffix}",
                f"FIM-v2 native ({prefix}L/{suffix}R)",
                "fim_native",
                "fim_v2",
                prefix,
                suffix,
            )
        )
    if not arms:
        raise ValueError("No arms requested")
    return arms


def output_stem(
    offset: int,
    window_stride: int,
    window_layout: str,
    max_windows_per_excerpt: int,
    prefix_length: int,
    middle_length: int,
    suffix_length: int,
) -> str:
    window_limit = "all" if max_windows_per_excerpt == 0 else str(max_windows_per_excerpt)
    layout_prefix = "" if window_layout == "matched_context" else f"layout_{window_layout}_"
    return (
        f"{layout_prefix}"
        f"target_offset_{offset}_stride_{window_stride}_"
        f"windows_{window_limit}_prefix_{prefix_length}_"
        f"middle_{middle_length}_suffix_{suffix_length}"
    )


def comparison_stem(args: argparse.Namespace) -> str:
    if args.suite:
        return f"{args.suite}_{args.suite_report}"
    window_limit = "all" if args.max_windows_per_excerpt == 0 else str(args.max_windows_per_excerpt)
    layout = "" if args.window_layout == "matched_context" else f"_layout_{args.window_layout}"
    return (
        f"{args.study_name}_attention_target_offset_{args.offset}_stride_{args.window_stride}_"
        f"windows_{window_limit}_middle_{args.middle_length}_context_{args.context_budget}{layout}"
    )


def summary_path(args: argparse.Namespace, arm: ArmSpec, repetition: int) -> Path:
    if args.suite:
        if arm.suite_arm_id is None:
            raise ValueError(f"Suite arm id missing for {arm.arm_label}")
        return arm_root(args.results_root, args.suite, arm.suite_arm_id, repetition) / "attention.summary.json"
    stem = output_stem(
        args.offset,
        args.window_stride,
        args.window_layout,
        args.max_windows_per_excerpt,
        arm.prefix_length,
        args.middle_length,
        arm.suffix_length,
    )
    return (
        args.results_root
        / "attention"
        / args.study_name
        / arm.prompt_format
        / arm.model_label
        / f"rep_{repetition}"
        / f"{stem}.attention.summary.json"
    )


def metric_mean(summary: dict[str, Any], metric: str) -> float:
    return float(summary["metrics"].get(metric, {}).get("mean", float("nan")))


def metric_std(summary: dict[str, Any], metric: str) -> float:
    return float(summary["metrics"].get(metric, {}).get("std", float("nan")))


def finite_sum(values: list[float]) -> float:
    if not all(math.isfinite(value) for value in values):
        return float("nan")
    return float(sum(values))


def add_attention_partition_fields(row: dict[str, Any]) -> None:
    mass = finite_sum([float(row.get(metric, float("nan"))) for metric in ATTENTION_PARTITION_METRICS])
    row["attn_partition_mass"] = mass
    row["attn_partition_residual"] = 1.0 - mass if math.isfinite(mass) else float("nan")
    row["attn_partition_abs_residual"] = abs(row["attn_partition_residual"]) if math.isfinite(mass) else float("nan")


def load_summaries(args: argparse.Namespace, arms: list[ArmSpec]) -> dict[str, dict[int, dict[str, Any]]]:
    summaries: dict[str, dict[int, dict[str, Any]]] = {arm.arm_label: {} for arm in arms}
    missing: list[Path] = []
    for arm in arms:
        for repetition in args.repetitions:
            path = summary_path(args, arm, repetition)
            if not path.exists():
                missing.append(path)
                continue
            with path.open("r", encoding="utf-8") as handle:
                summaries[arm.arm_label][repetition] = json.load(handle)
    if missing:
        raise FileNotFoundError("Missing attention summaries:\n" + "\n".join(str(path) for path in missing))
    return summaries


def long_rows(
    args: argparse.Namespace,
    arms: list[ArmSpec],
    summaries: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        for repetition in args.repetitions:
            summary = summaries[arm.arm_label][repetition]
            row: dict[str, Any] = {
                "arm_label": arm.arm_label,
                "display_label": arm.display_label,
                "prompt_format": arm.prompt_format,
                "model_label": arm.model_label,
                "repetition": repetition,
                "num_windows": summary["num_windows"],
                "num_excerpts": summary["num_excerpts"],
                "prefix_length": summary["prefix_length"],
                "middle_length": summary["middle_length"],
                "suffix_length": summary["suffix_length"],
                "window_stride": summary["window_stride"],
                "window_layout": summary["window_layout"],
                "max_windows_per_excerpt": summary["max_windows_per_excerpt"],
            }
            for metric in METRICS:
                row[metric] = metric_mean(summary, metric)
                row[f"{metric}_std"] = metric_std(summary, metric)
                row[f"{metric}_ci95"] = metric_ci95_from_row(row, metric)
            add_attention_partition_fields(row)
            rows.append(row)
    return rows


def weighted_overall(
    args: argparse.Namespace,
    arms: list[ArmSpec],
    summaries: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    overall: dict[str, dict[str, float]] = {}
    for arm in arms:
        total_windows = sum(summaries[arm.arm_label][rep]["num_windows"] for rep in args.repetitions)
        arm_metrics: dict[str, float] = {"num_windows": float(total_windows)}
        for metric in METRICS:
            weighted_sum = sum(
                metric_mean(summaries[arm.arm_label][rep], metric)
                * summaries[arm.arm_label][rep]["num_windows"]
                for rep in args.repetitions
            )
            mean = weighted_sum / total_windows if total_windows else float("nan")
            arm_metrics[metric] = mean
            if total_windows > 1 and math.isfinite(mean):
                m2 = 0.0
                for rep in args.repetitions:
                    summary = summaries[arm.arm_label][rep]
                    n = int(summary["num_windows"])
                    rep_mean = metric_mean(summary, metric)
                    rep_std = metric_std(summary, metric)
                    if n <= 0 or not math.isfinite(rep_mean):
                        continue
                    if math.isfinite(rep_std) and n > 1:
                        m2 += (n - 1) * rep_std * rep_std
                    m2 += n * (rep_mean - mean) ** 2
                combined_std = math.sqrt(m2 / (total_windows - 1)) if total_windows > 1 else float("nan")
                arm_metrics[f"{metric}_std"] = combined_std
                arm_metrics[f"{metric}_ci95"] = ci95_from_std(combined_std, total_windows)
            else:
                arm_metrics[f"{metric}_std"] = float("nan")
                arm_metrics[f"{metric}_ci95"] = 0.0
        partition_mass = finite_sum([arm_metrics[metric] for metric in ATTENTION_PARTITION_METRICS])
        arm_metrics["attn_partition_mass"] = partition_mass
        arm_metrics["attn_partition_residual"] = 1.0 - partition_mass if math.isfinite(partition_mass) else float("nan")
        arm_metrics["attn_partition_abs_residual"] = (
            abs(arm_metrics["attn_partition_residual"]) if math.isfinite(partition_mass) else float("nan")
        )
        overall[arm.arm_label] = arm_metrics
    return overall


def attention_partition_rows(
    rows: list[dict[str, Any]],
    overall: dict[str, dict[str, float]],
    arms: list[ArmSpec],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for arm in arms:
        arm_rows = [row for row in rows if row["arm_label"] == arm.arm_label]
        max_rep_abs_residual = max(
            (
                float(row["attn_partition_abs_residual"])
                for row in arm_rows
                if math.isfinite(float(row["attn_partition_abs_residual"]))
            ),
            default=float("nan"),
        )
        arm_overall = overall[arm.arm_label]
        out.append(
            {
                "arm_label": arm.arm_label,
                "display_label": arm.display_label,
                "prompt_format": arm.prompt_format,
                "prefix_length": arm.prefix_length,
                "suffix_length": arm.suffix_length,
                "overall_prefix_mass": arm_overall["attn_prefix_mass"],
                "overall_suffix_mass": arm_overall["attn_suffix_mass"],
                "overall_fim_sentinel_mass": arm_overall["attn_fim_marker_mass"],
                "overall_previous_target_mass": arm_overall["attn_target_prev_mass"],
                "overall_partition_mass": arm_overall["attn_partition_mass"],
                "overall_residual_to_one": arm_overall["attn_partition_residual"],
                "max_repetition_abs_residual": max_rep_abs_residual,
                "closes_to_one": (
                    math.isfinite(max_rep_abs_residual)
                    and max_rep_abs_residual <= ATTENTION_PARTITION_TOLERANCE
                    and abs(arm_overall["attn_partition_residual"]) <= ATTENTION_PARTITION_TOLERANCE
                ),
            }
        )
    return out


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["arm_label"], row["repetition"]): row for row in rows}
    out: list[dict[str, Any]] = []
    repetitions = sorted({int(row["repetition"]) for row in rows})
    for repetition in repetitions:
        left = by_key.get(("fim_v2_ltr", repetition))
        right = by_key.get(("no_fim_ltr", repetition))
        if left is None or right is None:
            continue
        row: dict[str, Any] = {
            "pair_label": "fim_v2_ltr_minus_no_fim_ltr",
            "repetition": repetition,
            "num_windows": right["num_windows"],
        }
        for metric in METRICS:
            row[f"fim_v2_ltr_{metric}"] = left[metric]
            row[f"no_fim_ltr_{metric}"] = right[metric]
            row[f"delta_{metric}"] = left[metric] - right[metric]
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_plotting() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/codex-cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_conference_style(plt)
    return plt


def save_figure(plt: Any, figure_dir: Path, stem: str) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / f"{stem}.png"
    pdf = figure_dir / f"{stem}.pdf"
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    return {"png": str(png), "pdf": str(pdf)}


def plot_repetition_curves(args: argparse.Namespace, rows: list[dict[str, Any]], arms: list[ArmSpec], figure_dir: Path) -> dict[str, Any]:
    if not any(arm.prompt_format == "ltr_prefix" for arm in arms):
        return {}
    plt = ensure_plotting()
    colors = {"no_fim_ltr": "#356AA0", "fim_v2_ltr": "#7A5195"}
    panels = [
        ("attn_prefix_mass", "Attention to prefix", "Mass"),
        ("attn_target_prev_mass", "Attention to previous target", "Mass"),
        ("attn_entropy_norm", "Normalized attention entropy", "Entropy"),
        ("cooper_extractable", "Top-k extractable", "Rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.4), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes.ravel(), panels):
        all_values: list[float] = []
        all_errors: list[float] = []
        for arm in arms:
            if arm.prompt_format != "ltr_prefix":
                continue
            series_rows = [
                next(
                    row
                    for row in rows
                    if row["arm_label"] == arm.arm_label and int(row["repetition"]) == rep
                )
                for rep in args.repetitions
            ]
            y_values = [
                float(row[metric])
                for row in series_rows
            ]
            yerr = [metric_ci95_from_row(row, metric) for row in series_rows]
            lower, upper = interval_band(
                y_values,
                yerr,
                lower_floor=0.0,
                upper_ceiling=1.0
                if metric.startswith("attn_") or metric == "cooper_extractable"
                else None,
            )
            all_values.extend(y_values)
            all_errors.extend(yerr)
            color = colors.get(arm.arm_label, "#555555")
            ax.fill_between(
                args.repetitions,
                lower,
                upper,
                color=color,
                alpha=0.16,
                linewidth=0.0,
                zorder=1,
            )
            ax.plot(
                args.repetitions,
                y_values,
                linewidth=1.9,
                color=color,
                label=arm.display_label,
                solid_capstyle="round",
                zorder=3,
            )
        set_repetition_axis(ax, list(args.repetitions), rotate=0)
        ylim = finite_ylim(all_values, all_errors, lower_floor=0.0, pad=0.16)
        if ylim is not None:
            upper = (
                min(1.0, ylim[1])
                if metric.startswith("attn_") or metric == "cooper_extractable"
                else ylim[1]
            )
            ax.set_ylim(ylim[0], upper)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Training repetitions")
        ax.grid(True, axis="y", alpha=0.16)
    axes[0, 0].legend(frameon=False)
    return {"attention_repetition_curves": save_figure(plt, figure_dir, "attention_repetition_curves")}


def plot_ltr_attention_partition_stacked(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    arms: list[ArmSpec],
    figure_dir: Path,
) -> dict[str, Any]:
    ltr_arms = [arm for arm in arms if arm.prompt_format == "ltr_prefix"]
    if not ltr_arms:
        return {}

    model_rank = {"no_fim": 0, "fim_v2": 1}
    ltr_arms = sorted(ltr_arms, key=lambda arm: (model_rank.get(arm.model_label, 99), arm.display_label))
    if len(ltr_arms) > 2:
        ltr_arms = ltr_arms[:2]

    by_key = {(row["arm_label"], int(row["repetition"])): row for row in rows}
    repetitions = [rep for rep in args.repetitions if all((arm.arm_label, rep) in by_key for arm in ltr_arms)]
    if not repetitions:
        return {}

    plt = ensure_plotting()
    fig, ax = plt.subplots(figsize=(9.0, 3.7), constrained_layout=True)
    components = [
        ("attn_prefix_mass", "Prefix", "#4C78A8"),
        ("attn_target_prev_mass", "Previous target", "#F28E2B"),
    ]
    bar_width = 0.32 if len(ltr_arms) > 1 else 0.44
    offsets = [-bar_width * 0.55, bar_width * 0.55] if len(ltr_arms) > 1 else [0.0]
    x_values = list(range(len(repetitions)))

    for arm, offset in zip(ltr_arms, offsets):
        positions = [x + offset for x in x_values]
        bottoms = [0.0 for _ in repetitions]
        for metric, label, color in components:
            values = [float(by_key[(arm.arm_label, rep)][metric]) for rep in repetitions]
            ax.bar(
                positions,
                values,
                width=bar_width,
                bottom=bottoms,
                color=color,
                edgecolor="white",
                linewidth=0.65,
                label=label if arm == ltr_arms[0] else None,
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(rep) for rep in repetitions])
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xlabel("Training repetitions")
    ax.set_ylabel("Mean attention mass")
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.65)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, handlelength=1.4)

    artifacts = save_figure(plt, figure_dir, "attention_ltr_partition_stacked")
    caption_lines = [
        "Mean attention allocation in the LTR probe; bars are stacked into prefix and previous-target mass.",
        "Within each repetition, bars are ordered "
        + " then ".join(arm.display_label for arm in ltr_arms)
        + ". Values are prefix/previous-target mass:",
    ]
    for arm in ltr_arms:
        values = []
        for rep in repetitions:
            row = by_key[(arm.arm_label, rep)]
            values.append(
                f"{rep}: {float(row['attn_prefix_mass']):.3f}/{float(row['attn_target_prev_mass']):.3f}"
            )
        caption_lines.append(f"{arm.display_label}: " + ", ".join(values) + ".")
    caption = "\n".join(caption_lines)
    caption_path = figure_dir / "attention_ltr_partition_stacked.caption.txt"
    caption_path.write_text(caption + "\n", encoding="utf-8")
    artifacts["caption"] = str(caption_path)
    return {"attention_ltr_partition_stacked": artifacts}


def plot_native_sweep(overall: dict[str, dict[str, float]], arms: list[ArmSpec], figure_dir: Path) -> dict[str, Any]:
    native = sorted([arm for arm in arms if arm.prompt_format == "fim_native"], key=lambda arm: arm.prefix_length)
    if not native:
        return {}
    plt = ensure_plotting()
    panels = [
        ("attn_prefix_mass", "Attention to prefix", "Mass"),
        ("attn_suffix_mass", "Attention to suffix", "Mass"),
        ("attn_suffix_share_of_context", "Suffix share of context attention", "Share"),
        ("attn_target_prev_mass", "Attention to previous target", "Mass"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.3), constrained_layout=True)
    x_values = [arm.prefix_length for arm in native]
    x_labels = [f"{arm.prefix_length}/{arm.suffix_length}" for arm in native]
    for ax, (metric, title, ylabel) in zip(axes.ravel(), panels):
        y_values = [overall[arm.arm_label][metric] for arm in native]
        yerr = [overall[arm.arm_label].get(f"{metric}_ci95", 0.0) for arm in native]
        lower, upper = interval_band(
            y_values,
            yerr,
            lower_floor=0.0,
            upper_ceiling=1.0,
        )
        ax.fill_between(
            x_values,
            lower,
            upper,
            color="#2A9D8F",
            alpha=0.16,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            x_values,
            y_values,
            linewidth=1.9,
            color="#2A9D8F",
            solid_capstyle="round",
            zorder=3,
        )
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels, rotation=35)
        ylim = finite_ylim(y_values, yerr, lower_floor=0.0, pad=0.16)
        if ylim is not None:
            ax.set_ylim(ylim[0], min(1.0, ylim[1]))
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Prefix tokens")
        ax.grid(True, axis="y", alpha=0.16)
    return {"attention_native_sweep": save_figure(plt, figure_dir, "attention_native_sweep")}


def set_split_xticks(
    ax: Any,
    positions: list[int],
    prefix_lengths: list[int],
    suffix_lengths: list[int],
    *,
    labelsize: float = 10.2,
    omit_prefixes: set[int] | None = None,
) -> None:
    omitted = omit_prefixes or set()
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            "" if prefix in omitted else f"{prefix}/{suffix}"
            for prefix, suffix in zip(prefix_lengths, suffix_lengths, strict=True)
        ],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
        color="black",
        fontsize=labelsize,
    )
    ax.tick_params(axis="x", colors="black")


def plot_native_attention_stack(overall: dict[str, dict[str, float]], arms: list[ArmSpec], figure_dir: Path) -> dict[str, Any]:
    native = sorted([arm for arm in arms if arm.prompt_format == "fim_native"], key=lambda arm: arm.prefix_length)
    if not native:
        return {}
    plt = ensure_plotting()
    x_values = [arm.prefix_length for arm in native]
    suffix_values = [arm.suffix_length for arm in native]
    prefix_text = "#8A3A0A"
    suffix_text = "#7A1F55"
    components = [
        ("attn_prefix_mass", "Prefix", prefix_text, "////"),
        ("attn_suffix_mass", "Suffix", suffix_text, "\\\\\\\\"),
        ("attn_fim_marker_mass", "FIM sentinels", "#4A4A4A", "----"),
        ("attn_target_prev_mass", "Previous target", "#C49A00", "||||"),
    ]
    y_values = [[overall[arm.arm_label][metric] for arm in native] for metric, _, _, _ in components]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.22, top=0.78)
    plt.rcParams["hatch.linewidth"] = 0.75
    stacked = ax.stackplot(
        x_values,
        *y_values,
        labels=[label for _, label, _, _ in components],
        colors=["#FFFFFF" for _ in components],
        alpha=1.0,
        linewidth=0.0,
    )
    for collection, (_, _label, color, hatch) in zip(stacked, components, strict=True):
        collection.set_facecolor((1.0, 1.0, 1.0, 0.0))
        collection.set_edgecolor(color)
        collection.set_linewidth(0.0)
        collection.set_hatch(hatch)
    cumulative = [0.0 for _ in x_values]
    for values, (_, _, color, _) in zip(y_values, components, strict=True):
        cumulative = [current + value for current, value in zip(cumulative, values)]
        ax.plot(x_values, cumulative, color=color, linewidth=1.25, alpha=0.9)
        ax.scatter(
            x_values,
            cumulative,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            s=24,
            zorder=4,
        )
    ax.axhline(1.0, color="#222222", linewidth=0.7, alpha=0.35)
    ax.set_xlim(-3.0, 103.0)
    ax.set_ylim(0.0, 1.02)
    set_split_xticks(
        ax,
        x_values,
        x_values,
        suffix_values,
        labelsize=10.2,
        omit_prefixes={1, 5},
    )
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_ylabel("Mean attention mass")
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.65)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.26),
        ncol=4,
        frameon=False,
        fontsize=12.0,
        columnspacing=0.9,
        handlelength=1.8,
    )
    artifacts = save_figure(plt, figure_dir, "attention_native_stack")
    caption_path = figure_dir / "attention_native_stack.caption.txt"
    caption_path.write_text(
        (
            r"Mean attention allocation under native \fim{} probing, averaged over target-token "
            r"prediction queries and repetition buckets. The x-axis is linear in prefix length; "
            r"tick labels show prefix/suffix tokens."
            "\n"
        ),
        encoding="utf-8",
    )
    artifacts["caption"] = str(caption_path)
    return {"attention_native_stack": artifacts}


def main() -> None:
    args = parse_args()
    apply_suite_defaults(args)
    arms = build_arms(args)
    summaries = load_summaries(args, arms)
    rows = long_rows(args, arms, summaries)
    pairs = paired_rows(rows)
    overall = weighted_overall(args, arms, summaries)
    partition_rows = attention_partition_rows(rows, overall, arms)

    output_dir = args.output_dir or ((suite_root(args.results_root, args.suite) / "summaries") if args.suite else (args.results_root / "summaries"))
    output_prefix = args.output_prefix or comparison_stem(args)
    figure_dir = args.figure_dir or (((suite_root(args.results_root, args.suite) / "figures" / output_prefix) if args.suite else (args.results_root / "figures" / output_prefix)))
    per_arm_path = output_dir / f"{output_prefix}.per_arm.csv"
    paired_path = output_dir / f"{output_prefix}.paired.csv"
    partition_path = output_dir / f"{output_prefix}.attention_partition.csv"
    json_path = output_dir / f"{output_prefix}.summary.json"

    write_csv(per_arm_path, rows)
    write_csv(paired_path, pairs)
    write_csv(partition_path, partition_rows)
    figures: dict[str, Any] = {}
    if not args.skip_figures:
        figures.update(plot_repetition_curves(args, rows, arms, figure_dir))
        figures.update(plot_ltr_attention_partition_stacked(args, rows, arms, figure_dir))
        figures.update(plot_native_sweep(overall, arms, figure_dir))
        figures.update(plot_native_attention_stack(overall, arms, figure_dir))

    json_path.parent.mkdir(parents=True, exist_ok=True)
    partition_closes = all(bool(row["closes_to_one"]) for row in partition_rows)
    max_partition_abs_residual = max(
        (
            float(row["max_repetition_abs_residual"])
            for row in partition_rows
            if math.isfinite(float(row["max_repetition_abs_residual"]))
        ),
        default=float("nan"),
    )
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "study_name": args.study_name,
                "suite_name": args.suite,
                "suite_report": args.suite_report if args.suite else None,
                "arms": [arm.__dict__ for arm in arms],
                "overall": overall,
                "attention_partition_check": {
                    "components": ATTENTION_PARTITION_METRICS,
                    "tolerance": ATTENTION_PARTITION_TOLERANCE,
                    "closes_to_one": partition_closes,
                    "max_repetition_abs_residual": max_partition_abs_residual,
                    "explanation": (
                        "The partition components cover prefix tokens, suffix tokens, FIM sentinel tokens, "
                        "and previous target tokens. They should sum to one for each causal prediction query "
                        "because these regions exhaust the sequence positions available to attention."
                        if partition_closes
                        else "The partition did not close to one. This indicates either a missing token region "
                        "in the collation partition, nonzero attention assigned outside causal positions, or "
                        "attention tensors whose rows are not normalized as expected."
                    ),
                    "per_arm": partition_rows,
                },
                "artifacts": {
                    "per_arm_csv": str(per_arm_path),
                    "paired_csv": str(paired_path) if pairs else None,
                    "attention_partition_csv": str(partition_path),
                    "summary_json": str(json_path),
                    "figures": figures,
                },
            },
            handle,
            indent=2,
        )
    print(f"Wrote attention per-arm CSV: {per_arm_path}")
    if pairs:
        print(f"Wrote attention paired CSV: {paired_path}")
    print(f"Wrote attention partition CSV: {partition_path}")
    if partition_closes:
        print(f"Attention partition check passed: max abs residual={max_partition_abs_residual:.3g}")
    else:
        print(f"WARNING: attention partition check failed: max abs residual={max_partition_abs_residual:.3g}")
    print(f"Wrote attention summary JSON: {json_path}")


if __name__ == "__main__":
    main()
