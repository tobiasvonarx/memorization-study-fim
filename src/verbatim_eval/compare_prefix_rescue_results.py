#!/usr/bin/env python3
"""Collate prefix-rescue context-intervention results.

Prefix rescue keeps the native FIM target windows fixed while varying how much of the
prompt context is true prefix versus true suffix. Distractor interventions
replace one or both context sides with same-length text from a different
selected excerpt. This asks whether the sharp 0L/100R failure mode is primarily
a lack of left anchor rather than an intrinsic inability to use suffix context.
"""

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
    finite_ci95,
    finite_mean,
    finite_sem,
    interval_band,
)
from verbatim_suite import arm_root, load_manifest_for_suite, suite_arms_for_report, suite_root


DEFAULT_REPETITIONS = [1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128]
DEFAULT_SPLITS = "0:100,1:99,5:95,10:90,20:80,40:60,60:40,80:20,100:0"
DEFAULT_INTERVENTIONS = "full,suffix_distractor,prefix_distractor,both_distractor"
LOG10_PROB_FLOOR = 1e-8
LOG10_FULL_PZ_FLOOR = 1e-30
INTERVENTION_LABELS = {
    "full": "True prefix + true suffix",
    "suffix_distractor": "True prefix + distractor suffix",
    "prefix_distractor": "Distractor prefix + true suffix",
    "both_distractor": "Distractor prefix + distractor suffix",
}
INTERVENTION_COLORS = {
    "full": "#1f4e79",
    "suffix_distractor": "#2f7d32",
    "prefix_distractor": "#b23a48",
    "both_distractor": "#6f6f6f",
}
PROFILE_METRICS = [
    ("cooper_token_geomean_log10_p_z", "Cooper memorization", "mean log10 token-geomean p(z)"),
    ("Rouge-L", "ROUGE-L", "mean ROUGE-L"),
    ("cooper_extractable_per_10k", "Cooper extractable", "windows per 10k"),
    ("Ref_PPL", "Target PPL", "mean target PPL"),
]
FULL_PZ_METRIC = ("cooper_mean_log10_p_z", "Cooper full-span p(z)", "mean log10 p(z)")
FULL_PZ_LINEAR_METRIC = ("cooper_p_z", "Cooper full-span p(z)", "mean p(z)")
TOPK_SUPPORT_METRIC = ("cooper_supported_token_rate_pct", "Target tokens in top-k", "percent")
AGGREGATE_METRICS = PROFILE_METRICS + [FULL_PZ_METRIC, FULL_PZ_LINEAR_METRIC, TOPK_SUPPORT_METRIC]


@dataclass(frozen=True)
class ArmSpec:
    intervention: str
    prefix_length: int
    suffix_length: int
    arm_id: str | None = None

    @property
    def split_label(self) -> str:
        return f"{self.prefix_length}L/{self.suffix_length}R"


def default_results_root() -> Path:
    if os.environ.get("VERBATIM_EVAL_RESULTS_ROOT"):
        return Path(os.environ["VERBATIM_EVAL_RESULTS_ROOT"])
    if os.environ.get("RESULTS_ROOT"):
        return Path(os.environ["RESULTS_ROOT"])
    return Path(__file__).resolve().parents[2] / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prefix-rescue probes")
    parser.add_argument("--results-root", type=Path, default=default_results_root())
    parser.add_argument("--suite", default=None, help="Read arms from results/verbatim_eval/suites/<suite>")
    parser.add_argument("--study-name", default="exp3_prefix_rescue_ctx100_m20")
    parser.add_argument("--repetitions", type=int, nargs="+", default=DEFAULT_REPETITIONS)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--window-stride", type=int, default=120)
    parser.add_argument(
        "--window-layout",
        choices=["matched_context", "cooper_nonoverlap", "cooper_sliding"],
        default="matched_context",
    )
    parser.add_argument("--max-windows-per-excerpt", type=int, default=0)
    parser.add_argument("--context-budget", type=int, default=100)
    parser.add_argument("--middle-length", type=int, default=20)
    parser.add_argument("--prefix-splits", default=DEFAULT_SPLITS)
    parser.add_argument("--interventions", default=DEFAULT_INTERVENTIONS)
    parser.add_argument("--profile-repetitions", type=int, nargs="*", default=[32, 48, 64, 96, 128])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def apply_suite_defaults(args: argparse.Namespace) -> None:
    if not args.suite:
        return
    args.study_name = f"{args.suite}_prefix_rescue"
    args.window_stride = 120
    args.window_layout = "matched_context"
    args.max_windows_per_excerpt = 0
    args.context_budget = 100
    args.middle_length = 20
    args.prefix_splits = DEFAULT_SPLITS
    args.interventions = DEFAULT_INTERVENTIONS


def parse_splits(value: str, context_budget: int) -> list[tuple[int, int]]:
    splits: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in value.replace(",", " ").split():
        if ":" not in item:
            raise ValueError(f"Invalid split '{item}'. Expected prefix:suffix.")
        prefix_raw, suffix_raw = item.split(":", 1)
        prefix = int(prefix_raw)
        suffix = int(suffix_raw)
        if prefix < 0 or suffix < 0:
            raise ValueError(f"Split values must be non-negative: {item}")
        if prefix + suffix != context_budget:
            raise ValueError(f"Split {item} does not sum to context budget {context_budget}")
        split = (prefix, suffix)
        if split not in seen:
            seen.add(split)
            splits.append(split)
    if not splits:
        raise ValueError("No prefix-rescue splits requested")
    return splits


def parse_interventions(value: str) -> list[str]:
    interventions: list[str] = []
    seen: set[str] = set()
    allowed = set(INTERVENTION_LABELS)
    for item in value.replace(",", " ").split():
        if item not in allowed:
            raise ValueError(f"Unknown intervention '{item}'. Expected one of {sorted(allowed)}")
        if item not in seen:
            seen.add(item)
            interventions.append(item)
    if not interventions:
        raise ValueError("No interventions requested")
    return interventions


def output_stem(
    offset: int,
    window_stride: int,
    window_layout: str,
    max_windows_per_excerpt: int,
    prefix_length: int,
    middle_length: int,
    suffix_length: int,
    intervention: str,
) -> str:
    window_limit = "all" if max_windows_per_excerpt == 0 else str(max_windows_per_excerpt)
    layout_prefix = "" if window_layout == "matched_context" else f"layout_{window_layout}_"
    stem = (
        f"{layout_prefix}"
        f"target_offset_{offset}_stride_{window_stride}_"
        f"windows_{window_limit}_prefix_{prefix_length}_"
        f"middle_{middle_length}_suffix_{suffix_length}"
    )
    if intervention != "full":
        stem = f"{stem}_intervention_{intervention}"
    return stem


def summary_path(args: argparse.Namespace, arm: ArmSpec, repetition: int) -> Path:
    if args.suite:
        if arm.arm_id is None:
            raise ValueError("Suite prefix-rescue arms must have arm_id")
        return arm_root(args.results_root, args.suite, arm.arm_id, repetition) / "windows.summary.json"
    stem = output_stem(
        offset=args.offset,
        window_stride=args.window_stride,
        window_layout=args.window_layout,
        max_windows_per_excerpt=args.max_windows_per_excerpt,
        prefix_length=arm.prefix_length,
        middle_length=args.middle_length,
        suffix_length=arm.suffix_length,
        intervention=arm.intervention,
    )
    return (
        args.results_root
        / args.study_name
        / "fim_native"
        / "fim_v2"
        / f"rep_{repetition}"
        / f"{stem}.summary.json"
    )


def metric_mean(summary: dict[str, Any], metric: str) -> float:
    metrics = summary.get("metrics", {})
    if metric in metrics and isinstance(metrics[metric], dict):
        value = metrics[metric].get("mean")
        return float(value) if value is not None else float("nan")
    return float("nan")


def metric_std(summary: dict[str, Any], metric: str) -> float:
    metrics = summary.get("metrics", {})
    if metric in metrics and isinstance(metrics[metric], dict):
        value = metrics[metric].get("std")
        return float(value) if value is not None else float("nan")
    return float("nan")


def safe_log10(value: float, floor: float = LOG10_PROB_FLOOR) -> float:
    if not math.isfinite(value):
        return float("nan")
    return math.log10(max(value, floor))


def load_rows(args: argparse.Namespace, arms: list[ArmSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[Path] = []
    for arm in arms:
        for repetition in args.repetitions:
            path = summary_path(args, arm, repetition)
            if not path.exists():
                missing.append(path)
                continue
            with path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            num_windows = int(summary.get("num_windows", 0))
            cooper_extractable = metric_mean(summary, "cooper_extractable")
            cooper_token_geomean_p_z = metric_mean(summary, "cooper_token_geomean_p_z")
            cooper_p_z = metric_mean(summary, "cooper_p_z")
            row = {
                "study_name": args.study_name,
                "model_label": summary.get("model_label", "fim_v2"),
                "prompt_format": summary.get("prompt_format", "fim_native"),
                "intervention": arm.intervention,
                "intervention_label": INTERVENTION_LABELS[arm.intervention],
                "repetition": repetition,
                "prefix_length": arm.prefix_length,
                "suffix_length": arm.suffix_length,
                "split_label": arm.split_label,
                "num_windows": num_windows,
                "cooper_extractable": cooper_extractable,
                "cooper_extractable_per_10k": cooper_extractable * 10000.0,
                "cooper_supported_token_rate": metric_mean(summary, "cooper_supported_token_rate"),
                "cooper_supported_token_rate_std": metric_std(summary, "cooper_supported_token_rate"),
                "cooper_supported_token_rate_pct": metric_mean(summary, "cooper_supported_token_rate") * 100.0,
                "cooper_supported_token_rate_pct_std": metric_std(summary, "cooper_supported_token_rate") * 100.0,
                "cooper_token_geomean_p_z": cooper_token_geomean_p_z,
                "cooper_token_geomean_log10_p_z": safe_log10(cooper_token_geomean_p_z),
                "cooper_p_z": cooper_p_z,
                "cooper_mean_log10_p_z": safe_log10(cooper_p_z, LOG10_FULL_PZ_FLOOR),
                "cooper_log10_p_z": metric_mean(summary, "cooper_log10_p_z"),
                "Rouge-L": metric_mean(summary, "Rouge-L"),
                "Ref_NLL": metric_mean(summary, "Ref_NLL"),
                "Ref_PPL": metric_mean(summary, "Ref_PPL"),
                "memorization_score": metric_mean(summary, "memorization_score"),
                "summary_path": str(path),
            }
            rows.append(row)
    if missing and not args.allow_missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing summary files:\n{formatted}")
    if missing:
        print(f"Warning: skipped {len(missing)} missing summary files")
    if not rows:
        raise RuntimeError("No prefix-rescue summaries loaded")
    return rows


def aggregate_rows(rows: list[dict[str, Any]], profile_repetitions: set[int]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row["repetition"] in profile_repetitions:
            groups.setdefault((row["intervention"], row["prefix_length"], row["suffix_length"]), []).append(row)

    aggregate: list[dict[str, Any]] = []
    for (intervention, prefix_length, suffix_length), group_rows in sorted(groups.items()):
        out: dict[str, Any] = {
            "intervention": intervention,
            "intervention_label": INTERVENTION_LABELS[intervention],
            "prefix_length": prefix_length,
            "suffix_length": suffix_length,
            "split_label": f"{prefix_length}L/{suffix_length}R",
            "num_repetition_buckets": len(group_rows),
            "num_windows_total": sum(int(row["num_windows"]) for row in group_rows),
        }
        for metric, _title, _ylabel in AGGREGATE_METRICS:
            values = [float(row[metric]) for row in group_rows]
            out[f"{metric}_mean"] = finite_mean(values)
            out[f"{metric}_sem"] = finite_sem(values)
            out[f"{metric}_ci95"] = finite_ci95(values)
        aggregate.append(out)
    return aggregate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_profile(args: argparse.Namespace, aggregate: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    apply_conference_style(plt)
    interventions = parse_interventions(args.interventions)
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), constrained_layout=True)
    axes_flat = list(axes.ravel())
    x_ticks = [prefix for prefix, _suffix in parse_splits(args.prefix_splits, args.context_budget)]

    for axis, (metric, title, ylabel) in zip(axes_flat, PROFILE_METRICS):
        for intervention in interventions:
            series = sorted(
                [row for row in aggregate if row["intervention"] == intervention],
                key=lambda row: row["prefix_length"],
            )
            if not series:
                continue
            x = [row["prefix_length"] for row in series]
            y = [row[f"{metric}_mean"] for row in series]
            yerr = [
                0.0 if not math.isfinite(row[f"{metric}_ci95"]) else row[f"{metric}_ci95"]
                for row in series
            ]
            lower_floor = None if metric == "cooper_token_geomean_log10_p_z" else 0.0
            upper_ceiling = 1.0 if metric == "Rouge-L" else None
            lower, upper = interval_band(
                y,
                yerr,
                lower_floor=lower_floor,
                upper_ceiling=upper_ceiling,
            )
            color = INTERVENTION_COLORS[intervention]
            axis.fill_between(
                x,
                lower,
                upper,
                color=color,
                alpha=0.15,
                linewidth=0.0,
                zorder=1,
            )
            axis.plot(
                x,
                y,
                color=color,
                label=INTERVENTION_LABELS[intervention],
                linewidth=2.0,
                solid_capstyle="round",
                zorder=3,
            )
        axis.set_title(title)
        axis.set_xlabel("Prefix-slot tokens (P), with suffix-slot S=100-P")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x_ticks)
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", color="#dddddd", linewidth=0.45, alpha=0.45)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(
        "Prefix-rescue context interventions",
        fontsize=12,
        fontweight="semibold",
        y=1.09,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_cooper_profile(args: argparse.Namespace, aggregate: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    apply_conference_style(plt)
    interventions = parse_interventions(args.interventions)
    x_ticks = [prefix for prefix, _suffix in parse_splits(args.prefix_splits, args.context_budget)]
    metric = "cooper_token_geomean_log10_p_z"

    fig, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for intervention in interventions:
        series = sorted(
            [row for row in aggregate if row["intervention"] == intervention],
            key=lambda row: row["prefix_length"],
        )
        if not series:
            continue
        x = [row["prefix_length"] for row in series]
        y = [row[f"{metric}_mean"] for row in series]
        yerr = [
            0.0 if not math.isfinite(row[f"{metric}_ci95"]) else row[f"{metric}_ci95"]
            for row in series
        ]
        lower, upper = interval_band(y, yerr)
        color = INTERVENTION_COLORS[intervention]
        axis.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.15,
            linewidth=0.0,
            zorder=1,
        )
        axis.plot(
            x,
            y,
            color=color,
            label=INTERVENTION_LABELS[intervention],
            linewidth=2.0,
            solid_capstyle="round",
            zorder=3,
        )

    axis.set_xlabel("Prefix-slot tokens (P), with suffix-slot S=100-P")
    axis.set_ylabel(f"Mean log10 token-geomean p(z), clipped at {LOG10_PROB_FLOOR:g}")
    axis.set_xticks(x_ticks)
    axis.tick_params(axis="x", labelrotation=35)
    axis.grid(axis="y", color="#dddddd", linewidth=0.45, alpha=0.45)
    axis.legend(loc="lower right", frameon=False, fontsize=8.8)
    axis.set_title("Prefix-slot rescue under native FIM extraction", fontsize=11, fontweight="semibold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_cooper_pz_profile(args: argparse.Namespace, aggregate: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    apply_conference_style(plt)
    interventions = parse_interventions(args.interventions)
    x_ticks = [prefix for prefix, _suffix in parse_splits(args.prefix_splits, args.context_budget)]
    metric = "cooper_mean_log10_p_z"

    fig, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for intervention in interventions:
        series = sorted(
            [row for row in aggregate if row["intervention"] == intervention],
            key=lambda row: row["prefix_length"],
        )
        if not series:
            continue
        x = [row["prefix_length"] for row in series]
        y = [row[f"{metric}_mean"] for row in series]
        yerr = [
            0.0 if not math.isfinite(row[f"{metric}_ci95"]) else row[f"{metric}_ci95"]
            for row in series
        ]
        lower, upper = interval_band(y, yerr)
        color = INTERVENTION_COLORS[intervention]
        axis.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.15,
            linewidth=0.0,
            zorder=1,
        )
        axis.plot(
            x,
            y,
            color=color,
            label=INTERVENTION_LABELS[intervention],
            linewidth=2.0,
            solid_capstyle="round",
            zorder=3,
        )

    axis.set_xlabel("Prefix-slot tokens (P), with suffix-slot S=100-P")
    axis.set_ylabel(f"Mean log10 p(z), clipped at {LOG10_FULL_PZ_FLOOR:g}")
    axis.set_xticks(x_ticks)
    axis.tick_params(axis="x", labelrotation=35)
    axis.grid(axis="y", color="#dddddd", linewidth=0.45, alpha=0.45)
    axis.legend(loc="lower right", frameon=False, fontsize=8.8)
    axis.set_title("Full-span Cooper p(z) across prefix rescue", fontsize=11, fontweight="semibold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_cooper_heatmap(args: argparse.Namespace, rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    interventions = parse_interventions(args.interventions)
    splits = parse_splits(args.prefix_splits, args.context_budget)
    prefixes = [prefix for prefix, _suffix in splits]
    repetitions = list(args.repetitions)
    fig, axes = plt.subplots(
        len(interventions),
        1,
        figsize=(9.2, 1.9 + 1.45 * len(interventions)),
        sharex=True,
        constrained_layout=True,
    )
    if len(interventions) == 1:
        axes = [axes]

    values_by_key = {
        (row["intervention"], row["repetition"], row["prefix_length"]): row[
            "cooper_token_geomean_log10_p_z"
        ]
        for row in rows
    }
    all_values = [
        float(row["cooper_token_geomean_log10_p_z"])
        for row in rows
        if math.isfinite(float(row["cooper_token_geomean_log10_p_z"]))
    ]
    vmin = min(all_values) if all_values else None
    vmax = max(all_values) if all_values else None
    image = None

    for axis, intervention in zip(axes, interventions):
        matrix = np.full((len(repetitions), len(prefixes)), np.nan)
        for i, repetition in enumerate(repetitions):
            for j, prefix in enumerate(prefixes):
                value = values_by_key.get((intervention, repetition, prefix), float("nan"))
                matrix[i, j] = value
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(INTERVENTION_LABELS[intervention], fontsize=10, loc="left")
        axis.set_yticks(range(len(repetitions)))
        axis.set_yticklabels(repetitions)
        axis.set_ylabel("Repetition")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
    axes[-1].set_xticks(range(len(prefixes)))
    axes[-1].set_xticklabels(prefixes)
    axes[-1].set_xlabel("Prefix-slot tokens (P), with suffix-slot S=100-P")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.84, pad=0.015)
        colorbar.set_label(f"Cooper mean log10 token-geomean p(z), clipped at {LOG10_PROB_FLOOR:g}")
    fig.suptitle("Cooper memorization across repetition and prefix rescue", fontsize=12, fontweight="semibold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_cooper_pz_profile_linear(args: argparse.Namespace, aggregate: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    apply_conference_style(plt)
    interventions = parse_interventions(args.interventions)
    x_ticks = [prefix for prefix, _suffix in parse_splits(args.prefix_splits, args.context_budget)]
    metric = "cooper_p_z"

    fig, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for intervention in interventions:
        series = sorted(
            [row for row in aggregate if row["intervention"] == intervention],
            key=lambda row: row["prefix_length"],
        )
        if not series:
            continue
        x = [row["prefix_length"] for row in series]
        y = [row[f"{metric}_mean"] for row in series]
        yerr = [
            0.0 if not math.isfinite(row[f"{metric}_ci95"]) else row[f"{metric}_ci95"]
            for row in series
        ]
        lower, upper = interval_band(y, yerr, lower_floor=0.0)
        color = INTERVENTION_COLORS[intervention]
        axis.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.15,
            linewidth=0.0,
            zorder=1,
        )
        axis.plot(
            x,
            y,
            color=color,
            label=INTERVENTION_LABELS[intervention],
            linewidth=2.0,
            solid_capstyle="round",
            zorder=3,
        )

    axis.set_xlabel("Prefix-slot tokens (P), with suffix-slot S=100-P")
    axis.set_ylabel("Mean p(z)")
    axis.set_xticks(x_ticks)
    axis.tick_params(axis="x", labelrotation=35)
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axis.grid(axis="y", color="#dddddd", linewidth=0.45, alpha=0.45)
    axis.legend(loc="upper left", frameon=False, fontsize=8.8)
    axis.set_title("Full-span Cooper p(z) across prefix rescue", fontsize=11, fontweight="semibold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def add_prefix_suffix_tick_rows(axis: Any, args: argparse.Namespace, *, tick_size: float = 10.2) -> None:
    splits = parse_splits(args.prefix_splits, args.context_budget)
    prefixes = [prefix for prefix, _suffix in splits]
    axis.set_xlim(-3.0, args.context_budget + 3.0)
    axis.set_xticks(prefixes)
    axis.set_xticklabels(
        ["" if prefix in {1, 5} else f"{prefix}/{suffix}" for prefix, suffix in splits],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
        color="black",
        fontsize=tick_size,
    )
    axis.tick_params(axis="x", colors="black")


def plot_target_ppl_profile(args: argparse.Namespace, aggregate: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    apply_conference_style(plt)
    interventions = parse_interventions(args.interventions)
    metric = "Ref_PPL"
    ppl_colors = {
        "full": "#111111",
        "suffix_distractor": "#8A3A0A",
        "prefix_distractor": "#7A1F55",
        "both_distractor": "#6F6F6F",
    }

    fig, axis = plt.subplots(figsize=(7.2, 4.05))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.215, top=0.78)
    for intervention in interventions:
        series = sorted(
            [row for row in aggregate if row["intervention"] == intervention],
            key=lambda row: row["prefix_length"],
        )
        if not series:
            continue
        x = [row["prefix_length"] for row in series]
        y = [row[f"{metric}_mean"] for row in series]
        yerr = [
            0.0 if not math.isfinite(row[f"{metric}_ci95"]) else row[f"{metric}_ci95"]
            for row in series
        ]
        lower, upper = interval_band(y, yerr, lower_floor=0.0)
        color = ppl_colors.get(intervention, INTERVENTION_COLORS[intervention])
        axis.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.15,
            linewidth=0.0,
            zorder=1,
        )
        axis.plot(
            x,
            y,
            color=color,
            label=INTERVENTION_LABELS[intervention],
            linewidth=2.0,
            marker="o",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=4.0,
            solid_capstyle="round",
            zorder=3,
        )

    add_prefix_suffix_tick_rows(axis, args)
    axis.set_ylabel("Target PPL")
    axis.grid(axis="y", color="#dddddd", linewidth=0.45, alpha=0.45)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=2,
        frameon=False,
        fontsize=8.8,
        columnspacing=1.1,
        handlelength=1.5,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_topk_support_profile(args: argparse.Namespace, rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    apply_conference_style(plt)
    interventions = parse_interventions(args.interventions)
    metric = "cooper_supported_token_rate_pct"
    profile_repetition = 128
    topk_colors = {
        "full": "#111111",
        "suffix_distractor": "#9A7B24",
        "prefix_distractor": "#5F5A8B",
        "both_distractor": "#7A7A7A",
    }

    fig, axis = plt.subplots(figsize=(7.2, 3.75))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.23, top=0.83)
    for intervention in interventions:
        series = sorted(
            [
                row
                for row in rows
                if row["intervention"] == intervention
                and int(row["repetition"]) == profile_repetition
            ],
            key=lambda row: row["prefix_length"],
        )
        if not series:
            continue
        x = [row["prefix_length"] for row in series]
        y = [row[metric] for row in series]
        yerr = [
            ci95_from_std(
                float(row.get("cooper_supported_token_rate_pct_std", float("nan"))),
                int(row.get("num_windows", 0)),
            )
            for row in series
        ]
        lower, upper = interval_band(y, yerr, lower_floor=0.0, upper_ceiling=100.0)
        color = topk_colors.get(intervention, INTERVENTION_COLORS[intervention])
        axis.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.15,
            linewidth=0.0,
            zorder=1,
        )
        axis.plot(
            x,
            y,
            color=color,
            label=INTERVENTION_LABELS[intervention],
            linewidth=2.0,
            marker="o",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=4.0,
            solid_capstyle="round",
            zorder=3,
        )

    add_prefix_suffix_tick_rows(axis, args)
    axis.set_ylabel("Target tokens in top-k (%)")
    axis.grid(axis="y", color="#dddddd", linewidth=0.45, alpha=0.45)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.27),
        ncol=2,
        frameon=False,
        fontsize=11.4,
        columnspacing=1.25,
        handlelength=2.0,
        handletextpad=0.55,
        labelspacing=0.35,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_cooper_pz_heatmap(args: argparse.Namespace, rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    interventions = parse_interventions(args.interventions)
    splits = parse_splits(args.prefix_splits, args.context_budget)
    prefixes = [prefix for prefix, _suffix in splits]
    repetitions = list(args.repetitions)
    fig, axes = plt.subplots(
        len(interventions),
        1,
        figsize=(9.2, 1.9 + 1.45 * len(interventions)),
        sharex=True,
        constrained_layout=True,
    )
    if len(interventions) == 1:
        axes = [axes]

    values_by_key = {
        (row["intervention"], row["repetition"], row["prefix_length"]): row[
            "cooper_mean_log10_p_z"
        ]
        for row in rows
    }
    all_values = [
        float(row["cooper_mean_log10_p_z"])
        for row in rows
        if math.isfinite(float(row["cooper_mean_log10_p_z"]))
    ]
    vmin = min(all_values) if all_values else None
    vmax = max(all_values) if all_values else None
    image = None

    for axis, intervention in zip(axes, interventions):
        matrix = np.full((len(repetitions), len(prefixes)), np.nan)
        for i, repetition in enumerate(repetitions):
            for j, prefix in enumerate(prefixes):
                value = values_by_key.get((intervention, repetition, prefix), float("nan"))
                matrix[i, j] = value
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(INTERVENTION_LABELS[intervention], fontsize=10, loc="left")
        axis.set_yticks(range(len(repetitions)))
        axis.set_yticklabels(repetitions)
        axis.set_ylabel("Repetition")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
    axes[-1].set_xticks(range(len(prefixes)))
    axes[-1].set_xticklabels(prefixes)
    axes[-1].set_xlabel("Prefix-slot tokens (P), with suffix-slot S=100-P")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.84, pad=0.015)
        colorbar.set_label(f"Cooper mean log10 p(z), clipped at {LOG10_FULL_PZ_FLOOR:g}")
    fig.suptitle("Full-span Cooper p(z) across repetition and prefix rescue", fontsize=12, fontweight="semibold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    artifacts: dict[str, str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile_repetitions = sorted({row["repetition"] for row in rows if row["repetition"] in set(args.profile_repetitions)})
    payload = {
        "study_name": args.study_name,
        "question": "Does adding a small true prefix rescue native FIM extraction when suffix context is present?",
        "context_budget": args.context_budget,
        "middle_length": args.middle_length,
        "prefix_splits": args.prefix_splits,
        "interventions": args.interventions,
        "repetitions": args.repetitions,
        "profile_repetitions": profile_repetitions,
        "num_loaded_cells": len(rows),
        "num_profile_cells": len(aggregate),
        "artifacts": artifacts,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    apply_suite_defaults(args)
    if args.suite:
        manifest = load_manifest_for_suite(args.results_root, args.suite)
        arms = [
            ArmSpec(
                intervention=str(arm.get("context_intervention", "full")),
                prefix_length=int(arm["prefix_length"]),
                suffix_length=int(arm["suffix_length"]),
                arm_id=str(arm["arm_id"]),
            )
            for arm in suite_arms_for_report(manifest, "prefix_rescue")
        ]
    else:
        splits = parse_splits(args.prefix_splits, args.context_budget)
        interventions = parse_interventions(args.interventions)
        arms = [
            ArmSpec(intervention=intervention, prefix_length=prefix, suffix_length=suffix)
            for intervention in interventions
            for prefix, suffix in splits
        ]
    output_prefix = args.output_prefix or args.study_name
    if args.suite:
        suite_out = suite_root(args.results_root, args.suite)
        output_dir = args.output_dir or (suite_out / "summaries")
        figure_dir = args.figure_dir or (suite_out / "figures" / output_prefix)
    else:
        output_dir = args.output_dir or (args.results_root / "summaries")
        figure_dir = args.figure_dir or (args.results_root / "figures" / output_prefix)

    rows = load_rows(args, arms)
    profile_repetitions = set(args.profile_repetitions or args.repetitions)
    aggregate = aggregate_rows(rows, profile_repetitions)

    per_rep_csv = output_dir / f"{output_prefix}.prefix_rescue_by_repetition.csv"
    aggregate_csv = output_dir / f"{output_prefix}.prefix_rescue_profile.csv"
    summary_json = output_dir / f"{output_prefix}.prefix_rescue.summary.json"
    write_csv(per_rep_csv, rows)
    write_csv(aggregate_csv, aggregate)

    artifacts = {
        "by_repetition_csv": str(per_rep_csv),
        "profile_csv": str(aggregate_csv),
    }
    if not args.skip_figures:
        cooper_profile_pdf = figure_dir / f"{output_prefix}.prefix_rescue_cooper_profile.pdf"
        cooper_pz_profile_pdf = figure_dir / f"{output_prefix}.prefix_rescue_cooper_pz_profile.pdf"
        cooper_pz_profile_linear_pdf = figure_dir / f"{output_prefix}.prefix_rescue_cooper_pz_profile_linear.pdf"
        target_ppl_profile_pdf = figure_dir / f"{output_prefix}.prefix_rescue_target_ppl_profile.pdf"
        target_ppl_caption = figure_dir / f"{output_prefix}.prefix_rescue_target_ppl_profile.caption.txt"
        topk_profile_pdf = figure_dir / f"{output_prefix}.prefix_rescue_topk_support_profile.pdf"
        topk_caption = figure_dir / f"{output_prefix}.prefix_rescue_topk_support_profile.caption.txt"
        profile_pdf = figure_dir / f"{output_prefix}.prefix_rescue_profile.pdf"
        heatmap_pdf = figure_dir / f"{output_prefix}.prefix_rescue_cooper_heatmap.pdf"
        pz_heatmap_pdf = figure_dir / f"{output_prefix}.prefix_rescue_cooper_pz_heatmap.pdf"
        plot_cooper_profile(args, aggregate, cooper_profile_pdf)
        plot_cooper_pz_profile(args, aggregate, cooper_pz_profile_pdf)
        plot_cooper_pz_profile_linear(args, aggregate, cooper_pz_profile_linear_pdf)
        plot_target_ppl_profile(args, aggregate, target_ppl_profile_pdf)
        plot_topk_support_profile(args, rows, topk_profile_pdf)
        target_ppl_caption.write_text(
            (
                r"Target perplexity under prefix-rescue context interventions. "
                r"Tick labels show prefix/suffix tokens; "
                r"bands are t-based 95\% confidence intervals over the profile repetition buckets."
                "\n"
            ),
            encoding="utf-8",
        )
        topk_caption.write_text(
            (
                r"Target-token top-$k$ support under prefix-rescue context interventions. "
                r"Tick labels show prefix/suffix tokens; "
                r"lines show the 128-repetition bucket and bands are nominal 95\% confidence intervals over windows."
                "\n"
            ),
            encoding="utf-8",
        )
        plot_profile(args, aggregate, profile_pdf)
        plot_cooper_heatmap(args, rows, heatmap_pdf)
        plot_cooper_pz_heatmap(args, rows, pz_heatmap_pdf)
        artifacts["cooper_profile_figure"] = str(cooper_profile_pdf)
        artifacts["cooper_pz_profile_figure"] = str(cooper_pz_profile_pdf)
        artifacts["cooper_pz_profile_linear_figure"] = str(cooper_pz_profile_linear_pdf)
        artifacts["target_ppl_profile_figure"] = str(target_ppl_profile_pdf)
        artifacts["target_ppl_profile_caption"] = str(target_ppl_caption)
        artifacts["topk_support_profile_figure"] = str(topk_profile_pdf)
        artifacts["topk_support_profile_caption"] = str(topk_caption)
        artifacts["profile_figure"] = str(profile_pdf)
        artifacts["cooper_heatmap"] = str(heatmap_pdf)
        artifacts["cooper_pz_heatmap"] = str(pz_heatmap_pdf)

    write_summary(args, rows, aggregate, artifacts, summary_json)
    print(f"Loaded {len(rows)} prefix-rescue summary cells")
    print(f"Wrote per-repetition CSV: {per_rep_csv}")
    print(f"Wrote profile CSV: {aggregate_csv}")
    print(f"Wrote summary JSON: {summary_json}")
    if not args.skip_figures:
        print(f"Wrote Cooper profile figure: {artifacts['cooper_profile_figure']}")
        print(f"Wrote Cooper p_z profile figure: {artifacts['cooper_pz_profile_figure']}")
        print(f"Wrote Cooper p_z linear profile figure: {artifacts['cooper_pz_profile_linear_figure']}")
        print(f"Wrote target PPL profile figure: {artifacts['target_ppl_profile_figure']}")
        print(f"Wrote top-k support profile figure: {artifacts['topk_support_profile_figure']}")
        print(f"Wrote profile figure: {artifacts['profile_figure']}")
        print(f"Wrote Cooper heatmap: {artifacts['cooper_heatmap']}")
        print(f"Wrote Cooper p_z heatmap: {artifacts['cooper_pz_heatmap']}")


if __name__ == "__main__":
    main()
