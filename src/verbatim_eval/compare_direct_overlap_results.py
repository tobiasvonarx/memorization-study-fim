#!/usr/bin/env python3
"""Collate and analyze the maintained verbatim memorization probes.

This comparison layer is intentionally opinionated: it studies no-FIM, FIM-v2,
and FineWeb-only checkpoints under LTR probing, and studies FIM-v2 under native
FIM prefix/suffix geometry. It writes compact aggregate tables, paired
same-window diagnostics, figures, and an auto-generated insights note.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
from verbatim_suite import arm_root, load_manifest_for_suite, suite_arms_for_report, suite_root


DEFAULT_REPETITIONS = [1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128]
LTR_COOPER_MIN_FLOOR = 1e-30
COOPER_PREFIX_TARGET_LENGTHS = (20, 30, 32)
CORE_METRICS = [
    "cooper_extractable",
    "cooper_supported_token_rate",
    "cooper_token_geomean_p_z",
    "cooper_p_z",
    "greedy_exact_match",
    "exact_match",
    "memorization_score",
    "token_accuracy",
    "Rouge-L",
    "LCS",
    "Ref_NLL",
    "Ref_PPL",
    "log_p_target",
]
CORE_METRICS.extend(
    metric
    for span_length in COOPER_PREFIX_TARGET_LENGTHS
    for metric in (
        f"cooper_extractable_first{span_length}",
        f"cooper_p_z_first{span_length}",
    )
)
COUNT_METRICS = [
    "cooper_extractable",
    "cooper_all_tokens_in_topk",
    "greedy_exact_match",
    "exact_match",
    "match_ge_0_25",
    "match_ge_0_50",
    "match_ge_0_75",
]
COUNT_METRICS.extend(f"cooper_extractable_first{span_length}" for span_length in COOPER_PREFIX_TARGET_LENGTHS)
WINDOW_FLOAT_FIELDS = [
    "cooper_extractable",
    "cooper_supported_token_rate",
    "cooper_token_geomean_p_z",
    "cooper_p_z",
    "cooper_log_p_z",
    "cooper_log10_p_z",
    "greedy_exact_match",
    "exact_match",
    "memorization_score",
    "token_accuracy",
    "Rouge-L",
    "LCS",
    "Ref_NLL",
    "Ref_PPL",
    "log_p_target",
]
WINDOW_FLOAT_FIELDS.extend(
    metric
    for span_length in COOPER_PREFIX_TARGET_LENGTHS
    for metric in (
        f"cooper_extractable_first{span_length}",
        f"cooper_p_z_first{span_length}",
    )
)


def parse_summary_span_lengths(summary: dict[str, Any]) -> tuple[int, ...]:
    explicit = summary.get("cooper_prefix_target_lengths")
    if explicit:
        spans = sorted({int(span) for span in explicit})
        if spans:
            return tuple(spans)

    found: set[int] = set()
    for metric_name in summary.get("metrics", {}):
        match = re.fullmatch(r"cooper_(?:extractable|p_z)_first(\d+)", str(metric_name))
        if match:
            found.add(int(match.group(1)))
    if found:
        return tuple(sorted(found))
    return COOPER_PREFIX_TARGET_LENGTHS


def resolve_cooper_prefix_target_lengths(
    summaries: dict[str, dict[int, dict[str, Any]]],
) -> tuple[int, ...]:
    found: set[int] = set()
    for summary_by_rep in summaries.values():
        for summary in summary_by_rep.values():
            found.update(parse_summary_span_lengths(summary))
    return tuple(sorted(found)) if found else COOPER_PREFIX_TARGET_LENGTHS


def cooper_prefix_target_lengths(args: argparse.Namespace) -> tuple[int, ...]:
    spans = getattr(args, "cooper_prefix_target_lengths", None)
    if spans:
        return tuple(sorted({int(span) for span in spans}))
    return COOPER_PREFIX_TARGET_LENGTHS


def core_metrics_for_args(args: argparse.Namespace) -> list[str]:
    metrics = list(CORE_METRICS)
    for span_length in cooper_prefix_target_lengths(args):
        for metric in (
            f"cooper_extractable_first{span_length}",
            f"cooper_p_z_first{span_length}",
        ):
            if metric not in metrics:
                metrics.append(metric)
    return metrics


def count_metrics_for_args(args: argparse.Namespace) -> list[str]:
    metrics = list(COUNT_METRICS)
    for span_length in cooper_prefix_target_lengths(args):
        metric = f"cooper_extractable_first{span_length}"
        if metric not in metrics:
            metrics.append(metric)
    return metrics


def window_float_fields_for_args(args: argparse.Namespace) -> list[str]:
    fields = list(WINDOW_FLOAT_FIELDS)
    for span_length in cooper_prefix_target_lengths(args):
        for metric in (
            f"cooper_extractable_first{span_length}",
            f"cooper_p_z_first{span_length}",
        ):
            if metric not in fields:
                fields.append(metric)
    return fields


@dataclass(frozen=True)
class ArmSpec:
    arm_label: str
    display_label: str
    study_name: str
    prompt_format: str
    model_label: str
    prefix_length: int
    suffix_length: int
    arm_id: str | None = None
    context_intervention: str = "full"
    experiment: str | None = None

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
    parser = argparse.ArgumentParser(description="Analyze direct-overlap memorization probes")
    parser.add_argument("--results-root", type=Path, default=default_results_root())
    parser.add_argument("--suite", default=None, help="Read arms from results/verbatim_eval/suites/<suite>")
    parser.add_argument(
        "--suite-report",
        choices=["ltr", "native_geometry"],
        default="ltr",
        help="Suite report to collate when --suite is set.",
    )
    parser.add_argument("--study-name", default="exp1_ltr_p100_m20")
    parser.add_argument("--repetitions", type=int, nargs="+", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--window-stride", type=int, default=120)
    parser.add_argument(
        "--window-layout",
        choices=["matched_context", "cooper_nonoverlap", "cooper_sliding"],
        default="cooper_nonoverlap",
    )
    parser.add_argument("--max-windows-per-excerpt", type=int, default=0)
    parser.add_argument("--context-budget", type=int, default=100)
    parser.add_argument("--middle-length", type=int, default=20)
    parser.add_argument(
        "--native-splits",
        default="none",
        help="Comma/space-separated FIM native prefix:suffix splits, or 'none'.",
    )
    parser.add_argument("--no-ltr-arms", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--skip-audit-figures",
        action="store_true",
        help="Do not generate LTR appendix audit figures during suite LTR collation.",
    )
    parser.add_argument("--audit-output-dir", type=Path, default=None)
    parser.add_argument("--audit-repetitions", default="128")
    parser.add_argument("--audit-examples-per-model", type=int, default=2)
    parser.add_argument("--audit-model-labels", default="fim_v2,no_fim,fineweb_only")
    parser.add_argument("--audit-device", default="cuda:0")
    parser.add_argument(
        "--skip-token-density-figure",
        action="store_true",
        help="Skip the GPU-backed token-level memorized-window density figure.",
    )
    parser.add_argument("--density-device", default="cuda:0")
    parser.add_argument("--density-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--density-batch-size", type=int, default=32)
    parser.add_argument(
        "--density-repetition",
        type=int,
        default=None,
        help="Repetition bucket for token-level density recomputation; defaults to the largest requested repetition.",
    )
    parser.add_argument(
        "--scatter-repetition",
        type=int,
        default=None,
        help="Repetition bucket for per-window Cooper survival diagnostics; defaults to the largest requested repetition.",
    )
    parser.add_argument("--no-validate-window-grid", action="store_true")
    return parser.parse_args()


def apply_suite_report_defaults(args: argparse.Namespace) -> None:
    if not args.suite:
        if args.repetitions is None:
            args.repetitions = DEFAULT_REPETITIONS
        return
    args.study_name = f"{args.suite}_{args.suite_report}"
    args.max_windows_per_excerpt = 0
    manifest = load_manifest_for_suite(args.results_root, args.suite)
    if args.repetitions is None:
        args.repetitions = [int(rep) for rep in manifest.get("repetitions", [])] or DEFAULT_REPETITIONS
    report_arms = suite_arms_for_report(manifest, args.suite_report)
    if report_arms:
        middle_lengths = {int(arm["middle_length"]) for arm in report_arms}
        window_strides = {int(arm["window_stride"]) for arm in report_arms}
        window_layouts = {str(arm["window_layout"]) for arm in report_arms}
        context_budgets = {int(arm["context_budget"]) for arm in report_arms}
        if len(middle_lengths) == 1:
            args.middle_length = middle_lengths.pop()
        if len(window_strides) == 1:
            args.window_stride = window_strides.pop()
        if len(window_layouts) == 1:
            args.window_layout = window_layouts.pop()
        if len(context_budgets) == 1:
            args.context_budget = context_budgets.pop()
    if args.suite_report == "ltr":
        args.native_splits = "none"
        args.no_ltr_arms = False
    elif args.suite_report == "native_geometry":
        args.native_splits = "0:100,20:80,40:60,60:40,80:20,100:0"
        args.no_ltr_arms = True


def log_axis_ylim(
    values: Iterable[float],
    errors: Iterable[float] = (),
    *,
    min_floor: float = LTR_COOPER_MIN_FLOOR,
    pad_decades: float = 0.28,
) -> tuple[float, float] | None:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    finite_errors = [float(error) for error in errors if math.isfinite(float(error))]
    use_errors = len(finite_values) == len(finite_errors)
    lows: list[float] = []
    highs: list[float] = []
    for index, value in enumerate(finite_values):
        if value <= 0.0:
            continue
        spread = finite_errors[index] if use_errors else 0.0
        spread = max(0.0, spread) if math.isfinite(spread) else 0.0
        lower = value - spread
        if not math.isfinite(lower) or lower <= 0.0:
            lower = value * 0.2
        upper = value + spread
        if not math.isfinite(upper) or upper <= value:
            upper = value
        lows.append(max(lower, min_floor))
        highs.append(max(upper, min_floor * 10.0))
    if not lows or not highs:
        return None
    lower = max(min_floor, 10 ** (math.log10(min(lows)) - pad_decades))
    upper = 10 ** (math.log10(max(highs)) + pad_decades)
    if upper <= lower:
        upper = lower * 10.0
    return lower, upper


def padded_linear_upper(
    values: Iterable[float],
    *,
    minimum: float = 1.0,
    ceiling: float = 100.0,
    pad: float = 0.12,
) -> float:
    finite = [max(0.0, float(value)) for value in values if math.isfinite(float(value))]
    if not finite:
        return minimum
    target = max(finite) * (1.0 + pad)
    if target <= minimum:
        return minimum
    magnitude = 10 ** math.floor(math.log10(target))
    for multiplier in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        candidate = multiplier * magnitude
        if target <= candidate:
            return min(ceiling, max(minimum, candidate))
    return ceiling


def suite_arm_spec(args: argparse.Namespace, arm: dict[str, Any]) -> ArmSpec:
    prefix = int(arm["prefix_length"])
    suffix = int(arm["suffix_length"])
    if arm["experiment"] == "ltr":
        arm_label = {
            "no_fim": "no_fim_ltr",
            "fim_v2": "fim_v2_ltr",
            "fineweb_only": "fineweb_only_ltr",
        }.get(str(arm["model_label"]), str(arm["arm_id"]))
    else:
        arm_label = f"fim_v2_native_p{prefix}_s{suffix}"
    return ArmSpec(
        arm_label=arm_label,
        display_label=str(arm.get("display_label", arm_label)),
        study_name=args.study_name,
        prompt_format=str(arm["prompt_format"]),
        model_label=str(arm["model_label"]),
        prefix_length=prefix,
        suffix_length=suffix,
        arm_id=str(arm["arm_id"]),
        context_intervention=str(arm.get("context_intervention", "full")),
        experiment=str(arm.get("experiment", "")),
    )


def parse_native_splits(value: str, context_budget: int) -> list[tuple[int, int]]:
    if value.strip().lower() in {"", "none", "null", "off"}:
        return []
    splits: list[tuple[int, int]] = []
    for item in value.replace(",", " ").split():
        if ":" not in item:
            raise ValueError(f"Invalid native split '{item}'. Expected prefix:suffix.")
        prefix_raw, suffix_raw = item.split(":", 1)
        prefix = int(prefix_raw)
        suffix = int(suffix_raw)
        if prefix < 0 or suffix < 0:
            raise ValueError(f"Native split values must be non-negative: {item}")
        if prefix + suffix != context_budget:
            raise ValueError(
                f"Native split {item} does not match --context-budget={context_budget}"
            )
        splits.append((prefix, suffix))

    deduped: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for split in splits:
        if split in seen:
            continue
        seen.add(split)
        deduped.append(split)
    return deduped


def build_arms(args: argparse.Namespace) -> list[ArmSpec]:
    if args.suite:
        manifest = load_manifest_for_suite(args.results_root, args.suite)
        return [suite_arm_spec(args, arm) for arm in suite_arms_for_report(manifest, args.suite_report)]

    arms: list[ArmSpec] = []
    if not args.no_ltr_arms:
        arms.extend(
            [
                ArmSpec(
                    arm_label="no_fim_ltr",
                    display_label=f"no-FIM LTR ({args.context_budget}L)",
                    study_name=args.study_name,
                    prompt_format="ltr_prefix",
                    model_label="no_fim",
                    prefix_length=args.context_budget,
                    suffix_length=0,
                ),
                ArmSpec(
                    arm_label="fim_v2_ltr",
                    display_label=f"FIM-v2 LTR ({args.context_budget}L)",
                    study_name=args.study_name,
                    prompt_format="ltr_prefix",
                    model_label="fim_v2",
                    prefix_length=args.context_budget,
                    suffix_length=0,
                ),
                ArmSpec(
                    arm_label="fineweb_only_ltr",
                    display_label=f"FineWeb-only LTR ({args.context_budget}L)",
                    study_name=args.study_name,
                    prompt_format="ltr_prefix",
                    model_label="fineweb_only",
                    prefix_length=args.context_budget,
                    suffix_length=0,
                ),
            ]
        )

    for prefix, suffix in parse_native_splits(args.native_splits, args.context_budget):
        arms.append(
            ArmSpec(
                arm_label=f"fim_v2_native_p{prefix}_s{suffix}",
                display_label=f"FIM-v2 native ({prefix}L/{suffix}R)",
                study_name=args.study_name,
                prompt_format="fim_native",
                model_label="fim_v2",
                prefix_length=prefix,
                suffix_length=suffix,
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


def arm_stem(args: argparse.Namespace, arm: ArmSpec) -> str:
    return output_stem(
        offset=args.offset,
        window_stride=args.window_stride,
        window_layout=args.window_layout,
        max_windows_per_excerpt=args.max_windows_per_excerpt,
        prefix_length=arm.prefix_length,
        middle_length=args.middle_length,
        suffix_length=arm.suffix_length,
    )


def comparison_stem(args: argparse.Namespace) -> str:
    if args.suite:
        return f"{args.suite}_{args.suite_report}"
    window_limit = "all" if args.max_windows_per_excerpt == 0 else str(args.max_windows_per_excerpt)
    layout_part = "" if args.window_layout == "matched_context" else f"_layout_{args.window_layout}"
    return (
        f"{args.study_name}_target_offset_{args.offset}_stride_{args.window_stride}_"
        f"windows_{window_limit}_middle_{args.middle_length}_context_{args.context_budget}{layout_part}"
    )


def summary_path(args: argparse.Namespace, arm: ArmSpec, repetition: int) -> Path:
    if args.suite:
        if arm.arm_id is None:
            raise ValueError("Suite arms must have arm_id")
        return arm_root(args.results_root, args.suite, arm.arm_id, repetition) / "windows.summary.json"
    return (
        args.results_root
        / arm.study_name
        / arm.prompt_format
        / arm.model_label
        / f"rep_{repetition}"
        / f"{arm_stem(args, arm)}.summary.json"
    )


def row_jsonl_path(args: argparse.Namespace, arm: ArmSpec, repetition: int) -> Path:
    if args.suite:
        if arm.arm_id is None:
            raise ValueError("Suite arms must have arm_id")
        return arm_root(args.results_root, args.suite, arm.arm_id, repetition) / "windows.jsonl"
    return (
        args.results_root
        / arm.study_name
        / arm.prompt_format
        / arm.model_label
        / f"rep_{repetition}"
        / f"{arm_stem(args, arm)}.jsonl"
    )


def metric_mean(summary: dict[str, Any], metric: str) -> float:
    if metric == "exact_match":
        return float(summary.get("match_rates", {}).get("exact_match", float("nan")))
    if metric in summary.get("match_rates", {}):
        return float(summary["match_rates"][metric])
    if metric in summary.get("metrics", {}):
        return float(summary["metrics"][metric]["mean"])
    return float("nan")


def metric_std(summary: dict[str, Any], metric: str) -> float:
    if metric in summary.get("metrics", {}):
        return float(summary["metrics"][metric].get("std", float("nan")))
    return float("nan")


def metric_count(summary: dict[str, Any], metric: str) -> int:
    value = metric_mean(summary, metric)
    if not math.isfinite(value):
        return 0
    return round(value * int(summary["num_windows"]))


def safe_log10(value: float) -> float:
    if value <= 0 or not math.isfinite(value):
        return float("nan")
    return math.log10(value)


def add_log10_token_geomean_fields(row: dict[str, Any]) -> None:
    mean = float(row.get("cooper_token_geomean_p_z", float("nan")))
    log_mean = safe_log10(mean)
    row["log10_cooper_token_geomean_p_z"] = log_mean
    if mean <= 0 or not math.isfinite(log_mean):
        row["log10_cooper_token_geomean_p_z_std"] = 0.0
        row["log10_cooper_token_geomean_p_z_ci95"] = 0.0
        return

    p_std = float(row.get("cooper_token_geomean_p_z_std", float("nan")))
    if math.isfinite(p_std):
        row["log10_cooper_token_geomean_p_z_std"] = abs(
            safe_log10(mean + p_std) - log_mean
        )
    else:
        row["log10_cooper_token_geomean_p_z_std"] = 0.0

    p_ci = float(row.get("cooper_token_geomean_p_z_ci95", float("nan")))
    if not math.isfinite(p_ci):
        p_ci = ci95_from_std(p_std, row.get("num_windows", 0))
    if math.isfinite(p_ci):
        row["log10_cooper_token_geomean_p_z_ci95"] = abs(
            safe_log10(mean + p_ci) - log_mean
        )
    else:
        row["log10_cooper_token_geomean_p_z_ci95"] = 0.0


def load_summaries(
    args: argparse.Namespace,
    arms: list[ArmSpec],
) -> dict[str, dict[int, dict[str, Any]]]:
    loaded: dict[str, dict[int, dict[str, Any]]] = {arm.arm_label: {} for arm in arms}
    missing: list[Path] = []
    for arm in arms:
        for repetition in args.repetitions:
            path = summary_path(args, arm, repetition)
            if not path.exists():
                missing.append(path)
                continue
            with path.open("r", encoding="utf-8") as handle:
                loaded[arm.arm_label][repetition] = json.load(handle)
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing summary files:\n{formatted}")
    return loaded


def per_arm_rows(
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
                "num_windows": int(summary["num_windows"]),
                "num_excerpts": int(summary["num_excerpts"]),
                "num_candidate_windows": int(summary["num_candidate_windows"]),
                "context_budget": int(summary["context_budget"]),
                "prefix_length": int(summary["prefix_length"]),
                "middle_length": int(summary["middle_length"]),
                "suffix_length": int(summary["suffix_length"]),
                "window_stride": int(summary["window_stride"]),
                "window_layout": summary.get("window_layout", "matched_context"),
                "max_windows_per_excerpt": int(summary["max_windows_per_excerpt"]),
                "sample_seed": int(summary["sample_seed"]),
                "prob_extraction_top_k": summary.get("prob_extraction_top_k"),
                "prob_extraction_temperature": summary.get("prob_extraction_temperature"),
                "prob_extraction_threshold": summary.get("prob_extraction_threshold"),
                "fim_split_mode": summary.get("fim_split_mode", "fixed_by_excerpt"),
            }
            for metric in core_metrics_for_args(args):
                row[metric] = metric_mean(summary, metric)
                row[f"{metric}_std"] = metric_std(summary, metric)
            row["cooper_extractable_count"] = metric_count(summary, "cooper_extractable")
            row["greedy_exact_match_count"] = metric_count(summary, "greedy_exact_match")
            row["cooper_extractable_per_10k"] = (
                row["cooper_extractable_count"] / row["num_windows"] * 10_000
            )
            row["greedy_exact_match_per_10k"] = (
                row["greedy_exact_match_count"] / row["num_windows"] * 10_000
            )
            row["match_ge_0_50_count"] = metric_count(summary, "match_ge_0_50")
            row["match_ge_0_50"] = row["match_ge_0_50_count"] / row["num_windows"]
            row["match_ge_0_50_std"] = metric_std(summary, "match_ge_0_50")
            row["match_ge_0_50_per_10k"] = (
                row["match_ge_0_50_count"] / row["num_windows"] * 10_000
            )
            row["cooper_supported_token_rate_pct"] = row["cooper_supported_token_rate"] * 100
            add_log10_token_geomean_fields(row)
            for span_length in cooper_prefix_target_lengths(args):
                metric = f"cooper_extractable_first{span_length}"
                count_key = f"{metric}_count"
                row[count_key] = metric_count(summary, metric)
                row[f"{metric}_per_10k"] = row[count_key] / row["num_windows"] * 10_000
            rows.append(row)
    return rows


def weighted_overall(
    args: argparse.Namespace,
    arms: list[ArmSpec],
    summaries: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    overall: dict[str, dict[str, float]] = {}
    for arm in arms:
        total_windows = sum(
            int(summaries[arm.arm_label][rep]["num_windows"]) for rep in args.repetitions
        )
        row: dict[str, float] = {"num_windows": float(total_windows)}
        for metric in core_metrics_for_args(args):
            mean = sum(
                metric_mean(summaries[arm.arm_label][rep], metric)
                * int(summaries[arm.arm_label][rep]["num_windows"])
                for rep in args.repetitions
            ) / total_windows
            row[metric] = mean
            if total_windows > 1 and math.isfinite(mean):
                m2 = 0.0
                for rep in args.repetitions:
                    summary = summaries[arm.arm_label][rep]
                    n = int(summary["num_windows"])
                    rep_mean = metric_mean(summary, metric)
                    rep_std = metric_std(summary, metric)
                    if n <= 0 or not math.isfinite(rep_mean):
                        continue
                    if n > 1 and math.isfinite(rep_std):
                        m2 += (n - 1) * rep_std * rep_std
                    m2 += n * (rep_mean - mean) ** 2
                std = math.sqrt(m2 / (total_windows - 1))
                row[f"{metric}_std"] = std
                row[f"{metric}_ci95"] = ci95_from_std(std, total_windows)
            else:
                row[f"{metric}_std"] = float("nan")
                row[f"{metric}_ci95"] = 0.0
        for metric in count_metrics_for_args(args):
            count = sum(metric_count(summaries[arm.arm_label][rep], metric) for rep in args.repetitions)
            row[f"{metric}_count"] = float(count)
            row[f"{metric}_rate"] = count / total_windows
        row["cooper_extractable_per_10k"] = row["cooper_extractable_rate"] * 10_000
        row["greedy_exact_match_per_10k"] = row["greedy_exact_match_rate"] * 10_000
        row["match_ge_0_50_per_10k"] = row["match_ge_0_50_rate"] * 10_000
        for span_length in cooper_prefix_target_lengths(args):
            metric = f"cooper_extractable_first{span_length}"
            row[f"{metric}_per_10k"] = row.get(f"{metric}_rate", float("nan")) * 10_000
        row["cooper_supported_token_rate_pct"] = row["cooper_supported_token_rate"] * 100
        row["cooper_supported_token_rate_pct_std"] = row["cooper_supported_token_rate_std"] * 100
        row["cooper_supported_token_rate_pct_ci95"] = row["cooper_supported_token_rate_ci95"] * 100
        add_log10_token_geomean_fields(row)
        overall[arm.arm_label] = row
    return overall


def window_key(row: dict[str, Any]) -> tuple[str, int, int, int, int]:
    return (
        str(row["excerpt_id"]),
        int(row["sample_index"]),
        int(row["global_window_id"]),
        int(row["window_index"]),
        int(row["target_start"]),
    )


def load_window_rows(
    args: argparse.Namespace,
    arm: ArmSpec,
    repetition: int,
) -> dict[tuple[str, int, int, int, int], dict[str, float]]:
    path = row_jsonl_path(args, arm, repetition)
    if not path.exists():
        raise FileNotFoundError(f"Missing per-window JSONL: {path}")
    rows: dict[tuple[str, int, int, int, int], dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            values: dict[str, float] = {}
            for field in window_float_fields_for_args(args):
                if field in raw:
                    values[field] = float(raw[field])
            rows[window_key(raw)] = values
    return rows


def validate_grid(args: argparse.Namespace, arms: list[ArmSpec]) -> dict[str, Any]:
    if args.no_validate_window_grid:
        return {"validated": False, "reason": "--no-validate-window-grid"}

    baseline = arms[0]
    report: dict[str, Any] = {"validated": True, "baseline_arm": baseline.arm_label, "repetitions": {}}
    for repetition in args.repetitions:
        baseline_rows = load_window_rows(args, baseline, repetition)
        baseline_keys = list(baseline_rows)
        rep_report = {"num_windows": len(baseline_keys), "arms": {}}
        for arm in arms:
            rows = load_window_rows(args, arm, repetition)
            matches = list(rows) == baseline_keys
            rep_report["arms"][arm.arm_label] = matches
            if not matches:
                raise ValueError(
                    f"Window grid mismatch for arm={arm.arm_label} rep={repetition}"
                )
        report["repetitions"][str(repetition)] = rep_report
    return report


def summarize_pair_rows(
    pair_label: str,
    comparison_label: str,
    baseline_label: str,
    repetition: str | int,
    paired: Iterable[tuple[dict[str, float], dict[str, float]]],
) -> dict[str, Any]:
    count = 0
    comp_extract = 0
    base_extract = 0
    shared_extract = 0
    comp_only_extract = 0
    base_only_extract = 0
    neither_extract = 0
    comp_greedy = 0
    base_greedy = 0
    comp_higher_geo = 0
    base_higher_geo = 0
    tied_geo = 0
    comp_lower_nll = 0
    base_lower_nll = 0
    tied_nll = 0
    deltas: dict[str, list[float]] = {
        "cooper_token_geomean_p_z": [],
        "log10_cooper_token_geomean_p_z": [],
        "cooper_supported_token_rate": [],
        "Ref_NLL": [],
        "memorization_score": [],
        "Rouge-L": [],
        "token_accuracy": [],
    }

    for comp, base in paired:
        count += 1
        comp_is_extract = comp.get("cooper_extractable", 0.0) >= 0.5
        base_is_extract = base.get("cooper_extractable", 0.0) >= 0.5
        comp_extract += int(comp_is_extract)
        base_extract += int(base_is_extract)
        shared_extract += int(comp_is_extract and base_is_extract)
        comp_only_extract += int(comp_is_extract and not base_is_extract)
        base_only_extract += int(base_is_extract and not comp_is_extract)
        neither_extract += int(not comp_is_extract and not base_is_extract)
        comp_greedy += int(comp.get("greedy_exact_match", 0.0) >= 0.5)
        base_greedy += int(base.get("greedy_exact_match", 0.0) >= 0.5)

        comp_geo = comp.get("cooper_token_geomean_p_z", float("nan"))
        base_geo = base.get("cooper_token_geomean_p_z", float("nan"))
        if comp_geo > base_geo:
            comp_higher_geo += 1
        elif base_geo > comp_geo:
            base_higher_geo += 1
        else:
            tied_geo += 1

        comp_nll = comp.get("Ref_NLL", float("nan"))
        base_nll = base.get("Ref_NLL", float("nan"))
        if comp_nll < base_nll:
            comp_lower_nll += 1
        elif base_nll < comp_nll:
            base_lower_nll += 1
        else:
            tied_nll += 1

        for metric in [
            "cooper_token_geomean_p_z",
            "cooper_supported_token_rate",
            "Ref_NLL",
            "memorization_score",
            "Rouge-L",
            "token_accuracy",
        ]:
            comp_value = comp.get(metric, float("nan"))
            base_value = base.get(metric, float("nan"))
            if math.isfinite(comp_value) and math.isfinite(base_value):
                deltas[metric].append(comp_value - base_value)
        if comp_geo > 0 and base_geo > 0:
            deltas["log10_cooper_token_geomean_p_z"].append(
                math.log10(comp_geo) - math.log10(base_geo)
            )

    row: dict[str, Any] = {
        "pair_label": pair_label,
        "comparison_arm": comparison_label,
        "baseline_arm": baseline_label,
        "repetition": repetition,
        "num_windows": count,
        "comparison_extractable_count": comp_extract,
        "baseline_extractable_count": base_extract,
        "delta_extractable_count": comp_extract - base_extract,
        "comparison_extractable_rate": comp_extract / count if count else float("nan"),
        "baseline_extractable_rate": base_extract / count if count else float("nan"),
        "delta_extractable_per_10k": (comp_extract - base_extract) / count * 10_000
        if count
        else float("nan"),
        "shared_extractable_count": shared_extract,
        "comparison_only_extractable_count": comp_only_extract,
        "baseline_only_extractable_count": base_only_extract,
        "neither_extractable_count": neither_extract,
        "comparison_greedy_exact_count": comp_greedy,
        "baseline_greedy_exact_count": base_greedy,
        "delta_greedy_exact_count": comp_greedy - base_greedy,
        "comparison_higher_geomean_fraction": comp_higher_geo / count if count else float("nan"),
        "baseline_higher_geomean_fraction": base_higher_geo / count if count else float("nan"),
        "tied_geomean_fraction": tied_geo / count if count else float("nan"),
        "comparison_lower_ref_nll_fraction": comp_lower_nll / count if count else float("nan"),
        "baseline_lower_ref_nll_fraction": base_lower_nll / count if count else float("nan"),
        "tied_ref_nll_fraction": tied_nll / count if count else float("nan"),
    }
    for metric, values in deltas.items():
        row[f"mean_delta_{metric}"] = statistics.fmean(values) if values else float("nan")
        row[f"median_delta_{metric}"] = statistics.median(values) if values else float("nan")
    return row


def native_arms(arms: list[ArmSpec]) -> list[ArmSpec]:
    return [arm for arm in arms if arm.prompt_format == "fim_native"]


def native_baseline_arm(arms: list[ArmSpec]) -> ArmSpec | None:
    native = native_arms(arms)
    if not native:
        return None
    return max(native, key=lambda arm: (arm.prefix_length, -arm.suffix_length))


def paired_native_rows(args: argparse.Namespace, arms: list[ArmSpec]) -> list[dict[str, Any]]:
    native = native_arms(arms)
    baseline = native_baseline_arm(arms)
    if baseline is None or len(native) <= 1:
        return []

    rows: list[dict[str, Any]] = []
    all_pairs_by_arm: dict[str, list[tuple[dict[str, float], dict[str, float]]]] = {
        arm.arm_label: [] for arm in native
    }
    for repetition in args.repetitions:
        baseline_rows = load_window_rows(args, baseline, repetition)
        for arm in native:
            comp_rows = load_window_rows(args, arm, repetition)
            if list(comp_rows) != list(baseline_rows):
                raise ValueError(
                    f"Native window grid mismatch for arm={arm.arm_label} rep={repetition}"
                )
            pairs = [(comp_rows[key], baseline_rows[key]) for key in comp_rows]
            rows.append(
                summarize_pair_rows(
                    f"{arm.arm_label}_minus_{baseline.arm_label}",
                    arm.arm_label,
                    baseline.arm_label,
                    repetition,
                    pairs,
                )
                | {
                    "comparison_split": arm.split_label,
                    "baseline_split": baseline.split_label,
                    "comparison_prefix_length": arm.prefix_length,
                    "comparison_suffix_length": arm.suffix_length,
                }
            )
            all_pairs_by_arm[arm.arm_label].extend(pairs)

    for arm in native:
        rows.append(
            summarize_pair_rows(
                f"{arm.arm_label}_minus_{baseline.arm_label}",
                arm.arm_label,
                baseline.arm_label,
                "all",
                all_pairs_by_arm[arm.arm_label],
            )
            | {
                "comparison_split": arm.split_label,
                "baseline_split": baseline.split_label,
                "comparison_prefix_length": arm.prefix_length,
                "comparison_suffix_length": arm.suffix_length,
            }
        )
    return rows


def best_native_split_rows(args: argparse.Namespace, arms: list[ArmSpec]) -> list[dict[str, Any]]:
    native = native_arms(arms)
    if len(native) <= 1:
        return []

    counts_by_rep: list[dict[str, Any]] = []
    total_best_geo = {arm.arm_label: 0 for arm in native}
    total_best_nll = {arm.arm_label: 0 for arm in native}
    total_extractable = {arm.arm_label: 0 for arm in native}
    total_windows = 0

    for repetition in args.repetitions:
        rows_by_arm = {arm.arm_label: load_window_rows(args, arm, repetition) for arm in native}
        keys = list(rows_by_arm[native[0].arm_label])
        for arm in native[1:]:
            if list(rows_by_arm[arm.arm_label]) != keys:
                raise ValueError(f"Native split grid mismatch at repetition {repetition}")

        rep_best_geo = {arm.arm_label: 0 for arm in native}
        rep_best_nll = {arm.arm_label: 0 for arm in native}
        rep_extractable = {arm.arm_label: 0 for arm in native}
        for key in keys:
            best_geo_arm = max(
                native,
                key=lambda arm: rows_by_arm[arm.arm_label][key].get(
                    "cooper_token_geomean_p_z", float("-inf")
                ),
            )
            best_nll_arm = min(
                native,
                key=lambda arm: rows_by_arm[arm.arm_label][key].get("Ref_NLL", float("inf")),
            )
            rep_best_geo[best_geo_arm.arm_label] += 1
            rep_best_nll[best_nll_arm.arm_label] += 1
            for arm in native:
                rep_extractable[arm.arm_label] += int(
                    rows_by_arm[arm.arm_label][key].get("cooper_extractable", 0.0) >= 0.5
                )

        total_windows += len(keys)
        for arm in native:
            total_best_geo[arm.arm_label] += rep_best_geo[arm.arm_label]
            total_best_nll[arm.arm_label] += rep_best_nll[arm.arm_label]
            total_extractable[arm.arm_label] += rep_extractable[arm.arm_label]
            counts_by_rep.append(
                {
                    "repetition": repetition,
                    "arm_label": arm.arm_label,
                    "split": arm.split_label,
                    "prefix_length": arm.prefix_length,
                    "suffix_length": arm.suffix_length,
                    "num_windows": len(keys),
                    "best_geomean_count": rep_best_geo[arm.arm_label],
                    "best_geomean_fraction": rep_best_geo[arm.arm_label] / len(keys),
                    "best_ref_nll_count": rep_best_nll[arm.arm_label],
                    "best_ref_nll_fraction": rep_best_nll[arm.arm_label] / len(keys),
                    "extractable_count": rep_extractable[arm.arm_label],
                    "extractable_per_10k": rep_extractable[arm.arm_label] / len(keys) * 10_000,
                }
            )

    for arm in native:
        counts_by_rep.append(
            {
                "repetition": "all",
                "arm_label": arm.arm_label,
                "split": arm.split_label,
                "prefix_length": arm.prefix_length,
                "suffix_length": arm.suffix_length,
                "num_windows": total_windows,
                "best_geomean_count": total_best_geo[arm.arm_label],
                "best_geomean_fraction": total_best_geo[arm.arm_label] / total_windows,
                "best_ref_nll_count": total_best_nll[arm.arm_label],
                "best_ref_nll_fraction": total_best_nll[arm.arm_label] / total_windows,
                "extractable_count": total_extractable[arm.arm_label],
                "extractable_per_10k": total_extractable[arm.arm_label] / total_windows * 10_000,
            }
        )
    return counts_by_rep


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def per_10k(value: float) -> str:
    return f"{value:.3f}/10k"


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
    png_path = figure_dir / f"{stem}.png"
    pdf_path = figure_dir / f"{stem}.pdf"
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    return {"png": str(png_path), "pdf": str(pdf_path)}


def arm_color(label: str) -> str:
    colors = {
        "no_fim_ltr": "#0072B2",
        "fim_v2_ltr": "#CC79A7",
        "fineweb_only_ltr": "#009E73",
        "fim_v2_native_p100_s0": "#1B9E77",
        "fim_v2_native_p80_s20": "#4C78A8",
        "fim_v2_native_p60_s40": "#59A14F",
        "fim_v2_native_p40_s60": "#F28E2B",
        "fim_v2_native_p20_s80": "#E15759",
        "fim_v2_native_p0_s100": "#B07AA1",
    }
    return colors.get(label, "#555555")


def arm_short_label(label: str) -> str:
    labels = {
        "no_fim_ltr": "no-FIM",
        "fim_v2_ltr": "FIM-v2",
        "fineweb_only_ltr": "FineWeb-only",
    }
    return labels.get(label, label.replace("_ltr", "").replace("_", "-"))


def ltr_arm_order(rows: list[dict[str, Any]]) -> list[str]:
    available = {str(row["arm_label"]) for row in rows}
    return [
        label
        for label in ["no_fim_ltr", "fim_v2_ltr", "fineweb_only_ltr"]
        if label in available
    ]


def display_label_for_arm(rows: list[dict[str, Any]], arm_label: str) -> str:
    for row in rows:
        if str(row["arm_label"]) == arm_label:
            return str(row["display_label"])
    return arm_label


def row_by_arm_rep(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["arm_label"]), int(row["repetition"])): row for row in rows}


def binomial_ci95(count: float, total: float, scale: float = 10_000.0) -> float:
    if total <= 0:
        return 0.0
    rate = count / total
    return 1.96 * math.sqrt(rate * (1.0 - rate) / total) * scale


def rate_metric_ci95_pct(row: dict[str, Any], metric: str) -> float:
    spread = metric_ci95_from_row(row, metric) * 100.0
    if math.isfinite(spread) and spread > 0.0:
        return spread
    count = float(row.get(f"{metric}_count", 0.0))
    total = float(row.get("num_windows", 0.0))
    return binomial_ci95(count, total, scale=100.0)


def target_ppl_rows(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    overall: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    arm_order = ltr_arm_order(rows)
    if not arm_order:
        return []
    lookup = row_by_arm_rep(rows)
    table_rows: list[dict[str, Any]] = []
    for repetition in args.repetitions:
        row: dict[str, Any] = {"repetition": repetition}
        for arm_label in arm_order:
            arm_row = lookup[(arm_label, repetition)]
            row[f"{arm_label}_target_ppl"] = math.exp(float(arm_row["Ref_NLL"]))
            row[f"{arm_label}_ref_nll"] = float(arm_row["Ref_NLL"])
        table_rows.append(row)

    overall_row: dict[str, Any] = {"repetition": "all"}
    for arm_label in arm_order:
        overall_row[f"{arm_label}_target_ppl"] = math.exp(float(overall[arm_label]["Ref_NLL"]))
        overall_row[f"{arm_label}_ref_nll"] = float(overall[arm_label]["Ref_NLL"])
    table_rows.append(overall_row)
    return table_rows


def plot_target_ppl_table(
    table_rows: list[dict[str, Any]],
    figure_dir: Path,
) -> dict[str, dict[str, str]]:
    if not table_rows:
        return {}

    arm_order = [
        label
        for label in ["no_fim_ltr", "fim_v2_ltr", "fineweb_only_ltr"]
        if f"{label}_target_ppl" in table_rows[0]
    ]
    if not arm_order:
        return {}

    plt = ensure_plotting()
    columns = ["Repetition", *(arm_short_label(label) for label in arm_order)]
    cell_text = []
    for row in table_rows:
        values = [str(row["repetition"])]
        values.extend(f"{float(row[f'{label}_target_ppl']):.2f}" for label in arm_order)
        cell_text.append(values)

    fig_height = max(2.8, 0.24 * len(cell_text) + 0.75)
    fig, ax = plt.subplots(figsize=(5.4, fig_height), constrained_layout=True)
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.12)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        cell.set_edgecolor("#B8B8B8")
        if row_idx == 0:
            cell.set_facecolor("#F1F3F5")
            cell.set_text_props(weight="bold")
        elif row_idx == len(cell_text):
            cell.set_facecolor("#F8F9FA")
            cell.set_text_props(weight="bold")
    ax.set_title("Target perplexity by repetition", pad=8, fontweight="bold")
    return {"target_ppl_table": save_figure(plt, figure_dir, "target_ppl_table")}


def plot_ltr_token_geomean(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> dict[str, dict[str, str]]:
    arm_order = ltr_arm_order(rows)
    if len(arm_order) < 2:
        return {}

    plt = ensure_plotting()
    from matplotlib.lines import Line2D

    lookup = row_by_arm_rep(rows)
    x_values = list(args.repetitions)
    metric = "cooper_token_geomean_p_z"
    all_values: list[float] = []
    all_spreads: list[float] = []

    fig, ax = plt.subplots(figsize=(4.8, 3.15))
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.18, top=0.82)
    for arm_label in arm_order:
        values: list[float] = []
        spreads: list[float] = []
        for repetition in x_values:
            row = lookup[(arm_label, repetition)]
            value = float(row.get(metric, float("nan")))
            spread = float(row.get(f"{metric}_ci95", float("nan")))
            if not math.isfinite(spread):
                spread = metric_ci95_from_row(row, metric)
            if not math.isfinite(value):
                value = float("nan")
                spread = 0.0
            values.append(value)
            spreads.append(spread if math.isfinite(spread) else 0.0)
        lower, upper = interval_band(values, spreads, lower_floor=0.0, upper_ceiling=1.0)
        all_values.extend(values)
        all_spreads.extend(spreads)
        color = arm_color(arm_label)
        ax.fill_between(
            x_values,
            lower,
            upper,
            color=color,
            alpha=0.18,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            x_values,
            values,
            color=color,
            linewidth=2.15,
            label=arm_short_label(arm_label),
            solid_capstyle="round",
            zorder=3,
        )

    set_repetition_axis(ax, x_values)
    ylim = finite_ylim(all_values, all_spreads, lower_floor=0.0, pad=0.16)
    if ylim is not None:
        ax.set_ylim(ylim[0], min(1.0, ylim[1]))
    ax.set_xlabel("Training repetitions")
    ax.set_ylabel(r"Mean token-geomean $p_z$")
    ax.set_title("Length-normalized Cooper score")
    ax.grid(True, axis="y", alpha=0.16)
    ax.tick_params(length=2.5, width=0.7)
    ax.tick_params(axis="x", which="minor", length=1.8, width=0.55)

    handles = [
        Line2D(
            [0],
            [0],
            color=arm_color(label),
            linewidth=2.2,
            label=arm_short_label(label),
        )
        for label in arm_order
    ]
    band_handle = Line2D(
        [0],
        [0],
        color="#555555",
        linewidth=6.0,
        alpha=0.18,
        label="95% CI",
    )
    ax.legend(
        handles=[*handles, band_handle],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.20),
        ncol=2,
        handlelength=2.1,
    )

    artifacts = save_figure(plt, figure_dir, "ltr_token_geomean")
    caption_path = figure_dir / "ltr_token_geomean.caption.txt"
    caption_path.write_text(
        (
            r"Length-normalized Cooper memorization under LTR-prefix probes. "
            r"Token-geomean \(p_z\) is the geometric mean of the "
            r"top-40-renormalized probabilities assigned to the ground-truth "
            r"target tokens, normalizing exact-span probability by target length. "
            r"Lines show mean token-geomean \(p_z\) by repetition on a linear "
            r"scale; bands are nominal 95\% confidence intervals over windows."
            "\n"
        ),
        encoding="utf-8",
    )
    artifacts["caption"] = str(caption_path)
    return {"ltr_token_geomean": artifacts}


def plot_ltr_cooper_span_extractability(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> dict[str, dict[str, str]]:
    arm_order = ltr_arm_order(rows)
    if len(arm_order) < 2:
        return {}
    spans = [
        span_length
        for span_length in cooper_prefix_target_lengths(args)
        if any(
            math.isfinite(float(row.get(f"cooper_extractable_first{span_length}", float("nan"))))
            for row in rows
        )
    ]
    if not spans:
        return {}

    plt = ensure_plotting()
    from matplotlib.lines import Line2D

    lookup = row_by_arm_rep(rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 4.9), constrained_layout=True)
    axes_flat = list(axes.ravel())
    x_values = list(args.repetitions)
    threshold = next(
        (
            float(row.get("prob_extraction_threshold"))
            for row in rows
            if row.get("prob_extraction_threshold") is not None
        ),
        0.001,
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    caption_path = figure_dir / "ltr_cooper_span_extractability.caption.txt"
    max_y = 0.0

    if len(spans) >= 4:
        curve_arm_order = [label for label in arm_order if label != "fineweb_only_ltr"] or arm_order
        curve_spans = spans[:4]
        reference_span = 50 if 50 in curve_spans else curve_spans[-1]
        reference_values: list[float] = []
        for arm_label in curve_arm_order:
            reference_metric = f"cooper_extractable_first{reference_span}"
            for repetition in x_values:
                row = lookup[(arm_label, repetition)]
                value = float(row.get(reference_metric, float("nan"))) * 100.0
                spread = rate_metric_ci95_pct(row, reference_metric)
                if math.isfinite(value):
                    reference_values.append(value + (spread if math.isfinite(spread) else 0.0))
        if not reference_values:
            return {}

        y_upper = padded_linear_upper(reference_values)
        fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.55), constrained_layout=True, sharey=True)
        axes_flat = list(axes.ravel())
        for ax, span_length in zip(axes_flat, curve_spans, strict=False):
            metric = f"cooper_extractable_first{span_length}"
            for arm_label in curve_arm_order:
                series_rows = [lookup[(arm_label, repetition)] for repetition in x_values]
                y_values = [
                    float(row.get(metric, float("nan"))) * 100.0
                    for row in series_rows
                ]
                yerr = [rate_metric_ci95_pct(row, metric) for row in series_rows]
                lower, upper = interval_band(y_values, yerr, lower_floor=0.0, upper_ceiling=100.0)
                color = arm_color(arm_label)
                ax.fill_between(
                    x_values,
                    lower,
                    upper,
                    color=color,
                    alpha=0.16,
                    linewidth=0.0,
                    zorder=1,
                )
                ax.plot(
                    x_values,
                    y_values,
                    color=color,
                    linewidth=2.1,
                    label=arm_short_label(arm_label),
                    solid_capstyle="round",
                    zorder=3,
                )
            set_repetition_axis(ax, x_values)
            ax.text(
                0.03,
                0.93,
                f"{span_length} tokens",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11.0,
                color="#333333",
            )
            ax.set_xlabel("Training repetitions")
            if ax is axes_flat[0]:
                ax.set_ylabel("Extractable windows (%)")
            else:
                ax.set_ylabel("")
            ax.set_ylim(0.0, y_upper)
            ax.grid(True, axis="y", alpha=0.16)

        caption = (
            "Cooper extractability by target-prefix length. Curves show the percentage "
            f"of windows with $p_z \\geq {threshold:g}$ for the first 20, 30, 40, or "
            f"50 target tokens; all panels use the {reference_span}-token y-axis scale. "
            r"Shaded bands denote nominal 95\% confidence intervals for the per-window "
            r"extractability rate in each repetition bucket."
        )
        caption_path.write_text(caption + "\n", encoding="utf-8")
        return {
            "ltr_cooper_span_extractability": {
                **save_figure(plt, figure_dir, "ltr_cooper_span_extractability"),
                "caption": str(caption_path),
            }
        }

    for ax, span_length in zip(axes_flat, spans, strict=False):
        metric = f"cooper_extractable_first{span_length}"
        for arm_label in arm_order:
            series_rows = [lookup[(arm_label, repetition)] for repetition in x_values]
            y_values = [
                float(row.get(metric, float("nan"))) * 100.0
                for row in series_rows
            ]
            yerr = [rate_metric_ci95_pct(row, metric) for row in series_rows]
            lower, upper = interval_band(y_values, yerr, lower_floor=0.0, upper_ceiling=100.0)
            finite_upper = [value for value in upper if math.isfinite(value)]
            max_y = max(max_y, *finite_upper, 0.0)
            color = arm_color(arm_label)
            ax.fill_between(
                x_values,
                lower,
                upper,
                color=color,
                alpha=0.16,
                linewidth=0.0,
                zorder=1,
            )
            ax.plot(
                x_values,
                y_values,
                color=color,
                linewidth=2.1,
                label=arm_short_label(arm_label),
                solid_capstyle="round",
                zorder=3,
            )
        set_repetition_axis(ax, x_values)
        ax.set_title(f"First {span_length} target tokens")
        ax.text(
            0.02,
            0.94,
            rf"$p_z^{{({span_length})}} \geq {threshold:g}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            color="#444444",
        )
        ax.set_xlabel("Training repetitions")
        ax.set_ylabel("Extractable windows (%)")
        ax.grid(True, axis="y", alpha=0.16)

    for ax in axes_flat[len(spans):]:
        ax.set_visible(False)
    y_upper = padded_linear_upper([max_y])
    for ax in axes_flat[: len(spans)]:
        ax.set_ylim(0.0, y_upper)

    handles = [
        Line2D([0], [0], color=arm_color(label), linewidth=2.1, label=arm_short_label(label))
        for label in arm_order
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=len(handles),
        handlelength=2.2,
    )
    caption = (
        "LTR Cooper span extractability by repetition. Each panel shows the percentage of "
        "evaluated windows whose first target tokens meet the exact-target Cooper "
        f"extractability criterion, $p_z \\geq {threshold:g}$, for the indicated span length. "
        r"Shaded bands denote nominal 95\% confidence intervals for the per-window "
        r"extractability rate in each repetition bucket."
    )
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return {
        "ltr_cooper_span_extractability": {
            **save_figure(plt, figure_dir, "ltr_cooper_span_extractability"),
            "caption": str(caption_path),
        }
    }


def is_target_length_sensitivity_report(args: argparse.Namespace) -> bool:
    labels = [
        str(getattr(args, "suite", "") or ""),
        str(getattr(args, "study_name", "") or ""),
        str(getattr(args, "output_prefix", "") or ""),
    ]
    return any("target_length_sensitivity" in label for label in labels)


def ltr_density_label(arm_label: str) -> str:
    return {
        "no_fim_ltr": "LTR-trained",
        "fim_v2_ltr": "FIM-trained",
    }.get(arm_label, arm_short_label(arm_label))


def ltr_window_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["excerpt_id"]),
        int(row["target_start"]),
        int(row["middle_length"]),
    )


def read_ltr_windows_for_density(
    args: argparse.Namespace,
    arm: ArmSpec,
    repetition: int,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    if not args.suite or arm.arm_id is None:
        return {}
    jsonl_path = arm_root(args.results_root, args.suite, arm.arm_id, repetition) / "windows.jsonl"
    if not jsonl_path.exists():
        return {}
    windows: dict[tuple[str, int, int], dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("experiment") != "ltr" or row.get("prompt_format") != "ltr_prefix":
                continue
            windows[ltr_window_key(row)] = row
    return windows


def is_cooper_memorized(row: dict[str, Any]) -> bool:
    try:
        return float(row.get("cooper_extractable", 0.0)) >= 0.5
    except (TypeError, ValueError):
        return False


def density_bins(values: Iterable[float], *, bins: int = 28, fixed_range: tuple[float, float] | None = None) -> list[float]:
    if fixed_range is not None:
        lo, hi = fixed_range
        step = (hi - lo) / bins
        return [lo + step * index for index in range(bins + 1)]
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return []
    lo = finite[0]
    hi = finite[-1]
    if lo == hi:
        pad = max(0.01, abs(lo) * 0.05)
        lo -= pad
        hi += pad
    else:
        pad = (hi - lo) * 0.06
        lo -= pad
        hi += pad
    step = (hi - lo) / bins
    return [lo + step * index for index in range(bins + 1)]


def plot_density_panel(
    ax: Any,
    datasets: dict[str, list[float]],
    window_counts: dict[str, int],
    *,
    title: str,
    subtitle: str,
    xlabel: str,
) -> None:
    all_values = [value for values in datasets.values() for value in values]
    bins = density_bins(all_values, fixed_range=(0.0, 1.0))
    if not bins:
        ax.set_visible(False)
        return
    count_text = "; ".join(
        f"{ltr_density_label(arm_label)} n={window_counts.get(arm_label, len(values))}"
        for arm_label, values in datasets.items()
        if values
    )
    for arm_label, values in datasets.items():
        if not values:
            continue
        color = arm_color(arm_label)
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            color=color,
            alpha=0.16,
            linewidth=0.0,
            zorder=1,
        )
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            color=color,
            linewidth=1.75,
            zorder=3,
        )
        median = statistics.median(values)
        ax.axvline(
            median,
            color=color,
            linestyle=(0, (4, 2)),
            linewidth=1.15,
            alpha=0.9,
            zorder=2,
        )

    ax.set_title(title, fontsize=9.5, pad=6)
    ax.text(
        0.02,
        0.94,
        f"{subtitle}\n{count_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="#444444",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(xlabel, fontsize=9.0)
    ax.set_ylabel("Density", fontsize=9.0)
    ax.grid(True, axis="y", alpha=0.16)
    upper = ax.get_ylim()[1]
    if math.isfinite(upper) and upper > 0.0:
        ax.set_ylim(0.0, upper * 1.08)


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def logspace(start_exp: float, stop_exp: float, points: int) -> list[float]:
    if points <= 1:
        return [10 ** start_exp]
    step = (stop_exp - start_exp) / (points - 1)
    return [10 ** (start_exp + step * index) for index in range(points)]


def survival_fraction(values: list[float], threshold: float) -> tuple[int, float]:
    if not values:
        return 0, float("nan")
    positives = sorted(value for value in values if value > 0.0)
    count = len(positives) - bisect.bisect_left(positives, threshold)
    return count, count / len(values)


def plot_ltr_cooper_pz_survival(
    args: argparse.Namespace,
    arms: list[ArmSpec],
    figure_dir: Path,
) -> dict[str, dict[str, str]]:
    if not args.suite or args.suite_report != "ltr":
        return {}
    arm_order = [
        label
        for label in ["no_fim_ltr", "fim_v2_ltr", "fineweb_only_ltr"]
        if any(arm.arm_label == label for arm in arms)
    ]
    if len(arm_order) < 2:
        return {}
    arm_by_label = {arm.arm_label: arm for arm in arms}

    repetition = (
        int(args.scatter_repetition)
        if args.scatter_repetition is not None
        else max(int(rep) for rep in args.repetitions)
    )
    values_by_arm: dict[str, list[float]] = {}
    threshold = 0.001
    for arm_label in arm_order:
        windows = read_ltr_windows_for_density(args, arm_by_label[arm_label], repetition)
        values: list[float] = []
        for row in windows.values():
            cooper_p_z = finite_float(row.get("cooper_p_z"))
            if cooper_p_z is None or cooper_p_z < 0.0:
                continue
            values.append(cooper_p_z)
            parsed_threshold = finite_float(row.get("prob_extraction_threshold"))
            if parsed_threshold is not None and parsed_threshold > 0.0:
                threshold = parsed_threshold
        if values:
            values_by_arm[arm_label] = values
    if len(values_by_arm) < 2:
        return {}

    x_min = 1e-10
    x_max = 1.0
    thresholds = sorted(set([*logspace(-10, 0, 251), threshold]))
    csv_rows: list[dict[str, Any]] = []
    survival_by_arm: dict[str, list[float]] = {}
    survival_ci_by_arm: dict[str, list[float]] = {}
    counts_by_arm: dict[str, list[int]] = {}
    for arm_label, values in values_by_arm.items():
        survival_values: list[float] = []
        ci_values: list[float] = []
        count_values: list[int] = []
        total_windows = len(values)
        for current_threshold in thresholds:
            count, fraction = survival_fraction(values, current_threshold)
            ci95 = binomial_ci95(count, total_windows, scale=100.0)
            survival_values.append(fraction * 100.0)
            ci_values.append(ci95)
            count_values.append(count)
            csv_rows.append(
                {
                    "repetition": repetition,
                    "arm_label": arm_label,
                    "model": ltr_density_label(arm_label),
                    "threshold": current_threshold,
                    "survival_count": count,
                    "survival_fraction": fraction,
                    "survival_fraction_ci95": ci95 / 100.0,
                    "survival_percent": fraction * 100.0,
                    "survival_percent_ci95": ci95,
                    "total_windows": total_windows,
                }
            )
        survival_by_arm[arm_label] = survival_values
        survival_ci_by_arm[arm_label] = ci_values
        counts_by_arm[arm_label] = count_values

    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_path = figure_dir / "ltr_cooper_pz_survival.csv"
    caption_path = figure_dir / "ltr_cooper_pz_survival.caption.txt"
    write_csv(csv_path, csv_rows)

    plt = ensure_plotting()
    fig, ax = plt.subplots(figsize=(6.15, 3.65))
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.20, top=0.97)
    for arm_label in arm_order:
        if arm_label not in survival_by_arm:
            continue
        lower, upper = interval_band(
            survival_by_arm[arm_label],
            survival_ci_by_arm[arm_label],
            lower_floor=0.0,
            upper_ceiling=100.0,
        )
        ax.fill_between(
            thresholds,
            lower,
            upper,
            color=arm_color(arm_label),
            alpha=0.24,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            thresholds,
            lower,
            color=arm_color(arm_label),
            linewidth=0.65,
            alpha=0.45,
            zorder=2,
        )
        ax.plot(
            thresholds,
            upper,
            color=arm_color(arm_label),
            linewidth=0.65,
            alpha=0.45,
            zorder=2,
        )
        ax.plot(
            thresholds,
            survival_by_arm[arm_label],
            color=arm_color(arm_label),
            linewidth=2.45,
            solid_capstyle="round",
            zorder=3,
        )
    ax.axvline(
        threshold,
        color="#333333",
        linewidth=1.2,
        linestyle=(0, (4, 2)),
        alpha=0.85,
    )
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    visible_upper_values = [
        min(100.0, value + spread)
        for arm_label, arm_values in survival_by_arm.items()
        for current_threshold, value, spread in zip(
            thresholds,
            arm_values,
            survival_ci_by_arm[arm_label],
            strict=True,
        )
        if x_min <= current_threshold <= x_max
    ]
    y_upper = min(100.0, max(1.0, max(visible_upper_values, default=0.0) * 1.12))
    ax.set_ylim(0.0, y_upper)
    ax.set_xlabel(r"Extraction threshold $t$", fontsize=12.6)
    ax.set_ylabel(r"Windows with $p_z \geq t$ (%)", fontsize=12.6)
    ax.grid(True, axis="both", alpha=0.15)
    ax.tick_params(length=3.0, width=0.8, labelsize=11.2)
    threshold_label = "t=0.001" if math.isclose(threshold, 0.001, rel_tol=0.0, abs_tol=1e-12) else f"t={threshold:g}"
    ax.text(
        threshold,
        ax.get_ylim()[1] * 0.90,
        threshold_label,
        rotation=90,
        ha="center",
        va="top",
        fontsize=12.2,
        fontweight="bold",
        color="#333333",
        clip_on=False,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.8},
    )

    caption = (
        f"LTR survival curves at repetition {repetition}. Each line shows the percentage "
        "of evaluated 32-token target windows with Cooper exact-target probability "
        rf"$p_z \geq t$ as the extraction threshold $t$ varies over "
        rf"{x_min:g} to {x_max:g}. The vertical dashed line marks the standard "
        rf"extractability cutoff, $t={threshold:g}$. Shaded bands denote nominal "
        r"95\% confidence intervals for the per-window survival rate at each threshold."
    )
    caption_path.write_text(caption + "\n", encoding="utf-8")

    return {
        "ltr_cooper_pz_survival": {
            **save_figure(plt, figure_dir, "ltr_cooper_pz_survival"),
            "csv": str(csv_path),
            "caption": str(caption_path),
        }
    }


def density_torch_dtype(name: str) -> Any:
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def resolve_density_device(value: str) -> Any:
    import torch

    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {value}, but CUDA is not available")
    return torch.device(value)


def load_density_samples(dataset_path: Path, excerpt_ids: set[str]) -> dict[str, list[int]]:
    samples: dict[str, list[int]] = {}
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            excerpt_id = str(obj["excerpt_id"])
            if excerpt_id not in excerpt_ids:
                continue
            samples[excerpt_id] = [int(token_id) for token_id in obj["input_ids"]]
            if len(samples) == len(excerpt_ids):
                break
    missing = excerpt_ids - set(samples)
    if missing:
        raise RuntimeError(f"{dataset_path} is missing {len(missing)} selected excerpts: {sorted(missing)[:5]}")
    return samples


def density_summary_for_arm(args: argparse.Namespace, arm: ArmSpec, repetition: int) -> dict[str, Any]:
    if not args.suite or arm.arm_id is None:
        raise ValueError("Token density recomputation requires suite arms")
    summary_path = arm_root(args.results_root, args.suite, arm.arm_id, repetition) / "windows.summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary for token density recomputation: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def window_prefix_target(row: dict[str, Any], token_ids: list[int]) -> tuple[list[int], list[int]]:
    target_start = int(row["target_start"])
    prefix_length = int(row["prefix_length"])
    middle_length = int(row["middle_length"])
    prefix_start = target_start - prefix_length
    middle_end = target_start + middle_length
    if prefix_start < 0 or middle_end > len(token_ids):
        raise ValueError(f"Window extends outside token sequence for {row['excerpt_id']}")
    return token_ids[prefix_start:target_start], token_ids[target_start:middle_end]


def recompute_token_topk_probabilities(
    args: argparse.Namespace,
    model_path: str,
    dataset_path: Path,
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int, int], list[float]]:
    if not rows:
        return {}

    import torch
    from transformers import AutoModelForCausalLM

    from direct_overlap_eval import target_only_shift_logits

    device = resolve_density_device(args.density_device)
    excerpt_ids = {str(row["excerpt_id"]) for row in rows}
    samples = load_density_samples(dataset_path, excerpt_ids)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=density_torch_dtype(args.density_dtype),
        trust_remote_code=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    probabilities: dict[tuple[str, int, int], list[float]] = {}
    try:
        for batch in iter_batches(rows, args.density_batch_size):
            prefixes: list[list[int]] = []
            targets: list[list[int]] = []
            for row in batch:
                prefix, target = window_prefix_target(row, samples[str(row["excerpt_id"])])
                prefixes.append(prefix)
                targets.append(target)

            prompt_lengths = {len(prefix) for prefix in prefixes}
            target_lengths = {len(target) for target in targets}
            if len(prompt_lengths) != 1 or len(target_lengths) != 1:
                raise ValueError("Token-density recomputation requires equal prompt and target lengths per batch")
            prompt_length = next(iter(prompt_lengths))
            target_length = next(iter(target_lengths))
            input_ids = torch.tensor(
                [prefix + target for prefix, target in zip(prefixes, targets, strict=True)],
                dtype=torch.long,
                device=device,
            )
            target_labels = torch.tensor(targets, dtype=torch.long, device=device)
            positions = (
                torch.arange(target_length, dtype=torch.long, device=device).unsqueeze(0)
                + prompt_length
                - 1
            ).expand(input_ids.shape[0], -1)

            with torch.inference_mode():
                target_logits = target_only_shift_logits(model, input_ids, positions)
                if target_logits is None:
                    logits = model(input_ids=input_ids, use_cache=False).logits
                    rows_index = torch.arange(input_ids.shape[0], dtype=torch.long, device=device).unsqueeze(1)
                    target_logits = logits[:, :-1, :].float()[rows_index, positions, :]
                scaled_logits = target_logits / float(batch[0].get("prob_extraction_temperature", 1.0))
                top_k = min(int(batch[0].get("prob_extraction_top_k", 40)), scaled_logits.shape[-1])
                top_values, top_indices = torch.topk(scaled_logits, k=top_k, dim=-1)
                top_probs = torch.softmax(top_values, dim=-1)
                matches = top_indices.eq(target_labels.unsqueeze(-1))
                ranks = matches.to(torch.long).argmax(dim=-1)
                token_probs = top_probs.gather(-1, ranks.unsqueeze(-1)).squeeze(-1)
                token_probs = torch.where(matches.any(dim=-1), token_probs, torch.zeros_like(token_probs))

            for row, values in zip(batch, token_probs.detach().cpu().tolist(), strict=True):
                probabilities[ltr_window_key(row)] = [float(value) for value in values]
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return probabilities


def write_token_density_csv(
    csv_path: Path,
    token_values_by_panel: dict[str, dict[str, dict[tuple[str, int, int], list[float]]]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "panel",
        "arm_label",
        "excerpt_id",
        "target_start",
        "middle_length",
        "token_index",
        "top40_renormalized_prob",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for panel, arm_values in token_values_by_panel.items():
            for arm_label, keyed_values in arm_values.items():
                for key, values in keyed_values.items():
                    excerpt_id, target_start, middle_length = key
                    for token_index, value in enumerate(values):
                        writer.writerow(
                            {
                                "panel": panel,
                                "arm_label": arm_label,
                                "excerpt_id": excerpt_id,
                                "target_start": target_start,
                                "middle_length": middle_length,
                                "token_index": token_index,
                                "top40_renormalized_prob": value,
                            }
                        )


def write_window_token_stats_csv(
    csv_path: Path,
    stats_by_panel: dict[str, dict[str, dict[tuple[str, int, int], dict[str, float]]]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "panel",
        "arm_label",
        "excerpt_id",
        "target_start",
        "middle_length",
        "min_top40_renormalized_prob",
        "median_top40_renormalized_prob",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for panel, arm_values in stats_by_panel.items():
            for arm_label, keyed_values in arm_values.items():
                for key, stats in keyed_values.items():
                    excerpt_id, target_start, middle_length = key
                    writer.writerow(
                        {
                            "panel": panel,
                            "arm_label": arm_label,
                            "excerpt_id": excerpt_id,
                            "target_start": target_start,
                            "middle_length": middle_length,
                            "min_top40_renormalized_prob": stats["min"],
                            "median_top40_renormalized_prob": stats["median"],
                        }
                    )


def keyed_window_token_stats(
    keyed_token_probs: dict[tuple[str, int, int], list[float]],
    keys: Iterable[tuple[str, int, int]],
) -> dict[tuple[str, int, int], dict[str, float]]:
    values: dict[tuple[str, int, int], dict[str, float]] = {}
    for key in keys:
        token_probs = keyed_token_probs.get(key)
        if token_probs:
            values[key] = {
                "min": min(token_probs),
                "median": statistics.median(token_probs),
            }
    return values


def plot_ltr_memorized_min_tokenprob_density(
    args: argparse.Namespace,
    arms: list[ArmSpec],
    figure_dir: Path,
) -> dict[str, dict[str, str]]:
    if not args.suite or args.skip_token_density_figure:
        return {}
    arm_by_label = {arm.arm_label: arm for arm in arms}
    target_labels = ["no_fim_ltr", "fim_v2_ltr"]
    if any(label not in arm_by_label for label in target_labels):
        return {}

    repetition = int(args.density_repetition) if args.density_repetition is not None else max(int(rep) for rep in args.repetitions)
    windows_by_arm = {
        label: read_ltr_windows_for_density(args, arm_by_label[label], repetition)
        for label in target_labels
    }
    if any(not windows_by_arm[label] for label in target_labels):
        return {}

    summaries_by_arm = {
        label: density_summary_for_arm(args, arm_by_label[label], repetition)
        for label in target_labels
    }
    ltr_label, fim_label = target_labels
    common_keys = set(windows_by_arm[target_labels[0]]).intersection(windows_by_arm[target_labels[1]])
    ltr_only_keys = sorted(
        key
        for key in common_keys
        if is_cooper_memorized(windows_by_arm[ltr_label][key])
        and not is_cooper_memorized(windows_by_arm[fim_label][key])
    )
    fim_only_keys = sorted(
        key
        for key in common_keys
        if is_cooper_memorized(windows_by_arm[fim_label][key])
        and not is_cooper_memorized(windows_by_arm[ltr_label][key])
    )
    shared_keys = sorted(
        key
        for key in common_keys
        if all(is_cooper_memorized(windows_by_arm[label][key]) for label in target_labels)
    )
    scored_keys = sorted(set(ltr_only_keys).union(fim_only_keys).union(shared_keys))
    if not scored_keys:
        return {}

    scored_rows = {
        label: [windows_by_arm[label][key] for key in scored_keys]
        for label in target_labels
    }
    token_probs_by_arm: dict[str, dict[tuple[str, int, int], list[float]]] = {}
    for label in target_labels:
        summary = summaries_by_arm[label]
        token_probs_by_arm[label] = recompute_token_topk_probabilities(
            args=args,
            model_path=str(summary["model_path"]),
            dataset_path=Path(summary["dataset"]),
            rows=scored_rows[label],
        )

    ltr_only_stats = {
        label: keyed_window_token_stats(token_probs_by_arm[label], ltr_only_keys)
        for label in target_labels
    }
    fim_only_stats = {
        label: keyed_window_token_stats(token_probs_by_arm[label], fim_only_keys)
        for label in target_labels
    }
    shared_stats = {
        label: keyed_window_token_stats(token_probs_by_arm[label], shared_keys)
        for label in target_labels
    }
    ltr_only_min_values = {
        label: [stats["min"] for stats in ltr_only_stats[label].values()]
        for label in target_labels
    }
    fim_only_min_values = {
        label: [stats["min"] for stats in fim_only_stats[label].values()]
        for label in target_labels
    }
    shared_min_values = {
        label: [stats["min"] for stats in shared_stats[label].values()]
        for label in target_labels
    }
    ltr_only_median_values = {
        label: [stats["median"] for stats in ltr_only_stats[label].values()]
        for label in target_labels
    }
    fim_only_median_values = {
        label: [stats["median"] for stats in fim_only_stats[label].values()]
        for label in target_labels
    }
    shared_median_values = {
        label: [stats["median"] for stats in shared_stats[label].values()]
        for label in target_labels
    }
    ltr_only_window_counts = {label: len(ltr_only_stats[label]) for label in target_labels}
    fim_only_window_counts = {label: len(fim_only_stats[label]) for label in target_labels}
    shared_window_counts = {label: len(shared_stats[label]) for label in target_labels}

    threshold = next(
        (
            float(row.get("prob_extraction_threshold"))
            for values in windows_by_arm.values()
            for row in values.values()
            if row.get("prob_extraction_threshold") is not None
        ),
        0.001,
    )
    middle_length = next(
        (
            int(row.get("middle_length"))
            for values in windows_by_arm.values()
            for row in values.values()
            if row.get("middle_length") is not None
        ),
        args.middle_length,
    )

    plt = ensure_plotting()
    from matplotlib.lines import Line2D

    figure_dir.mkdir(parents=True, exist_ok=True)
    token_csv_path = figure_dir / "ltr_memorized_min_tokenprob_density.tokens.csv"
    min_csv_path = figure_dir / "ltr_memorized_min_tokenprob_density.windows.csv"
    write_token_density_csv(
        token_csv_path,
        {
            "ltr_only": {
                label: {key: values for key, values in token_probs_by_arm[label].items() if key in set(ltr_only_keys)}
                for label in target_labels
            },
            "fim_only": {
                label: {key: values for key, values in token_probs_by_arm[label].items() if key in set(fim_only_keys)}
                for label in target_labels
            },
            "shared_extractable": {
                label: {key: values for key, values in token_probs_by_arm[label].items() if key in set(shared_keys)}
                for label in target_labels
            },
        },
    )
    write_window_token_stats_csv(
        min_csv_path,
        {
            "ltr_only": ltr_only_stats,
            "fim_only": fim_only_stats,
            "shared_extractable": shared_stats,
        },
    )

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.25))
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.12, top=0.84, wspace=0.26, hspace=0.42)
    subtitle = rf"rep. {repetition}; {middle_length}-token target; Cooper $p_z \geq {threshold:g}$"
    plot_density_panel(
        axes[0, 0],
        ltr_only_min_values,
        ltr_only_window_counts,
        title="Min token: LTR-only extractable",
        subtitle=subtitle,
        xlabel=r"Minimum token $q_i$ (top-40 renorm.)",
    )
    plot_density_panel(
        axes[0, 1],
        fim_only_min_values,
        fim_only_window_counts,
        title="Min token: FIM-only extractable",
        subtitle=subtitle,
        xlabel=r"Minimum token $q_i$ (top-40 renorm.)",
    )
    plot_density_panel(
        axes[0, 2],
        shared_min_values,
        shared_window_counts,
        title="Min token: both extractable",
        subtitle=subtitle,
        xlabel=r"Minimum token $q_i$ (top-40 renorm.)",
    )
    plot_density_panel(
        axes[1, 0],
        ltr_only_median_values,
        ltr_only_window_counts,
        title="Median token: LTR-only extractable",
        subtitle=subtitle,
        xlabel=r"Median token $q_i$ (top-40 renorm.)",
    )
    plot_density_panel(
        axes[1, 1],
        fim_only_median_values,
        fim_only_window_counts,
        title="Median token: FIM-only extractable",
        subtitle=subtitle,
        xlabel=r"Median token $q_i$ (top-40 renorm.)",
    )
    plot_density_panel(
        axes[1, 2],
        shared_median_values,
        shared_window_counts,
        title="Median token: both extractable",
        subtitle=subtitle,
        xlabel=r"Median token $q_i$ (top-40 renorm.)",
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=arm_color(label),
            linewidth=2.0,
            label=ltr_density_label(label),
        )
        for label in target_labels
    ]
    median_handle = Line2D(
        [0],
        [0],
        color="#555555",
        linestyle=(0, (4, 2)),
        linewidth=1.15,
        label="median",
    )
    fig.legend(
        handles=[*handles, median_handle],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        handlelength=2.3,
    )
    return {
        "ltr_memorized_min_tokenprob_density": {
            **save_figure(plt, figure_dir, "ltr_memorized_min_tokenprob_density"),
            "token_csv": str(token_csv_path),
            "window_csv": str(min_csv_path),
        }
    }


def sorted_native_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if str(row["prompt_format"]) == "fim_native"],
        key=lambda row: -int(row["prefix_length"]),
    )


def unique_native_labels(native_rows: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for row in native_rows:
        label = str(row["arm_label"])
        if label in seen:
            continue
        labels.append(label)
        seen.add(label)
    return labels


def split_xtick_labels(prefix_lengths: list[int], suffix_lengths: list[int]) -> list[str]:
    return [
        f"{prefix}/{suffix}" for prefix, suffix in zip(prefix_lengths, suffix_lengths, strict=True)
    ]


def set_split_xticks(
    ax: Any,
    positions: list[int],
    prefix_lengths: list[int],
    suffix_lengths: list[int],
    *,
    labelsize: float,
    rotation: float = 35.0,
    omit_prefixes: set[int] | None = None,
) -> None:
    omitted = omit_prefixes or set()
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            "" if prefix in omitted else f"{prefix}/{suffix}"
            for prefix, suffix in zip(prefix_lengths, suffix_lengths, strict=True)
        ],
        rotation=rotation,
        ha="right",
        rotation_mode="anchor",
        color="black",
        fontsize=labelsize,
    )
    ax.tick_params(axis="x", colors="black")


def plot_native_profile(
    rows: list[dict[str, Any]],
    overall: dict[str, dict[str, float]],
    figure_dir: Path,
) -> dict[str, dict[str, str]]:
    native_rows = sorted_native_from_rows(rows)
    if not native_rows:
        return {}

    representative = {str(row["arm_label"]): row for row in native_rows}
    unique_labels = unique_native_labels(native_rows)
    ordered = sorted(
        [
            (
                int(representative[label]["prefix_length"]),
                int(representative[label]["suffix_length"]),
                label,
            )
            for label in unique_labels
        ],
        key=lambda item: item[0],
    )
    prefix_lengths = [prefix for prefix, _, _ in ordered]
    suffix_lengths = [suffix for _, suffix, _ in ordered]
    unique_labels = [label for _, _, label in ordered]
    x = prefix_lengths
    panels = [
        ("cooper_extractable_per_10k", "Cooper extractability", "Windows per 10k"),
        ("cooper_supported_token_rate_pct", "Target tokens in top-k", "Percent"),
        ("log10_cooper_token_geomean_p_z", "Token geomean p(z)", "log10 probability"),
        ("Ref_NLL", "Teacher-forced target NLL", "NLL"),
    ]

    plt = ensure_plotting()
    curve_color = "#111111"
    panel_letters = ["a", "b", "c", "d"]

    def draw_panel(
        ax: Any,
        metric: str,
        title: str | None,
        ylabel: str,
        letter: str | None,
        tick_size: float,
    ) -> None:
        y_values = [float(overall[label][metric]) for label in unique_labels]
        if metric == "cooper_extractable_per_10k":
            yerr = [
                binomial_ci95(
                    float(overall[label].get("cooper_extractable_count", 0.0)),
                    float(overall[label].get("num_windows", 0.0)),
                    scale=10_000.0,
                )
                for label in unique_labels
            ]
        else:
            yerr = [float(overall[label].get(f"{metric}_ci95", 0.0)) for label in unique_labels]
        lower, upper = interval_band(
            y_values,
            yerr,
            lower_floor=0.0
            if metric in {"cooper_extractable_per_10k", "cooper_supported_token_rate_pct"}
            else None,
        )
        ax.fill_between(
            x,
            lower,
            upper,
            color=curve_color,
            alpha=0.11,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            x,
            y_values,
            color=curve_color,
            linewidth=1.8,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            x,
            y_values,
            color=curve_color,
            edgecolor="white",
            linewidth=0.55,
            s=28,
            zorder=3,
        )
        lower_floor = 0.0 if metric in {"cooper_extractable_per_10k", "cooper_supported_token_rate_pct"} else None
        ylim = finite_ylim(y_values, yerr, lower_floor=lower_floor, pad=0.14)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if title:
            ax.set_title(title, fontsize=10.5)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-3.0, 103.0)
        set_split_xticks(
            ax,
            x,
            prefix_lengths,
            suffix_lengths,
            labelsize=tick_size,
            omit_prefixes={1, 5},
        )
        ax.tick_params(length=2.5, width=0.7)
        if letter:
            ax.text(
                0.015,
                0.965,
                letter,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11.0,
                fontweight="bold",
            )
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.7)

    artifacts: dict[str, dict[str, str]] = {}
    fig, axes = plt.subplots(2, 2, figsize=(8.9, 5.9))
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.135, top=0.955, hspace=0.55, wspace=0.3)
    for ax, (metric, title, ylabel), letter in zip(axes.ravel(), panels, panel_letters):
        draw_panel(ax, metric, title, ylabel, letter, tick_size=7.0)
    artifacts["native_geometry_profile"] = save_figure(plt, figure_dir, "native_geometry_profile")

    panel_stems = [
        "native_geometry_profile_a",
        "native_geometry_profile_b",
        "native_geometry_profile_c",
        "native_geometry_profile_d",
    ]
    for (metric, title, ylabel), letter, stem in zip(panels, panel_letters, panel_stems):
        fig, ax = plt.subplots(figsize=(4.25, 3.1))
        fig.subplots_adjust(left=0.18, right=0.985, bottom=0.24, top=0.985)
        draw_panel(ax, metric, None, ylabel, None, tick_size=9.4)
        artifacts[stem] = save_figure(plt, figure_dir, stem)

    caption_path = figure_dir / "native_geometry_profile.caption.txt"
    caption_path.write_text(
        (
            r"Native FIM geometry profile. Panels show (a) Cooper extractability, "
            r"(b) target-token top-$k$ support, (c) token-geomean Cooper probability, "
            r"and (d) teacher-forced NLL, weighted over repetition buckets. "
            r"The x-axis is linear in prefix length and tick labels show prefix/suffix tokens; "
            r"bands are nominal 95\% confidence intervals."
            "\n"
        ),
        encoding="utf-8",
    )
    artifacts["native_geometry_profile"]["caption"] = str(caption_path)
    panel_captions = {
        "native_geometry_profile_a": "Cooper extractability.",
        "native_geometry_profile_b": r"Target tokens in top-$k$.",
        "native_geometry_profile_c": "Token-geomean Cooper probability.",
        "native_geometry_profile_d": "Teacher-forced target NLL.",
    }
    for stem, caption in panel_captions.items():
        path = figure_dir / f"{stem}.caption.txt"
        path.write_text(caption + "\n", encoding="utf-8")
        artifacts[stem]["caption"] = str(path)
    return artifacts


def matrix_from_rows(
    rows: list[dict[str, Any]],
    metric: str,
) -> tuple[list[int], list[int], list[int], list[list[float]]]:
    native_rows = sorted_native_from_rows(rows)
    arm_order = unique_native_labels(native_rows)
    representative = {str(row["arm_label"]): row for row in native_rows}
    prefix_lengths = [int(representative[arm]["prefix_length"]) for arm in arm_order]
    suffix_lengths = [int(representative[arm]["suffix_length"]) for arm in arm_order]
    reps = sorted({int(row["repetition"]) for row in native_rows})
    lookup = {(str(row["arm_label"]), int(row["repetition"])): row for row in native_rows}
    matrix = [[float(lookup[(arm, rep)][metric]) for arm in arm_order] for rep in reps]
    return reps, prefix_lengths, suffix_lengths, matrix


def plot_native_heatmaps(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> dict[str, dict[str, str]]:
    if not sorted_native_from_rows(rows):
        return {}

    plt = ensure_plotting()
    omitted_prefixes = {1, 5, 10}
    panels = [
        (
            "log10_cooper_token_geomean_p_z",
            "log10 probability",
            "viridis",
            "a",
        ),
        (
            "cooper_supported_token_rate_pct",
            "Percent",
            "magma",
            "b",
        ),
    ]

    def draw_heatmap(
        ax: Any,
        fig: Any,
        metric: str,
        cbar_label: str,
        cmap: str,
        tick_size: float,
    ) -> None:
        reps, prefix_lengths, suffix_lengths, matrix = matrix_from_rows(rows, metric)
        kept_indices = [
            index for index, prefix in enumerate(prefix_lengths) if prefix not in omitted_prefixes
        ]
        prefix_lengths = [prefix_lengths[index] for index in kept_indices]
        suffix_lengths = [suffix_lengths[index] for index in kept_indices]
        matrix = [[row[index] for index in kept_indices] for row in matrix]
        image = ax.imshow(matrix, aspect="auto", cmap=cmap)
        set_split_xticks(
            ax,
            list(range(len(prefix_lengths))),
            prefix_lengths,
            suffix_lengths,
            labelsize=tick_size,
        )
        ax.set_yticks(range(len(reps)))
        ax.set_yticklabels([str(rep) for rep in reps])
        ax.set_ylabel("Training repetitions")
        ax.tick_params(length=2.5, width=0.7)
        cbar = fig.colorbar(image, ax=ax, pad=0.02)
        cbar.set_label(cbar_label)

    artifacts: dict[str, dict[str, str]] = {}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.35))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.985, wspace=0.28)
    for ax, (metric, cbar_label, cmap, letter) in zip(axes, panels, strict=True):
        draw_heatmap(ax, fig, metric, cbar_label, cmap, tick_size=8.0)
    artifacts["native_geometry_heatmaps"] = save_figure(plt, figure_dir, "native_geometry_heatmaps")

    panel_stems = ["native_geometry_heatmaps_a", "native_geometry_heatmaps_b"]
    for (metric, cbar_label, cmap, letter), stem in zip(panels, panel_stems, strict=True):
        fig, ax = plt.subplots(figsize=(4.8, 3.65))
        fig.subplots_adjust(left=0.135, right=0.91, bottom=0.18, top=0.985)
        draw_heatmap(ax, fig, metric, cbar_label, cmap, tick_size=9.4)
        artifacts[stem] = save_figure(plt, figure_dir, stem)

    caption_path = figure_dir / "native_geometry_heatmaps.caption.txt"
    caption_path.write_text(
        (
            r"Native FIM geometry across repetition buckets. Panel (a) shows token-geomean "
            r"Cooper probability; panel (b) shows target-token top-$k$ support. "
            r"Tick labels show prefix/suffix tokens; "
            r"near-LTR splits with 1, 5, or 10 prefix tokens are omitted."
            "\n"
        ),
        encoding="utf-8",
    )
    artifacts["native_geometry_heatmaps"]["caption"] = str(caption_path)

    panel_captions = {
        "native_geometry_heatmaps_a": (
            r"Token-geomean Cooper probability by repetition bucket and native FIM split. "
            r"Tick labels show prefix/suffix tokens; 1, 5, and 10 prefix-token splits are omitted."
        ),
        "native_geometry_heatmaps_b": (
            r"Target-token top-$k$ support by repetition bucket and native FIM split. "
            r"Tick labels show prefix/suffix tokens; 1, 5, and 10 prefix-token splits are omitted."
        ),
    }
    for stem, caption in panel_captions.items():
        path = figure_dir / f"{stem}.caption.txt"
        path.write_text(caption + "\n", encoding="utf-8")
        artifacts[stem]["caption"] = str(path)
    return artifacts


def plot_native_topk_support_profile(
    rows: list[dict[str, Any]],
    overall: dict[str, dict[str, float]],
    figure_dir: Path,
) -> dict[str, dict[str, str]]:
    native_rows = sorted_native_from_rows(rows)
    if not native_rows:
        return {}

    unique_labels = unique_native_labels(native_rows)
    representative = {str(row["arm_label"]): row for row in native_rows}
    ordered = sorted(
        [
            (
                int(representative[label]["prefix_length"]),
                int(representative[label]["suffix_length"]),
                label,
            )
            for label in unique_labels
        ],
        key=lambda item: item[0],
    )
    prefix_lengths = [prefix for prefix, _, _ in ordered]
    suffix_lengths = [suffix for _, suffix, _ in ordered]
    y_values = [
        float(overall[label]["cooper_supported_token_rate_pct"])
        for _, _, label in ordered
    ]
    yerr = [
        float(overall[label].get("cooper_supported_token_rate_pct_ci95", 0.0))
        for _, _, label in ordered
    ]
    lower, upper = interval_band(y_values, yerr, lower_floor=0.0, upper_ceiling=100.0)

    plt = ensure_plotting()
    curve_color = "#111111"

    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.25, top=0.96)
    ax.fill_between(
        prefix_lengths,
        lower,
        upper,
        color=curve_color,
        alpha=0.12,
        linewidth=0.0,
        zorder=1,
    )
    ax.plot(
        prefix_lengths,
        y_values,
        color=curve_color,
        linewidth=2.15,
        solid_capstyle="round",
        zorder=3,
    )
    ax.scatter(
        prefix_lengths,
        y_values,
        color=curve_color,
        edgecolor="white",
        linewidth=0.65,
        s=42,
        zorder=4,
    )
    ylim = finite_ylim(y_values, yerr, lower_floor=0.0, pad=0.16)
    if ylim is not None:
        ax.set_ylim(ylim[0], min(100.0, ylim[1]))
    ax.set_xlim(-3.0, 103.0)
    set_split_xticks(
        ax,
        prefix_lengths,
        prefix_lengths,
        suffix_lengths,
        labelsize=10.2,
        omit_prefixes={1, 5},
    )
    ax.tick_params(length=2.5, width=0.7)
    ax.set_ylabel("Target tokens in top-k (%)")
    ax.grid(True, axis="y", alpha=0.16)

    artifacts = save_figure(plt, figure_dir, "native_topk_support_profile")
    caption_path = figure_dir / "native_topk_support_profile.caption.txt"
    caption_path.write_text(
        (
            r"Target-token top-$k$ support under native FIM geometry. The x-axis is linear in prefix length; "
            r"tick labels show prefix/suffix tokens. "
            r"The line shows the percentage of target tokens assigned nonzero top-$40$ Cooper support "
            r"averaged over all repetition buckets; the shaded band is a nominal 95\% confidence interval over windows."
            "\n"
        ),
        encoding="utf-8",
    )
    artifacts["caption"] = str(caption_path)
    return {"native_topk_support_profile": artifacts}


def plot_native_topk_support_heatmap_linear(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> dict[str, dict[str, str]]:
    native_rows = sorted_native_from_rows(rows)
    if not native_rows:
        return {}

    unique_labels = unique_native_labels(native_rows)
    representative = {str(row["arm_label"]): row for row in native_rows}
    ordered = sorted(
        [
            (
                int(representative[label]["prefix_length"]),
                int(representative[label]["suffix_length"]),
                label,
            )
            for label in unique_labels
        ],
        key=lambda item: item[0],
    )
    prefix_lengths = [prefix for prefix, _, _ in ordered]
    suffix_lengths = [suffix for _, suffix, _ in ordered]
    arm_order = [label for _, _, label in ordered]
    reps = sorted({int(row["repetition"]) for row in native_rows})
    lookup = {(str(row["arm_label"]), int(row["repetition"])): row for row in native_rows}
    matrix = [
        [
            float(lookup[(arm_label, repetition)]["cooper_supported_token_rate_pct"])
            for arm_label in arm_order
        ]
        for repetition in reps
    ]

    plt = ensure_plotting()
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    fig.subplots_adjust(left=0.105, right=0.91, bottom=0.25, top=0.96)
    flat_values = [value for row_values in matrix for value in row_values]
    norm = Normalize(vmin=min(flat_values), vmax=max(flat_values))
    cmap = plt.get_cmap("magma")
    xs = [prefix for prefix in prefix_lengths for _ in reps]
    ys = [index + 0.5 for _prefix in prefix_lengths for index in range(len(reps))]
    colors = [
        matrix[index][column]
        for column, _prefix in enumerate(prefix_lengths)
        for index in range(len(reps))
    ]
    image = ax.scatter(
        xs,
        ys,
        c=colors,
        cmap="magma",
        norm=norm,
        marker="s",
        s=150,
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlim(-3.0, 103.0)
    ax.set_ylim(0.0, len(reps))
    set_split_xticks(
        ax,
        prefix_lengths,
        prefix_lengths,
        suffix_lengths,
        labelsize=10.2,
        omit_prefixes={1, 5},
    )
    ax.tick_params(length=2.5, width=0.7)
    ax.set_yticks([index + 0.5 for index in range(len(reps))])
    ax.set_yticklabels([str(rep) for rep in reps])
    ax.set_ylabel("Training repetitions")
    ax.invert_yaxis()

    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("Target tokens in top-k (%)")

    artifacts = save_figure(plt, figure_dir, "native_topk_support_heatmap_linear")
    caption_path = figure_dir / "native_topk_support_heatmap_linear.caption.txt"
    caption_path.write_text(
        (
            r"Target-token top-$k$ support by repetition under native FIM geometry. The x-axis is linear "
            r"in prefix length; tick labels show prefix/suffix tokens. Color gives the percentage "
            r"of target tokens assigned nonzero top-$40$ Cooper support."
            "\n"
        ),
        encoding="utf-8",
    )
    artifacts["caption"] = str(caption_path)
    return {"native_topk_support_heatmap_linear": artifacts}


def plot_native_paired(
    pair_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    figure_dir: Path,
) -> dict[str, dict[str, str]]:
    overall_pairs = sorted(
        [row for row in pair_rows if row["repetition"] == "all"],
        key=lambda row: -int(row["comparison_prefix_length"]),
    )
    if not overall_pairs:
        return {}

    plt = ensure_plotting()
    x = list(range(len(overall_pairs)))
    labels = [str(row["comparison_split"]).replace("L/", "/").replace("R", "") for row in overall_pairs]
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 5.9), constrained_layout=True)
    for ax, metric, title, ylabel, zero in [
        (
            axes[0, 0],
            "mean_delta_log10_cooper_token_geomean_p_z",
            "Delta vs 100L/0R token geomean",
            "Mean delta log10 p",
            0.0,
        ),
        (
            axes[0, 1],
            "mean_delta_cooper_supported_token_rate",
            "Delta vs 100L/0R top-k support",
            "Mean delta rate",
            0.0,
        ),
        (
            axes[1, 0],
            "mean_delta_Ref_NLL",
            "Delta vs 100L/0R target NLL",
            "Mean delta NLL",
            0.0,
        ),
        (
            axes[1, 1],
            "comparison_higher_geomean_fraction",
            "Windows where split beats 100L/0R",
            "Fraction",
            0.5,
        ),
    ]:
        values = [float(row[metric]) for row in overall_pairs]
        colors = [arm_color(str(row["comparison_arm"])) for row in overall_pairs]
        ax.bar(x, values, color=colors)
        ax.axhline(zero, color="#333333", linewidth=0.9, linestyle=":")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35)
        ax.set_xlabel("Prefix/suffix split")
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    artifacts = {
        "native_vs_prefix100_paired_deltas": save_figure(
            plt, figure_dir, "native_vs_prefix100_paired_deltas"
        )
    }

    overall_best = sorted(
        [row for row in best_rows if row["repetition"] == "all"],
        key=lambda row: -int(row["prefix_length"]),
    )
    if overall_best:
        plt = ensure_plotting()
        fig, ax = plt.subplots(figsize=(7.1, 3.8), constrained_layout=True)
        x = list(range(len(overall_best)))
        labels = [str(row["split"]).replace("L/", "/").replace("R", "") for row in overall_best]
        best_geo = [float(row["best_geomean_fraction"]) for row in overall_best]
        best_nll = [float(row["best_ref_nll_fraction"]) for row in overall_best]
        width = 0.38
        ax.bar(
            [value - width / 2 for value in x],
            best_geo,
            width=width,
            color="#1B9E77",
            label="best geomean",
        )
        ax.bar(
            [value + width / 2 for value in x],
            best_nll,
            width=width,
            color="#B07AA1",
            label="best Ref NLL",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35)
        ax.set_xlabel("Prefix/suffix split")
        ax.set_ylabel("Fraction of windows")
        ax.set_title("Which native split wins same-window comparisons?")
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
        ax.legend(frameon=False)
        artifacts["native_best_split_share"] = save_figure(
            plt, figure_dir, "native_best_split_share"
        )
    return artifacts


def generate_insights(
    args: argparse.Namespace,
    arms: list[ArmSpec],
    overall: dict[str, dict[str, float]],
    native_pairs: list[dict[str, Any]],
    best_native: list[dict[str, Any]],
) -> list[str]:
    lines = [
        f"# {args.study_name}",
        "",
        "## Key Insights",
    ]

    ltr_labels = [
        label
        for label in ["no_fim_ltr", "fim_v2_ltr", "fineweb_only_ltr"]
        if label in overall
    ]
    if ltr_labels:
        lines.append(
            "- Experiment 1 reports all LTR arms directly on matched target windows: "
            "mean Cooper p_z is the main probabilistic memorization score. This is "
            "the exact-target product of top-k-renormalized ground-truth token "
            "probabilities over the target span; mean greedy ROUGE-L is the overlap "
            "diagnostic."
        )
        for label in ltr_labels:
            metrics = overall[label]
            lines.append(
                f"- {arm_short_label(label)}: mean Cooper p_z "
                f"{metrics['cooper_p_z']:.6g}; ROUGE-L "
                f"{metrics['Rouge-L']:.4f}; threshold extractability "
                f"{int(metrics['cooper_extractable_count'])}/{int(metrics['num_windows'])} "
                f"({per_10k(metrics['cooper_extractable_per_10k'])})."
            )

    native = native_arms(arms)
    if native:
        native_overall = [(arm, overall[arm.arm_label]) for arm in native]
        best_by_geo = max(native_overall, key=lambda item: item[1]["cooper_token_geomean_p_z"])
        best_by_support = max(native_overall, key=lambda item: item[1]["cooper_supported_token_rate"])
        best_by_nll = min(native_overall, key=lambda item: item[1]["Ref_NLL"])
        suffix_only = next((item for item in native_overall if item[0].prefix_length == 0), None)
        prefix_only = native_baseline_arm(arms)
        lines.append(
            "- Native FIM geometry is left-prefix dominated. The best split by "
            f"token-geomean p(z) is {best_by_geo[0].split_label}; by top-k support it is "
            f"{best_by_support[0].split_label}; by Ref NLL it is {best_by_nll[0].split_label}."
        )
        if prefix_only is not None and suffix_only is not None:
            prefix_metrics = overall[prefix_only.arm_label]
            suffix_metrics = suffix_only[1]
            suffix_support_delta = (
                suffix_metrics["cooper_supported_token_rate"]
                - prefix_metrics["cooper_supported_token_rate"]
            ) * 100
            lines.append(
                "- Suffix-only prompting is a different regime rather than a fair recall cue: "
                f"{suffix_only[0].split_label} has "
                f"{per_10k(suffix_metrics['cooper_extractable_per_10k'])} extractable windows, "
                f"versus {per_10k(prefix_metrics['cooper_extractable_per_10k'])} for "
                f"{prefix_only.split_label}; support drops by "
                f"{suffix_support_delta:+.2f} pp."
            )
        all_native_pairs = [row for row in native_pairs if row["repetition"] == "all"]
        if all_native_pairs and prefix_only is not None:
            nonbaseline = [row for row in all_native_pairs if row["comparison_arm"] != prefix_only.arm_label]
            closest_nonbaseline = min(
                nonbaseline,
                key=lambda row: abs(float(row["mean_delta_log10_cooper_token_geomean_p_z"])),
                default=None,
            )
            if closest_nonbaseline is not None:
                lines.append(
                    "- Adding right context does not recover more verbatim recall in this run: "
                    f"the closest non-baseline split, {closest_nonbaseline['comparison_split']}, "
                    f"still shifts mean log10 token-geomean p(z) by "
                    f"{float(closest_nonbaseline['mean_delta_log10_cooper_token_geomean_p_z']):+.4f} "
                    f"relative to {closest_nonbaseline['baseline_split']}."
                )
        all_best = [row for row in best_native if row["repetition"] == "all"]
        if all_best:
            geo_winner = max(all_best, key=lambda row: float(row["best_geomean_fraction"]))
            nll_winner = max(all_best, key=lambda row: float(row["best_ref_nll_fraction"]))
            lines.append(
                "- Same-window native split summaries reinforce the aggregate result: "
                f"{geo_winner['split']} has the highest token-geomean p(z) on "
                f"{pct(float(geo_winner['best_geomean_fraction']))} of evaluated "
                f"windows. Full-softmax Ref NLL is less one-sided: "
                f"{nll_winner['split']} wins on "
                f"{pct(float(nll_winner['best_ref_nll_fraction']))} of windows."
            )

    lines.extend(["", "## Reading Guide"])
    if ltr_labels:
        lines.extend(
            [
                "- For Experiment 1, read the LTR figures as direct arm-level curves, not "
                "pairwise deltas. The token-geomean panel shows the length-normalized "
                "Cooper score by repetition; translucent bands mark 95% confidence "
                "intervals at each observed repetition bucket.",
            ]
        )
    lines.extend(
        [
            "- The threshold extractability count is reported for continuity with "
            "Cooper-style extraction, where a window is extractable when exact-target "
            "p(z) >= 0.001 under top-k sampling with k=40 and T=1.",
            "- Target perplexity measures teacher-forced likelihood rather than "
            "extraction behavior.",
        ]
    )
    if native:
        lines.append(
            "- For native FIM, compare splits on the same target windows. Aggregate "
            "split rates alone can hide that suffix-heavy prompts are consistently "
            "lower-ranked on the identical windows."
        )
    return lines


def write_insights(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def should_generate_ltr_audit_figures(args: argparse.Namespace) -> bool:
    return bool(
        args.suite
        and args.suite_report == "ltr"
        and not args.skip_figures
        and not args.skip_audit_figures
    )


def ltr_audit_output_dir(args: argparse.Namespace) -> Path:
    if args.audit_output_dir is not None:
        return args.audit_output_dir
    if not args.suite:
        raise ValueError("LTR audit figures require --suite")
    return suite_root(args.results_root, args.suite) / "appendix_audits"


def build_ltr_audit_figure_command(args: argparse.Namespace) -> list[str]:
    if not args.suite:
        raise ValueError("LTR audit figures require --suite")
    script = Path(__file__).resolve().parent / "make_ltr_audit_figures.py"
    return [
        sys.executable,
        str(script),
        "--results-root",
        str(args.results_root),
        "--suite",
        str(args.suite),
        "--output-dir",
        str(ltr_audit_output_dir(args)),
        "--repetitions",
        str(args.audit_repetitions),
        "--examples-per-model",
        str(args.audit_examples_per_model),
        "--model-labels",
        str(args.audit_model_labels),
        "--device",
        str(args.audit_device),
    ]


def generate_ltr_audit_figures(args: argparse.Namespace) -> dict[str, str]:
    output_dir = ltr_audit_output_dir(args)
    cmd = build_ltr_audit_figure_command(args)
    print("Generating LTR audit figures:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return {
        "audit_dir": str(output_dir),
        "audit_examples_csv": str(output_dir / "audit_examples.csv"),
        "audit_token_logits_csv": str(output_dir / "audit_token_logits.csv"),
        "audit_word_logits_csv": str(output_dir / "audit_word_logits.csv"),
        "audit_index_markdown": str(output_dir / "audit_examples.md"),
        "audit_figure_dir": str(output_dir / "figures"),
    }


def main() -> None:
    args = parse_args()
    apply_suite_report_defaults(args)
    arms = build_arms(args)
    if args.suite:
        suite_out = suite_root(args.results_root, args.suite)
        output_dir = args.output_dir or (suite_out / "summaries")
        figure_dir = args.figure_dir or (suite_out / "figures" / (args.output_prefix or comparison_stem(args)))
    else:
        output_dir = args.output_dir or (args.results_root / "summaries")
        figure_dir = args.figure_dir or (args.results_root / "figures" / (args.output_prefix or comparison_stem(args)))
    output_prefix = args.output_prefix or comparison_stem(args)

    summaries = load_summaries(args, arms)
    args.cooper_prefix_target_lengths = resolve_cooper_prefix_target_lengths(summaries)
    rows = per_arm_rows(args, arms, summaries)
    overall = weighted_overall(args, arms, summaries)
    grid_validation = validate_grid(args, arms)
    target_ppl_table = target_ppl_rows(args, rows, overall)
    native_pair_rows = paired_native_rows(args, arms)
    best_native_rows = best_native_split_rows(args, arms)

    per_arm_path = output_dir / f"{output_prefix}.per_arm.csv"
    target_ppl_path = output_dir / f"{output_prefix}.target_ppl.csv"
    native_pair_path = output_dir / f"{output_prefix}.native_paired_to_prefix100.csv"
    best_native_path = output_dir / f"{output_prefix}.native_best_split.csv"
    insights_path = output_dir / f"{output_prefix}.insights.md"
    json_path = output_dir / f"{output_prefix}.summary.json"

    write_csv(per_arm_path, rows)
    write_csv(target_ppl_path, target_ppl_table)
    write_csv(native_pair_path, native_pair_rows)
    write_csv(best_native_path, best_native_rows)

    figure_artifacts: dict[str, Any] = {}
    if not args.skip_figures:
        figure_artifacts.update(plot_ltr_token_geomean(args, rows, figure_dir))
        if is_target_length_sensitivity_report(args):
            figure_artifacts.update(plot_ltr_cooper_span_extractability(args, rows, figure_dir))
        figure_artifacts.update(plot_ltr_cooper_pz_survival(args, arms, figure_dir))
        figure_artifacts.update(plot_ltr_memorized_min_tokenprob_density(args, arms, figure_dir))
        figure_artifacts.update(plot_target_ppl_table(target_ppl_table, figure_dir))
        figure_artifacts.update(plot_native_profile(rows, overall, figure_dir))
        figure_artifacts.update(plot_native_heatmaps(rows, figure_dir))
        figure_artifacts.update(plot_native_topk_support_profile(rows, overall, figure_dir))
        figure_artifacts.update(plot_native_topk_support_heatmap_linear(rows, figure_dir))
        figure_artifacts.update(plot_native_paired(native_pair_rows, best_native_rows, figure_dir))

    insight_lines = generate_insights(
        args=args,
        arms=arms,
        overall=overall,
        native_pairs=native_pair_rows,
        best_native=best_native_rows,
    )
    write_insights(insights_path, insight_lines)

    audit_artifacts: dict[str, str] = {}
    if should_generate_ltr_audit_figures(args):
        audit_artifacts = generate_ltr_audit_figures(args)

    output_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_results_root": str(args.results_root),
                "suite": args.suite,
                "suite_report": args.suite_report if args.suite else None,
                "study_name": args.study_name,
                "repetitions": args.repetitions,
                "context_budget": args.context_budget,
                "middle_length": args.middle_length,
                "cooper_prefix_target_lengths": list(cooper_prefix_target_lengths(args)),
                "window_stride": args.window_stride,
                "window_layout": args.window_layout,
                "max_windows_per_excerpt": args.max_windows_per_excerpt,
                "arms": [arm.__dict__ for arm in arms],
                "overall": overall,
                "window_grid_validation": grid_validation,
                "insights": insight_lines,
                "artifacts": {
                    "per_arm_csv": str(per_arm_path),
                    "target_ppl_csv": str(target_ppl_path) if target_ppl_table else None,
                    "native_paired_to_prefix100_csv": str(native_pair_path)
                    if native_pair_rows
                    else None,
                    "native_best_split_csv": str(best_native_path) if best_native_rows else None,
                    "insights_markdown": str(insights_path),
                    "summary_json": str(json_path),
                    "figure_dir": str(figure_dir) if not args.skip_figures else None,
                    "figures": figure_artifacts,
                    "ltr_audit_figures": audit_artifacts or None,
                },
            },
            handle,
            indent=2,
        )

    print(f"Wrote per-arm CSV: {per_arm_path}")
    if target_ppl_table:
        print(f"Wrote target-PPL CSV: {target_ppl_path}")
    if native_pair_rows:
        print(f"Wrote native paired-window CSV: {native_pair_path}")
    if best_native_rows:
        print(f"Wrote native best-split CSV: {best_native_path}")
    print(f"Wrote insights: {insights_path}")
    print(f"Wrote summary JSON: {json_path}")
    if not args.skip_figures:
        print(f"Wrote figures under: {figure_dir}")
    if audit_artifacts:
        print(f"Wrote LTR audit figures under: {audit_artifacts['audit_dir']}")


if __name__ == "__main__":
    main()
