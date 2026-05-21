#!/usr/bin/env python3
"""Plot mean ROUGE-L for selected LTR windows by repetition."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))

from collation_utils import apply_conference_style, ci95_from_std, finite_ylim, interval_band, set_repetition_axis
from verbatim_suite import arm_root, load_manifest_for_suite, suite_arms_for_report, suite_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--suite", required=True)
    parser.add_argument("--repetitions", type=int, nargs="+", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--expected-max-windows-per-excerpt", type=int, default=10)
    parser.add_argument("--expected-window-selection", default="uniform")
    return parser.parse_args()


def load_summary(results_root: Path, suite: str, arm_id: str, repetition: int) -> dict[str, Any]:
    path = arm_root(results_root, suite, arm_id, repetition) / "windows.summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_summary(summary: dict[str, Any], *, repetition: int, args: argparse.Namespace) -> None:
    expected = {
        "repetition": repetition,
        "prompt_format": "ltr_prefix",
        "prefix_length": 100,
        "middle_length": 32,
        "suffix_length": 0,
        "window_layout": "cooper_nonoverlap",
        "max_windows_per_excerpt": args.expected_max_windows_per_excerpt,
        "window_selection": args.expected_window_selection,
        "generation_mode": "greedy",
    }
    mismatches = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} expected {expected!r}"
            for key, (actual, expected) in mismatches.items()
        )
        raise ValueError(f"Summary does not match selected-window LTR experiment: {details}")


def requested_repetitions(args: argparse.Namespace, manifest: dict[str, Any]) -> list[int]:
    if args.repetitions:
        return sorted(int(rep) for rep in args.repetitions)
    return sorted(int(rep) for rep in manifest.get("repetitions", []))


def model_display_label(model_label: str) -> str:
    return {
        "no_fim": "LTR",
        "fim_v2": "FIM",
    }.get(model_label, model_label.replace("_", "-"))


def main() -> None:
    args = parse_args()
    manifest = load_manifest_for_suite(args.results_root, args.suite)
    repetitions = requested_repetitions(args, manifest)
    if not repetitions:
        raise ValueError("No repetition buckets requested or found in the manifest.")

    arms = [
        arm
        for arm in suite_arms_for_report(manifest, "ltr")
        if str(arm.get("model_label")) in {"no_fim", "fim_v2"}
    ]
    if len(arms) != 2:
        labels = [str(arm.get("model_label")) for arm in arms]
        raise ValueError(f"Expected no_fim and fim_v2 LTR arms; found {labels}")
    arms.sort(key=lambda arm: ["no_fim", "fim_v2"].index(str(arm["model_label"])))

    rows: list[dict[str, Any]] = []
    for repetition in repetitions:
        for arm in arms:
            summary = load_summary(args.results_root, args.suite, str(arm["arm_id"]), repetition)
            validate_summary(summary, repetition=repetition, args=args)
            metric = summary["metrics"]["Rouge-L"]
            count = int(summary["num_windows"])
            std = float(metric.get("std", float("nan")))
            rows.append(
                {
                    "model_label": str(arm["model_label"]),
                    "display_label": model_display_label(str(arm["model_label"])),
                    "repetition": repetition,
                    "mean": float(metric["mean"]),
                    "std": std,
                    "ci95": ci95_from_std(std, count),
                    "num_windows": count,
                    "num_excerpts": int(summary["num_excerpts_with_windows"]),
                }
            )

    output_prefix = args.output_prefix or f"{args.suite}_ltr_selected_windows_rouge"
    output_dir = suite_root(args.results_root, args.suite) / "summaries"
    figure_dir = suite_root(args.results_root, args.suite) / "figures" / output_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{output_prefix}.csv"
    csv_path.write_text(
        (
            "model_label,display_label,repetition,num_excerpts,num_windows,"
            "rouge_l_mean,rouge_l_std,rouge_l_ci95\n"
        )
        + "\n".join(
            f"{row['model_label']},{row['display_label']},{row['repetition']},"
            f"{row['num_excerpts']},{row['num_windows']},"
            f"{row['mean']:.10g},{row['std']:.10g},{row['ci95']:.10g}"
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )

    import matplotlib.pyplot as plt

    apply_conference_style(plt)
    colors = {"no_fim": "#0072B2", "fim_v2": "#CC79A7"}
    all_values = [row["mean"] for row in rows]
    all_errors = [row["ci95"] if math.isfinite(row["ci95"]) else 0.0 for row in rows]

    fig, ax = plt.subplots(figsize=(4.25, 3.1))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.18, top=0.88)
    lookup = {(row["model_label"], row["repetition"]): row for row in rows}
    for arm in arms:
        model_label = str(arm["model_label"])
        values = [lookup[(model_label, rep)]["mean"] for rep in repetitions]
        errors = [lookup[(model_label, rep)]["ci95"] for rep in repetitions]
        lower, upper = interval_band(values, errors, lower_floor=0.0, upper_ceiling=1.0)
        color = colors[model_label]
        ax.fill_between(repetitions, lower, upper, color=color, alpha=0.18, linewidth=0.0)
        ax.plot(
            repetitions,
            values,
            color=color,
            marker="o",
            linewidth=2.1,
            label=model_display_label(model_label),
            solid_capstyle="round",
        )

    set_repetition_axis(ax, repetitions)
    ax.set_xlabel("Training repetitions")
    ax.set_ylabel("Mean ROUGE-L")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.18)

    pdf_path = figure_dir / "ltr_selected_windows_rouge.pdf"
    png_path = figure_dir / "ltr_selected_windows_rouge.png"
    caption_path = figure_dir / "ltr_selected_windows_rouge.caption.txt"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    window_counts = sorted({int(row["num_windows"]) for row in rows})
    if len(window_counts) == 1:
        window_count_text = f"n={window_counts[0]:,} selected windows"
    else:
        window_count_text = f"n={window_counts[0]:,}-{window_counts[-1]:,} selected windows"
    caption_path.write_text(
        (
            "Greedy ROUGE-L for the LTR probe using 10 uniformly sampled "
            "non-overlapping windows per Gutenberg excerpt. Each prompt uses "
            "100 prefix tokens to generate a 32-token continuation; means are "
            f"computed over {window_count_text} per model and repetition, with "
            "nominal 95% confidence intervals over windows.\n"
        ),
        encoding="utf-8",
    )

    summary_path = output_dir / f"{output_prefix}.summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "suite": args.suite,
                "repetitions": repetitions,
                "csv": str(csv_path),
                "figure": str(pdf_path),
                "png": str(png_path),
                "caption": str(caption_path),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote figure: {pdf_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
