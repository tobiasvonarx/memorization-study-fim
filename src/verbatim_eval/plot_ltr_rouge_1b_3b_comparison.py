#!/usr/bin/env python3
"""Plot 1B/3B ROUGE-L comparisons for LTR probe settings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))

from collation_utils import apply_conference_style, interval_band, set_repetition_axis


MODEL_ORDER = ("no_fim", "fim_v2")
MODEL_LABELS = {"no_fim": "LTR", "fim_v2": "FIM"}
MODEL_COLORS = {"no_fim": "#0072B2", "fim_v2": "#CC79A7"}
SIZE_ORDER = ("3B", "1B")
SIZE_MARKERS = {"3B": "o", "1B": "s"}
SIZE_LINEWIDTHS = {"3B": 2.2, "1B": 1.55}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-suite", default="comparison_1b_3b")
    parser.add_argument("--selected-3b-suite", default="exp1_ltr_direct_w10_all_seed0")
    parser.add_argument("--selected-1b-suite", default="exp1_ltr_direct_1b_w10_all_seed0")
    parser.add_argument("--start-3b-suite", default="exp2_ltr_start_window_all_reps")
    parser.add_argument("--start-1b-suite", default="exp2_ltr_start_window_1b_all_reps")
    return parser.parse_args()


def suite_root(results_root: Path, suite: str) -> Path:
    return results_root / "verbatim_eval" / "suites" / suite


def summary_csv(results_root: Path, suite: str, suffix: str) -> Path:
    path = suite_root(results_root, suite) / "summaries" / f"{suite}_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_rows(path: Path, *, model_size: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model_label = str(row["model_label"])
            if model_label not in MODEL_ORDER:
                continue
            rows.append(
                {
                    "model_label": model_label,
                    "model_size": model_size,
                    "display_label": f"{MODEL_LABELS[model_label]} {model_size}",
                    "repetition": int(row["repetition"]),
                    "num_windows": int(row["num_windows"]),
                    "rouge_l_mean": float(row["rouge_l_mean"]),
                    "rouge_l_ci95": float(row["rouge_l_ci95"]),
                }
            )
    return rows


def validate_rows(rows: list[dict[str, Any]], *, label: str) -> list[int]:
    if not rows:
        raise ValueError(f"No rows found for {label}.")
    by_series: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        key = (str(row["model_label"]), str(row["model_size"]))
        by_series.setdefault(key, []).append(int(row["repetition"]))
    missing = [
        f"{model_label} {model_size}"
        for model_label in MODEL_ORDER
        for model_size in SIZE_ORDER
        if (model_label, model_size) not in by_series
    ]
    if missing:
        raise ValueError(f"{label}: missing series: {', '.join(missing)}")
    repetitions = sorted(set(int(row["repetition"]) for row in rows))
    for key, reps in by_series.items():
        if sorted(reps) != repetitions:
            raise ValueError(f"{label}: series {key} has repetitions {sorted(reps)}, expected {repetitions}")
    return repetitions


def write_combined_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_label",
        "model_size",
        "display_label",
        "repetition",
        "num_windows",
        "rouge_l_mean",
        "rouge_l_ci95",
    ]
    ordered = sorted(
        rows,
        key=lambda row: (
            MODEL_ORDER.index(str(row["model_label"])),
            SIZE_ORDER.index(str(row["model_size"])),
            int(row["repetition"]),
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)


def endpoint_label_positions(endpoints: dict[str, float], *, min_gap: float = 0.045) -> dict[str, float]:
    ordered = sorted(endpoints.items(), key=lambda item: item[1])
    positions: dict[str, float] = {}
    previous = 0.025 - min_gap
    for label, value in ordered:
        positions[label] = min(max(value, previous + min_gap), 0.975)
        previous = positions[label]
    overflow = max(positions.values(), default=0.0) - 0.975
    if overflow > 0:
        positions = {label: max(0.025, value - overflow) for label, value in positions.items()}
    return positions


def plot_comparison(
    rows: list[dict[str, Any]],
    *,
    repetitions: list[int],
    pdf_path: Path,
    png_path: Path,
    caption_path: Path,
    caption: str,
) -> None:
    import matplotlib.pyplot as plt

    apply_conference_style(plt)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    lookup = {
        (str(row["model_label"]), str(row["model_size"]), int(row["repetition"])): row
        for row in rows
    }

    fig, ax = plt.subplots(figsize=(4.7, 3.1))
    fig.subplots_adjust(left=0.14, right=0.85, bottom=0.18, top=0.96)

    endpoints: dict[str, float] = {}
    endpoint_colors: dict[str, str] = {}
    for model_label in MODEL_ORDER:
        for model_size in SIZE_ORDER:
            label = f"{MODEL_LABELS[model_label]} {model_size}"
            values = [
                float(lookup[(model_label, model_size, repetition)]["rouge_l_mean"])
                for repetition in repetitions
            ]
            errors = [
                float(lookup[(model_label, model_size, repetition)]["rouge_l_ci95"])
                for repetition in repetitions
            ]
            lower, upper = interval_band(values, errors, lower_floor=0.0, upper_ceiling=1.0)
            color = MODEL_COLORS[model_label]
            alpha = 0.13 if model_size == "3B" else 0.08
            ax.fill_between(repetitions, lower, upper, color=color, alpha=alpha, linewidth=0.0)
            marker_face = color if model_size == "3B" else "white"
            ax.plot(
                repetitions,
                values,
                color=color,
                linestyle="-",
                marker=SIZE_MARKERS[model_size],
                markerfacecolor=marker_face,
                markeredgecolor=color,
                markeredgewidth=1.25,
                linewidth=SIZE_LINEWIDTHS[model_size],
                label=label,
                solid_capstyle="round",
                zorder=3 if model_size == "1B" else 2,
            )
            endpoints[label] = values[-1]
            endpoint_colors[label] = color

    set_repetition_axis(ax, repetitions)
    ax.set_xlim(min(repetitions) - 2, max(repetitions) + 31)
    ax.set_xlabel("Training repetitions")
    ax.set_ylabel("Mean ROUGE-L")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.18)
    label_x = max(repetitions) + 5.0
    for label, label_y in endpoint_label_positions(endpoints).items():
        endpoint_y = endpoints[label]
        color = endpoint_colors[label]
        ax.plot([max(repetitions), label_x - 0.8], [endpoint_y, label_y], color=color, alpha=0.65, linewidth=0.65)
        ax.text(label_x, label_y, label, color=color, va="center", ha="left", fontsize=8.4)

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    caption_path.write_text(caption + "\n", encoding="utf-8")


def make_plot(
    *,
    results_root: Path,
    output_root: Path,
    setting_name: str,
    csv_suffix: str,
    suite_3b: str,
    suite_1b: str,
    figure_name: str,
    caption: str,
) -> dict[str, Any]:
    rows = (
        read_rows(summary_csv(results_root, suite_3b, csv_suffix), model_size="3B")
        + read_rows(summary_csv(results_root, suite_1b, csv_suffix), model_size="1B")
    )
    repetitions = validate_rows(rows, label=setting_name)

    summaries_dir = output_root / "summaries"
    figures_dir = output_root / "figures" / figure_name
    csv_path = summaries_dir / f"{figure_name}.csv"
    pdf_path = figures_dir / f"{figure_name}.pdf"
    png_path = figures_dir / f"{figure_name}.png"
    caption_path = figures_dir / f"{figure_name}.caption.txt"

    write_combined_csv(csv_path, rows)
    plot_comparison(
        rows,
        repetitions=repetitions,
        pdf_path=pdf_path,
        png_path=png_path,
        caption_path=caption_path,
        caption=caption,
    )
    return {
        "setting": setting_name,
        "suites": {"3B": suite_3b, "1B": suite_1b},
        "repetitions": repetitions,
        "csv": str(csv_path),
        "figure": str(pdf_path),
        "png": str(png_path),
        "caption": str(caption_path),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    output_root = suite_root(args.results_root, args.output_suite)
    output_root.mkdir(parents=True, exist_ok=True)

    outputs = [
        make_plot(
            results_root=args.results_root,
            output_root=output_root,
            setting_name="10 selected windows",
            csv_suffix="ltr_selected_windows_rouge",
            suite_3b=args.selected_3b_suite,
            suite_1b=args.selected_1b_suite,
            figure_name="ltr_selected_windows_rouge_1b_3b",
            caption=(
                "Mean greedy ROUGE-L for LTR prefix probes over 10 uniformly sampled "
                "non-overlapping windows per excerpt. Blue and pink denote LTR- and "
                "FIM-trained models; filled circles and hollow squares denote 3B and 1B models. "
                "Shaded bands show nominal 95% confidence intervals over windows."
            ),
        ),
        make_plot(
            results_root=args.results_root,
            output_root=output_root,
            setting_name="first window",
            csv_suffix="ltr_start_window_rouge",
            suite_3b=args.start_3b_suite,
            suite_1b=args.start_1b_suite,
            figure_name="ltr_start_window_rouge_1b_3b",
            caption=(
                "Mean greedy ROUGE-L for the first LTR probe window in each excerpt. "
                "Blue and pink denote LTR- and FIM-trained models; filled circles and "
                "hollow squares denote 3B and 1B models. Shaded bands show nominal 95% "
                "confidence intervals over excerpts."
            ),
        ),
    ]

    summary_path = output_root / "summaries" / f"{args.output_suite}.summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"outputs": outputs}, indent=2) + "\n", encoding="utf-8")
    for output in outputs:
        print(f"Wrote figure: {output['figure']}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
