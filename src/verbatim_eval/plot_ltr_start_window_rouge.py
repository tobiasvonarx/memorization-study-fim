#!/usr/bin/env python3
"""Plot ROUGE-L for the first LTR window in each excerpt."""

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
    parser.add_argument("--suite", required=True, help="Output suite name.")
    parser.add_argument(
        "--source-suite",
        default=None,
        help="Optional existing suite to read windows from. Defaults to --suite.",
    )
    parser.add_argument("--repetition", type=int, default=None, help="Single repetition bucket to plot.")
    parser.add_argument("--repetitions", type=int, nargs="+", default=None, help="Repetition buckets to plot.")
    parser.add_argument("--target-start", type=int, default=100)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--skip-window-validation", action="store_true")
    return parser.parse_args()


def load_summary(results_root: Path, suite: str, arm_id: str, repetition: int) -> dict[str, Any]:
    path = arm_root(results_root, suite, arm_id, repetition) / "windows.summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_summary(summary: dict[str, Any], *, repetition: int) -> None:
    expected = {
        "repetition": repetition,
        "prompt_format": "ltr_prefix",
        "prefix_length": 100,
        "middle_length": 32,
        "suffix_length": 0,
        "window_layout": "cooper_nonoverlap",
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
        raise ValueError(f"Summary does not match first-window LTR experiment: {details}")


def load_start_window_rouge(
    results_root: Path,
    suite: str,
    arm_id: str,
    repetition: int,
    *,
    target_start: int,
    expected_excerpts: int | None,
    validate: bool,
) -> list[float]:
    path = arm_root(results_root, suite, arm_id, repetition) / "windows.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    seen_excerpts: set[str] = set()
    values: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if int(row.get("target_start", -1)) != target_start:
                continue
            excerpt_id = str(row.get("excerpt_id"))
            if excerpt_id in seen_excerpts:
                raise ValueError(f"{path}:{line_number}: duplicate excerpt_id={excerpt_id}")
            seen_excerpts.add(excerpt_id)
            values.append(float(row["Rouge-L"]))
    if validate and expected_excerpts is not None and len(values) != expected_excerpts:
        raise ValueError(
            f"{path}: expected one target_start={target_start} window for each of "
            f"{expected_excerpts} excerpts, found {len(values)}"
        )
    if not values:
        raise ValueError(f"{path}: no windows found with target_start={target_start}")
    return values


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def model_display_label(model_label: str) -> str:
    return {
        "no_fim": "LTR",
        "fim_v2": "FIM",
    }.get(model_label, model_label.replace("_", "-"))


def requested_repetitions(args: argparse.Namespace, manifest: dict[str, Any]) -> list[int]:
    if args.repetitions:
        return sorted(int(rep) for rep in args.repetitions)
    if args.repetition is not None:
        return [int(args.repetition)]
    return sorted(int(rep) for rep in manifest.get("repetitions", []))


def main() -> None:
    args = parse_args()
    source_suite = args.source_suite or args.suite
    manifest = load_manifest_for_suite(args.results_root, source_suite)
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
            summary = load_summary(args.results_root, source_suite, str(arm["arm_id"]), repetition)
            validate_summary(summary, repetition=repetition)
            expected_excerpts = int(summary.get("num_excerpts_with_windows", 0)) or None
            values = load_start_window_rouge(
                args.results_root,
                source_suite,
                str(arm["arm_id"]),
                repetition,
                target_start=args.target_start,
                expected_excerpts=expected_excerpts,
                validate=not args.skip_window_validation,
            )
            mean = sum(values) / len(values)
            std = sample_std(values)
            rows.append(
                {
                    "model_label": str(arm["model_label"]),
                    "display_label": model_display_label(str(arm["model_label"])),
                    "repetition": repetition,
                    "mean": mean,
                    "std": std,
                    "ci95": ci95_from_std(std, len(values)),
                    "num_windows": len(values),
                }
            )

    output_prefix = args.output_prefix or f"{args.suite}_ltr_start_window_rouge"
    output_dir = suite_root(args.results_root, args.suite) / "summaries"
    figure_dir = suite_root(args.results_root, args.suite) / "figures" / output_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{output_prefix}.csv"
    csv_path.write_text(
        "model_label,display_label,repetition,num_windows,rouge_l_mean,rouge_l_std,rouge_l_ci95\n"
        + "\n".join(
            f"{row['model_label']},{row['display_label']},{row['repetition']},"
            f"{row['num_windows']},{row['mean']:.10g},{row['std']:.10g},{row['ci95']:.10g}"
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

    if len(repetitions) == 1:
        labels = [row["display_label"] for row in rows]
        values = [row["mean"] for row in rows]
        errors = [row["ci95"] if math.isfinite(row["ci95"]) else 0.0 for row in rows]

        fig, ax = plt.subplots(figsize=(3.5, 3.1))
        fig.subplots_adjust(left=0.19, right=0.98, bottom=0.18, top=0.95)
        x = list(range(len(rows)))
        ax.bar(
            x,
            values,
            yerr=errors,
            color=[colors[row["model_label"]] for row in rows],
            edgecolor="black",
            linewidth=0.7,
            capsize=3.0,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    else:
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
    ax.set_ylim(*(finite_ylim(all_values, all_errors, lower_floor=0.0, pad=0.18) or (0.0, 1.0)))
    ax.set_ylim(ax.get_ylim()[0], min(1.0, ax.get_ylim()[1]))
    ax.grid(True, axis="y", alpha=0.18)

    pdf_path = figure_dir / "ltr_start_window_rouge.pdf"
    png_path = figure_dir / "ltr_start_window_rouge.png"
    caption_path = figure_dir / "ltr_start_window_rouge.caption.txt"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    window_counts = sorted({int(row["num_windows"]) for row in rows})
    if len(window_counts) == 1:
        window_count_text = f"n={window_counts[0]:,} per model and repetition"
    else:
        window_count_text = f"n={window_counts[0]:,}-{window_counts[-1]:,} per model and repetition"
    if len(repetitions) == 1:
        repetition_text = f"the {repetitions[0]}-repetition Gutenberg excerpts"
    else:
        repetition_text = "each Gutenberg repetition bucket"
    caption_path.write_text(
        (
            f"Greedy ROUGE-L for the first LTR probe window in {repetition_text}. "
            "Each prompt uses 100 prefix tokens to generate a "
            "32-token continuation starting at token 100; one window is evaluated "
            f"per excerpt ({window_count_text}). Means are shown with nominal "
            "95% confidence intervals over excerpts.\n"
        ),
        encoding="utf-8",
    )

    summary_path = output_dir / f"{output_prefix}.summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "suite": args.suite,
                "source_suite": source_suite,
                "repetitions": repetitions,
                "target_start": args.target_start,
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
