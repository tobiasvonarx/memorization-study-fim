#!/usr/bin/env python3
"""Create a FIM vs no-FIM spider plot from lm-eval JSON outputs."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_TASKS = [
    "arc_challenge",
    "arc_easy",
    "commonsense_qa",
    "hellaswag",
    "mmlu",
    "piqa",
    "wikitext",
    "winogrande",
]

DEFAULT_METRICS = {
    "arc_challenge": "acc",
    "arc_easy": "acc",
    "commonsense_qa": "acc",
    "hellaswag": "acc",
    "mmlu": "acc",
    "piqa": "acc",
    "wikitext": "word_perplexity",
    "winogrande": "acc",
}


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_metric(results: dict, task: str, metric: str) -> float:
    task_data = results.get(task)
    if task_data is None:
        raise KeyError(f"Task '{task}' not found in results JSON.")

    direct = task_data.get(metric)
    if isinstance(direct, (int, float)):
        return float(direct)

    key_none = f"{metric},none"
    value = task_data.get(key_none)
    if isinstance(value, (int, float)):
        return float(value)

    raise KeyError(f"Metric '{metric}' not found for task '{task}'.")


def _task_label(task: str, metric: str, wikitext_transform: str) -> str:
    if task != "wikitext":
        return task
    if wikitext_transform == "inverse":
        return f"{task}\n(1/{metric})"
    return f"{task}\n(-{metric})"


def _score(task: str, metric_value: float, wikitext_transform: str) -> float:
    if task != "wikitext":
        return metric_value
    if wikitext_transform == "inverse":
        if metric_value <= 0:
            raise ValueError("Wikitext metric must be > 0 for inverse transform.")
        return 1.0 / metric_value
    return -metric_value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot FIM vs no-FIM benchmark performance as a spider plot."
    )
    parser.add_argument("--fim-json", required=True, help="Path to FIM lm-eval JSON.")
    parser.add_argument("--no-fim-json", required=True, help="Path to no-FIM lm-eval JSON.")
    parser.add_argument(
        "--output",
        default="results/lm_eval/spider_fim_vs_no_fim.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="Task list for axes (default: arc_challenge arc_easy commonsense_qa hellaswag mmlu piqa wikitext winogrande).",
    )
    parser.add_argument(
        "--wikitext-metric",
        default="word_perplexity",
        choices=["word_perplexity", "byte_perplexity", "bits_per_byte"],
        help="Wikitext metric to use if wikitext is included in --tasks.",
    )
    parser.add_argument(
        "--wikitext-transform",
        default="inverse",
        choices=["inverse", "negative"],
        help="How to make wikitext 'higher is better' for plotting.",
    )
    parser.add_argument(
        "--title",
        default="FIM vs no-FIM benchmark spider plot",
        help="Plot title.",
    )
    args = parser.parse_args()

    fim = _load_json(args.fim_json)
    no_fim = _load_json(args.no_fim_json)
    fim_results = fim.get("results", {})
    no_fim_results = no_fim.get("results", {})

    tasks = args.tasks
    if len(tasks) < 3:
        raise ValueError("Use at least 3 tasks for a spider plot.")

    metrics = dict(DEFAULT_METRICS)
    if "wikitext" in tasks:
        metrics["wikitext"] = args.wikitext_metric

    fim_scores = []
    no_fim_scores = []
    axis_labels = []

    for task in tasks:
        metric = metrics.get(task, "acc")
        fim_metric = _get_metric(fim_results, task, metric)
        no_fim_metric = _get_metric(no_fim_results, task, metric)

        fim_scores.append(_score(task, fim_metric, args.wikitext_transform))
        no_fim_scores.append(_score(task, no_fim_metric, args.wikitext_transform))
        axis_labels.append(_task_label(task, metric, args.wikitext_transform))

    n = len(tasks)
    angles = [2.0 * math.pi * i / n for i in range(n)]
    angles_closed = angles + [angles[0]]
    fim_closed = fim_scores + [fim_scores[0]]
    no_fim_closed = no_fim_scores + [no_fim_scores[0]]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles_closed, no_fim_closed, linewidth=2, label="no-FIM", color="#1f77b4")
    ax.fill(angles_closed, no_fim_closed, alpha=0.15, color="#1f77b4")

    ax.plot(angles_closed, fim_closed, linewidth=2, label="FIM", color="#d62728")
    ax.fill(angles_closed, fim_closed, alpha=0.15, color="#d62728")

    ax.set_xticks(angles)
    ax.set_xticklabels(axis_labels)
    ax.set_title(args.title, pad=24)
    ax.grid(alpha=0.35)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15))

    min_val = min(min(fim_scores), min(no_fim_scores))
    max_val = max(max(fim_scores), max(no_fim_scores))
    if min_val >= 0:
        ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 1.0)
    else:
        pad = 0.1 * (max_val - min_val if max_val != min_val else 1.0)
        ax.set_ylim(min_val - pad, max_val + pad)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
