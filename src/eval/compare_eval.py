#!/usr/bin/env python3
import argparse
import csv
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Compare two lm-eval result JSON files")
    parser.add_argument("--fim", required=True, help="Path to FIM result JSON")
    parser.add_argument("--no-fim", required=True, dest="no_fim", help="Path to no-FIM result JSON")
    args = parser.parse_args()

    with open(args.fim, "r", encoding="utf-8") as f:
        fim = json.load(f)
    with open(args.no_fim, "r", encoding="utf-8") as f:
        no_fim = json.load(f)

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["task", "metric", "fim", "no_fim", "delta"])
    fim_results = fim.get("results", {})
    no_fim_results = no_fim.get("results", {})

    tasks = sorted(set(fim_results.keys()) & set(no_fim_results.keys()))
    for task in tasks:
        fim_metrics = fim_results[task]
        no_fim_metrics = no_fim_results[task]
        for metric in sorted(set(fim_metrics.keys()) & set(no_fim_metrics.keys())):
            fv = fim_metrics[metric]
            nv = no_fim_metrics[metric]
            if isinstance(fv, (int, float)) and isinstance(nv, (int, float)):
                writer.writerow([task, metric, fv, nv, fv - nv])


if __name__ == "__main__":
    main()
