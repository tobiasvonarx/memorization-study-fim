#!/usr/bin/env python3
"""Create and submit unified verbatim-eval debug runs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from verbatim_suite import DEFAULT_REPETITIONS, build_manifest, write_manifest

EVAL_KIND_NAMES = {"direct", "attention"}
EXPERIMENT_ALIASES = {
    "direct": ["ltr", "prefix_rescue"],
    "direct_core": ["ltr", "prefix_rescue"],
    "attention": ["attention_ltr", "attention_native_geometry"],
    "attention_core": ["attention_ltr", "attention_native_geometry"],
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_results_root() -> Path:
    return repo_root() / "results"


def parse_repetitions(value: str | None) -> list[int]:
    if not value:
        return DEFAULT_REPETITIONS
    return [int(item) for item in value.replace(",", " ").split()]


def parse_eval_kinds(value: str) -> list[str]:
    kinds = [item for item in value.replace(",", " ").split() if item]
    if not kinds or "all" in kinds:
        return sorted(EVAL_KIND_NAMES)
    unknown = sorted(set(kinds) - EVAL_KIND_NAMES)
    if unknown:
        raise ValueError(f"Unknown eval kind(s): {', '.join(unknown)}")
    return kinds


def resolve_experiments(value: str, manifest: dict) -> list[str]:
    available = {str(arm["experiment"]) for arm in manifest["arms"]}
    tokens = [item for item in value.replace(",", " ").split() if item]
    if not tokens or "all" in tokens:
        return sorted(available)
    selected: set[str] = set()
    for token in tokens:
        selected.update(EXPERIMENT_ALIASES.get(token, [token]))
    unknown = sorted(selected - available)
    if unknown:
        raise ValueError(f"Unknown experiment(s): {', '.join(unknown)}")
    return sorted(selected)


def parse_model_labels(value: str | None, manifest: dict) -> list[str] | None:
    if not value:
        return None
    available = {str(arm["model_label"]) for arm in manifest["arms"]}
    labels = [item for item in value.replace(",", " ").split() if item]
    unknown = sorted(set(labels) - available)
    if unknown:
        raise ValueError(f"Unknown model label(s): {', '.join(unknown)}")
    return labels


def filter_manifest_model_labels(manifest: dict, model_labels: list[str] | None) -> None:
    if not model_labels:
        return
    keep = set(model_labels)
    manifest["arms"] = [arm for arm in manifest["arms"] if str(arm["model_label"]) in keep]
    manifest["num_arms"] = len(manifest["arms"])
    manifest["num_tasks"] = len(manifest["arms"]) * len(manifest["repetitions"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit the unified verbatim memorization suite")
    parser.add_argument("--suite", default="core", help="Suite name below results/verbatim_eval/suites")
    parser.add_argument("--results-root", type=Path, default=default_results_root())
    parser.add_argument("--repetitions", default=None, help="Comma/space-separated repetition buckets")
    parser.add_argument("--max-excerpts", type=int, default=0, help="Use 0 for all deduped excerpts")
    parser.add_argument("--max-windows-per-excerpt", type=int, default=5)
    parser.add_argument("--window-selection", choices=["first", "uniform"], default="uniform")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--attention-batch-size", type=int, default=16)
    parser.add_argument("--attention-max-windows-per-excerpt", type=int, default=4)
    parser.add_argument(
        "--experiments",
        default="direct",
        help="Comma/space-separated experiments or aliases: direct, ltr, prefix_rescue, attention, all",
    )
    parser.add_argument("--eval-kinds", default="all", help="Usually leave as all; optionally restrict to direct or attention")
    parser.add_argument(
        "--model-labels",
        default=None,
        help="Optional comma/space-separated model labels to keep, e.g. no_fim,fim_v2.",
    )
    parser.add_argument("--num-gpu-workers", type=int, default=4)
    parser.add_argument(
        "--generation-mode",
        choices=["auto", "none", "greedy"],
        default="auto",
        help="auto computes greedy/Rouge for direct LTR and skips greedy elsewhere.",
    )
    parser.add_argument("--partition", default=None, help="Optional Slurm partition override.")
    parser.add_argument("--time", default="01:30:00", help="Slurm walltime for the single debug job")
    parser.add_argument("--walltime-minutes", type=float, default=90.0)
    parser.add_argument("--stop-margin-minutes", type=float, default=5.0)
    parser.add_argument(
        "--unique-eval-token-file",
        type=Path,
        default=None,
        help="Override the final unique Gutenberg token JSONL used for probing.",
    )
    parser.add_argument(
        "--replica-buckets-dir",
        type=Path,
        default=None,
        help="Override the replica-expanded directory used only for bucket membership.",
    )
    parser.add_argument(
        "--unique-eval-cache-dir",
        type=Path,
        default=None,
        help="Override where compact unique per-repetition eval files are cached.",
    )
    parser.add_argument(
        "--use-replica-jsonl-eval",
        action="store_true",
        help="Legacy mode: evaluate directly from replica-expanded rep_N_token.jsonl files.",
    )
    parser.add_argument(
        "--rebuild-unique-eval-cache",
        action="store_true",
        help="Force rebuild of compact unique eval cache files.",
    )
    parser.add_argument("--no-fim-model-path", type=Path, default=None)
    parser.add_argument("--fim-v2-model-path", type=Path, default=None)
    parser.add_argument("--fineweb-only-model-path", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write the manifest and print sbatch without submitting")
    return parser.parse_args()


def selected_task_count(manifest: dict, eval_kinds: list[str], experiments: list[str]) -> int:
    eval_kind_set = set(eval_kinds)
    experiment_set = set(experiments)
    selected_arms = [
        arm
        for arm in manifest["arms"]
        if str(arm.get("eval_kind", "direct")) in eval_kind_set
        and str(arm["experiment"]) in experiment_set
    ]
    return len(selected_arms) * len(manifest["repetitions"])


def selected_arms(manifest: dict, eval_kinds: list[str], experiments: list[str]) -> list[dict]:
    eval_kind_set = set(eval_kinds)
    experiment_set = set(experiments)
    return [
        arm
        for arm in manifest["arms"]
        if str(arm.get("eval_kind", "direct")) in eval_kind_set
        and str(arm["experiment"]) in experiment_set
    ]


def build_collation_commands(args: argparse.Namespace, manifest: dict, eval_kinds: list[str], experiments: list[str]) -> list[list[str]]:
    arms = selected_arms(manifest, eval_kinds, experiments)
    selected = {(str(arm.get("eval_kind", "direct")), str(arm["experiment"])) for arm in arms}
    python_cmd = "${PYTHON_BIN:-python}"
    script_dir = repo_root() / "src" / "verbatim_eval"
    common = ["--results-root", str(args.results_root), "--suite", args.suite]
    commands: list[list[str]] = []

    if ("direct", "ltr") in selected:
        commands.append(
            [
                python_cmd,
                str(script_dir / "compare_direct_overlap_results.py"),
                *common,
                "--suite-report",
                "ltr",
            ]
        )
    if ("direct", "prefix_rescue") in selected:
        commands.append(
            [
                python_cmd,
                str(script_dir / "compare_direct_overlap_results.py"),
                *common,
                "--suite-report",
                "native_geometry",
            ]
        )
        commands.append(
            [
                python_cmd,
                str(script_dir / "compare_prefix_rescue_results.py"),
                *common,
            ]
        )
    if ("attention", "attention_ltr") in selected:
        commands.append(
            [
                python_cmd,
                str(script_dir / "compare_attention_results.py"),
                *common,
                "--suite-report",
                "attention_ltr",
            ]
        )
    if ("attention", "attention_native_geometry") in selected:
        commands.append(
            [
                python_cmd,
                str(script_dir / "compare_attention_results.py"),
                *common,
                "--suite-report",
                "attention_native_geometry",
            ]
        )
    return commands


def build_sbatch_command(args: argparse.Namespace, manifest_path: Path) -> list[str]:
    slurm = repo_root() / "src" / "verbatim_eval" / "run_verbatim_suite.slurm"
    export_values = {
        "REPO_ROOT": str(repo_root()),
        "CONFIG_FILE": str(repo_root() / "config.env"),
        "SUITE_MANIFEST": str(manifest_path),
        "RESULTS_ROOT": str(args.results_root),
        "MAX_EXCERPTS": str(args.max_excerpts),
        "MAX_WINDOWS_PER_EXCERPT": str(args.max_windows_per_excerpt),
        "WINDOW_SELECTION": args.window_selection,
        "BATCH_SIZE": str(args.batch_size),
        "ATTENTION_MAX_WINDOWS_PER_EXCERPT": str(args.attention_max_windows_per_excerpt),
        "ATTENTION_BATCH_SIZE": str(args.attention_batch_size),
        "EVAL_KINDS": args.eval_kinds,
        "EXPERIMENTS": args.experiments,
        "NUM_GPU_WORKERS": str(args.num_gpu_workers),
        "GENERATION_MODE": args.generation_mode,
        "WALLTIME_MINUTES": str(args.walltime_minutes),
        "STOP_MARGIN_MINUTES": str(args.stop_margin_minutes),
    }
    if getattr(args, "unique_eval_token_file", None) is not None:
        export_values["UNIQUE_EVAL_TOKEN_FILE"] = str(args.unique_eval_token_file)
    if getattr(args, "replica_buckets_dir", None) is not None:
        export_values["REPLICA_BUCKETS_DIR"] = str(args.replica_buckets_dir)
    if getattr(args, "unique_eval_cache_dir", None) is not None:
        export_values["UNIQUE_EVAL_CACHE_DIR"] = str(args.unique_eval_cache_dir)
    if getattr(args, "use_replica_jsonl_eval", False):
        export_values["USE_REPLICA_JSONL_EVAL"] = "1"
    if getattr(args, "rebuild_unique_eval_cache", False):
        export_values["REBUILD_UNIQUE_EVAL_CACHE"] = "1"
    if getattr(args, "no_fim_model_path", None) is not None:
        export_values["NO_FIM_MODEL_PATH"] = str(args.no_fim_model_path)
    if getattr(args, "fim_v2_model_path", None) is not None:
        export_values["FIM_V2_MODEL_PATH"] = str(args.fim_v2_model_path)
    if getattr(args, "fineweb_only_model_path", None) is not None:
        export_values["FINEWEB_ONLY_MODEL_PATH"] = str(args.fineweb_only_model_path)
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in export_values.items())
    command = [
        "sbatch",
        f"--time={args.time}",
        f"--export={export_arg}",
    ]
    if args.partition:
        command.insert(1, f"--partition={args.partition}")
    command.append(str(slurm))
    return command


def main() -> None:
    args = parse_args()
    repetitions = parse_repetitions(args.repetitions)
    manifest = build_manifest(args.suite, repetitions)
    model_labels = parse_model_labels(args.model_labels, manifest)
    filter_manifest_model_labels(manifest, model_labels)
    eval_kinds = parse_eval_kinds(args.eval_kinds)
    experiments = resolve_experiments(args.experiments, manifest)
    args.eval_kinds = ",".join(eval_kinds)
    args.experiments = ",".join(experiments)
    manifest["defaults"].update(
        {
            "max_excerpts": args.max_excerpts,
            "max_windows_per_excerpt": args.max_windows_per_excerpt,
            "window_selection": args.window_selection,
            "direct_batch_size": args.batch_size,
            "attention_batch_size": args.attention_batch_size,
            "attention_max_windows_per_excerpt": args.attention_max_windows_per_excerpt,
            "eval_kinds": eval_kinds,
            "experiments": experiments,
            "num_gpu_workers": args.num_gpu_workers,
            "generation_mode": args.generation_mode,
            "walltime_minutes": args.walltime_minutes,
            "stop_margin_minutes": args.stop_margin_minutes,
            "eval_source": "replica_jsonl" if args.use_replica_jsonl_eval else "unique_token_cache",
        }
    )
    if model_labels is not None:
        manifest["defaults"]["model_labels"] = model_labels
    if args.no_fim_model_path is not None:
        manifest["defaults"]["no_fim_model_path"] = str(args.no_fim_model_path)
    if args.fim_v2_model_path is not None:
        manifest["defaults"]["fim_v2_model_path"] = str(args.fim_v2_model_path)
    if args.fineweb_only_model_path is not None:
        manifest["defaults"]["fineweb_only_model_path"] = str(args.fineweb_only_model_path)
    if args.unique_eval_token_file is not None:
        manifest["defaults"]["unique_eval_token_file"] = str(args.unique_eval_token_file)
    if args.replica_buckets_dir is not None:
        manifest["defaults"]["replica_buckets_dir"] = str(args.replica_buckets_dir)
    if args.unique_eval_cache_dir is not None:
        manifest["defaults"]["unique_eval_cache_dir"] = str(args.unique_eval_cache_dir)
    manifest_path, tasks_path = write_manifest(args.results_root, manifest)

    cmd = build_sbatch_command(args, manifest_path)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote tasks: {tasks_path}")
    print(f"Suite arms: {manifest['num_arms']}")
    print(f"Repetition buckets: {manifest['num_repetitions']}")
    print(f"Selected experiments: {', '.join(experiments)}")
    print(f"Selected runner tasks: {selected_task_count(manifest, eval_kinds, experiments)}")
    print("Slurm jobs: 1 (no array)")
    print("Submit command:")
    print(" ".join(cmd))

    if not args.dry_run:
        subprocess.run(cmd, check=True)

    collation_commands = build_collation_commands(args, manifest, eval_kinds, experiments)
    if collation_commands:
        print("Collation command(s) after the job finishes:")
        for collation_cmd in collation_commands:
            print(" ".join(collation_cmd))
    else:
        print("Collation command(s) after the job finishes: none; no matching tasks were selected.")


if __name__ == "__main__":
    main()
