#!/usr/bin/env python3
"""Suite manifest and path helpers for verbatim memorization evaluation."""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_REPETITIONS = [1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128]
PREFIX_RESCUE_SPLITS = [(0, 100), (1, 99), (5, 95), (10, 90), (20, 80), (40, 60), (60, 40), (80, 20), (100, 0)]
NATIVE_GEOMETRY_PREFIXES = {0, 1, 5, 10, 20, 40, 60, 80, 100}
CONTEXT_INTERVENTIONS = ["full", "suffix_distractor", "prefix_distractor", "both_distractor"]
ATTENTION_NATIVE_GEOMETRY_SPLITS = [(0, 100), (20, 80), (40, 60), (60, 40), (80, 20), (100, 0)]


@dataclass(frozen=True)
class SuiteArm:
    arm_id: str
    experiment: str
    eval_kind: str
    display_label: str
    model_label: str
    prompt_format: str
    prefix_length: int
    middle_length: int
    suffix_length: int
    window_layout: str
    window_stride: int
    context_intervention: str = "full"
    offset: int = 0

    @property
    def context_budget(self) -> int:
        return self.prefix_length + self.suffix_length


def default_arms() -> list[SuiteArm]:
    arms = [
        SuiteArm(
            arm_id=f"{model_label}_ltr_p100_m32",
            experiment="ltr",
            eval_kind="direct",
            display_label=display_label,
            model_label=model_label,
            prompt_format="ltr_prefix",
            prefix_length=100,
            middle_length=32,
            suffix_length=0,
            window_layout="cooper_nonoverlap",
            window_stride=132,
        )
        for model_label, display_label in [
            ("no_fim", "no-FIM LTR (100L)"),
            ("fim_v2", "FIM-v2 LTR (100L)"),
            ("fineweb_only", "FineWeb-only LTR (100L)"),
        ]
    ]

    for intervention in CONTEXT_INTERVENTIONS:
        for prefix, suffix in PREFIX_RESCUE_SPLITS:
            arms.append(
                SuiteArm(
                    arm_id=f"fim_v2_prefix_rescue_p{prefix}_s{suffix}_{intervention}",
                    experiment="prefix_rescue",
                    eval_kind="direct",
                    display_label=f"FIM-v2 native {prefix}L/{suffix}R ({intervention})",
                    model_label="fim_v2",
                    prompt_format="fim_native",
                    prefix_length=prefix,
                    middle_length=20,
                    suffix_length=suffix,
                    window_layout="matched_context",
                    window_stride=120,
                    context_intervention=intervention,
                )
            )

    arms.extend(
        [
            SuiteArm(
                arm_id=f"attn_{model_label}_ltr_p100_m20",
                experiment="attention_ltr",
                eval_kind="attention",
                display_label=display_label,
                model_label=model_label,
                prompt_format="ltr_prefix",
                prefix_length=100,
                middle_length=20,
                suffix_length=0,
                window_layout="cooper_nonoverlap",
                window_stride=120,
            )
            for model_label, display_label in [
                ("no_fim", "no-FIM LTR attention (100L)"),
                ("fim_v2", "FIM-v2 LTR attention (100L)"),
            ]
        ]
    )

    for prefix, suffix in ATTENTION_NATIVE_GEOMETRY_SPLITS:
        arms.append(
            SuiteArm(
                arm_id=f"attn_fim_v2_native_p{prefix}_s{suffix}",
                experiment="attention_native_geometry",
                eval_kind="attention",
                display_label=f"FIM-v2 native attention {prefix}L/{suffix}R",
                model_label="fim_v2",
                prompt_format="fim_native",
                prefix_length=prefix,
                middle_length=20,
                suffix_length=suffix,
                window_layout="matched_context",
                window_stride=120,
            )
        )
    return arms


def build_manifest(suite_name: str, repetitions: list[int] | None = None) -> dict[str, Any]:
    reps = repetitions or DEFAULT_REPETITIONS
    arms = default_arms()
    return {
        "schema_version": 1,
        "suite_name": suite_name,
        "description": (
            "Core verbatim suite: direct LTR/prefix-rescue extraction plus "
            "attention LTR/native-geometry probes."
        ),
        "repetitions": reps,
        "defaults": {
            "max_excerpts": 0,
            "max_windows_per_excerpt": 5,
            "window_selection": "uniform",
            "sample_seed": 0,
            "prob_extraction_top_k": 40,
            "prob_extraction_temperature": 1.0,
            "prob_extraction_threshold": 0.001,
            "dedupe_excerpts": True,
            "direct_batch_size": 512,
            "attention_batch_size": 16,
            "attention_max_windows_per_excerpt": 4,
            "eval_kinds": ["attention", "direct"],
            "experiments": ["ltr", "prefix_rescue"],
            "generation_mode": "auto",
            "num_gpu_workers": 4,
            "eval_source": "unique_token_cache",
        },
        "arms": [asdict(arm) | {"context_budget": arm.context_budget} for arm in arms],
        "num_arms": len(arms),
        "num_repetitions": len(reps),
        "num_tasks": len(arms) * len(reps),
    }


def suite_root(results_root: Path, suite_name: str) -> Path:
    return results_root / "verbatim_eval" / "suites" / suite_name


def arm_root(results_root: Path, suite_name: str, arm_id: str, repetition: int) -> Path:
    return suite_root(results_root, suite_name) / "arms" / arm_id / f"rep_{repetition}"


def manifest_path(results_root: Path, suite_name: str) -> Path:
    return suite_root(results_root, suite_name) / "manifest.json"


def tasks_path(results_root: Path, suite_name: str) -> Path:
    return suite_root(results_root, suite_name) / "tasks.jsonl"


def write_manifest(results_root: Path, manifest: dict[str, Any]) -> tuple[Path, Path]:
    root = suite_root(results_root, manifest["suite_name"])
    root.mkdir(parents=True, exist_ok=True)
    mpath = root / "manifest.json"
    tpath = root / "tasks.jsonl"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with tpath.open("w", encoding="utf-8") as handle:
        for repetition in manifest["repetitions"]:
            for arm in manifest["arms"]:
                handle.write(json.dumps({"repetition": repetition, "arm": arm}) + "\n")
    return mpath, tpath


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest_for_suite(results_root: Path, suite_name: str) -> dict[str, Any]:
    return load_manifest(manifest_path(results_root, suite_name))


def task_for_index(manifest: dict[str, Any], index: int) -> tuple[int, dict[str, Any]]:
    arms = manifest["arms"]
    repetitions = manifest["repetitions"]
    num_arms = len(arms)
    if index < 0 or index >= len(repetitions) * num_arms:
        raise IndexError(f"Task index {index} is outside 0..{len(repetitions) * num_arms - 1}")
    repetition = repetitions[index // num_arms]
    arm = arms[index % num_arms]
    return int(repetition), arm


def shell_exports_for_task(manifest: dict[str, Any], index: int) -> str:
    repetition, arm = task_for_index(manifest, index)
    values = {
        "SUITE_NAME": manifest["suite_name"],
        "REP": repetition,
        "ARM_ID": arm["arm_id"],
        "EXPERIMENT": arm["experiment"],
        "EVAL_KIND": arm.get("eval_kind", "direct"),
        "MODEL_LABEL": arm["model_label"],
        "PROMPT_FORMAT": arm["prompt_format"],
        "PREFIX_LENGTH": arm["prefix_length"],
        "MIDDLE_LENGTH": arm["middle_length"],
        "SUFFIX_LENGTH": arm["suffix_length"],
        "CONTEXT_BUDGET": arm["context_budget"],
        "CONTEXT_INTERVENTION": arm.get("context_intervention", "full"),
        "WINDOW_LAYOUT": arm["window_layout"],
        "WINDOW_STRIDE": arm["window_stride"],
        "ARM_OFFSET": arm.get("offset", 0),
    }
    return "\n".join(f"export {key}={shlex.quote(str(value))}" for key, value in values.items())


def suite_arms_for_report(manifest: dict[str, Any], report: str) -> list[dict[str, Any]]:
    arms = manifest["arms"]
    if report == "ltr":
        return [arm for arm in arms if arm.get("eval_kind", "direct") == "direct" and arm["experiment"] == "ltr"]
    if report == "native_geometry":
        return [
            arm
            for arm in arms
            if arm.get("eval_kind", "direct") == "direct"
            and arm["experiment"] == "prefix_rescue"
            and arm.get("context_intervention", "full") == "full"
            and int(arm["prefix_length"]) in NATIVE_GEOMETRY_PREFIXES
        ]
    if report == "prefix_rescue":
        return [
            arm
            for arm in arms
            if arm.get("eval_kind", "direct") == "direct" and arm["experiment"] == "prefix_rescue"
        ]
    if report == "attention_ltr":
        return [arm for arm in arms if arm.get("eval_kind") == "attention" and arm["experiment"] == "attention_ltr"]
    if report == "attention_native_geometry":
        return [
            arm
            for arm in arms
            if arm.get("eval_kind") == "attention" and arm["experiment"] == "attention_native_geometry"
        ]
    raise ValueError(f"Unknown suite report: {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect verbatim suite manifests")
    subparsers = parser.add_subparsers(dest="command", required=True)
    task_parser = subparsers.add_parser("task-env")
    task_parser.add_argument("--manifest", type=Path, required=True)
    task_parser.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()

    if args.command == "task-env":
        print(shell_exports_for_task(load_manifest(args.manifest), args.task_index))


if __name__ == "__main__":
    main()
