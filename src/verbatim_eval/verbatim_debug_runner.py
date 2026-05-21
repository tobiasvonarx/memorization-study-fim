#!/usr/bin/env python3
"""Single-node debug runner for selected verbatim-eval suite experiments."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

import direct_overlap_eval as direct
import attention_probe as attention
from unique_eval_cache import ensure_unique_repetition_files
from verbatim_suite import load_manifest, suite_root


SAMPLE_CACHE: dict[int, tuple[list[direct.EvalSample], int, int]] = {}
MODEL_ORDER = ["no_fim", "fim_v2", "fineweb_only"]
EVAL_KIND_ORDER = ["direct", "attention"]
EXPERIMENT_ALIASES = {
    "direct": ["ltr", "prefix_rescue"],
    "direct_core": ["ltr", "prefix_rescue"],
    "attention": ["attention_ltr", "attention_native_geometry"],
    "attention_core": ["attention_ltr", "attention_native_geometry"],
}


@dataclass(frozen=True)
class RunnerTask:
    repetition: int
    arm: dict[str, Any]


@dataclass(frozen=True)
class RunnerConfig:
    manifest_path: str
    results_root: str
    eval_replicas_dir: str
    unique_eval_token_file: str
    replica_buckets_dir: str
    unique_eval_cache_dir: str
    use_replica_jsonl_eval: bool
    rebuild_unique_eval_cache: bool
    no_fim_model_path: str
    fim_v2_model_path: str
    fineweb_only_model_path: str
    max_excerpts: int
    max_windows_per_excerpt: int
    attention_max_windows_per_excerpt: int
    window_selection: str
    sample_seed: int
    batch_size: int
    attention_batch_size: int
    generation_mode: str
    prob_extraction_top_k: int
    prob_extraction_temperature: float
    prob_extraction_threshold: float
    include_token_ids: bool
    include_text: bool
    include_fim_annotations: bool
    include_layer_metrics: bool
    fim_split_mode: str
    dedupe_excerpts: bool
    deadline_monotonic: float
    min_task_start_seconds: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_results_root() -> Path:
    return repo_root() / "results"


def parse_csv(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def parse_eval_kinds(value: str) -> set[str]:
    tokens = parse_csv(value)
    if not tokens or "all" in tokens:
        return set(EVAL_KIND_ORDER)
    kinds = set(tokens)
    unknown = kinds - set(EVAL_KIND_ORDER)
    if unknown:
        raise ValueError(f"Unknown eval kind(s): {', '.join(sorted(unknown))}")
    return kinds


def resolve_experiments(value: str, manifest: dict[str, Any]) -> set[str]:
    available = {str(arm["experiment"]) for arm in manifest["arms"]}
    tokens = parse_csv(value)
    if not tokens or "all" in tokens:
        return available
    selected: set[str] = set()
    for token in tokens:
        selected.update(EXPERIMENT_ALIASES.get(token, [token]))
    unknown = selected - available
    if unknown:
        raise ValueError(f"Unknown experiment(s): {', '.join(sorted(unknown))}")
    return selected


def env_default(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)


def parse_args() -> argparse.Namespace:
    persistent_root = f"/capstor/store/cscs/swissai/infra01/users/{os.environ.get('USER', 'tvonarx')}"
    runs_root = env_default("RUNS_ROOT", f"{persistent_root}/runs")
    dataset_root = env_default("DATASET_ROOT", f"{persistent_root}/datasets")
    parser = argparse.ArgumentParser(description="Run selected verbatim suite experiments on one debug node")
    parser.add_argument("--manifest", type=Path, default=os.environ.get("SUITE_MANIFEST"))
    parser.add_argument("--results-root", type=Path, default=Path(env_default("RESULTS_ROOT", str(default_results_root()))))
    parser.add_argument(
        "--eval-replicas-dir",
        type=Path,
        default=Path(
            env_default(
                "EVAL_REPLICAS_DIR",
                env_default(
                    "GUTENBERG_REPLICAS_V2_DIR",
                    f"{dataset_root}/gutenberg/4096_tokens_filtered_with_replicas_v2",
                ),
            )
        ),
    )
    parser.add_argument(
        "--unique-eval-token-file",
        type=Path,
        default=Path(
            env_default(
                "UNIQUE_EVAL_TOKEN_FILE",
                f"{dataset_root}/gutenberg/4096_tokens_filtered/token.jsonl",
            )
        ),
        help="Final unique, semantically deduped Gutenberg token JSONL used for probing.",
    )
    parser.add_argument(
        "--replica-buckets-dir",
        type=Path,
        default=Path(
            env_default(
                "REPLICA_BUCKETS_DIR",
                env_default(
                    "GUTENBERG_REPLICAS_V2_DIR",
                    f"{dataset_root}/gutenberg/4096_tokens_filtered_with_replicas_v2",
                ),
            )
        ),
        help="Replica-expanded directory used only to recover repetition bucket membership.",
    )
    parser.add_argument(
        "--unique-eval-cache-dir",
        type=Path,
        default=Path(os.environ["UNIQUE_EVAL_CACHE_DIR"]) if "UNIQUE_EVAL_CACHE_DIR" in os.environ else None,
        help="Directory for compact rep_N_token.jsonl files with one row per unique excerpt.",
    )
    parser.add_argument(
        "--use-replica-jsonl-eval",
        action="store_true",
        default=env_default("USE_REPLICA_JSONL_EVAL", "0") == "1",
        help="Legacy mode: evaluate directly from replica-expanded rep_N_token.jsonl files.",
    )
    parser.add_argument(
        "--rebuild-unique-eval-cache",
        action="store_true",
        default=env_default("REBUILD_UNIQUE_EVAL_CACHE", "0") == "1",
        help="Rebuild compact unique eval files even if cache metadata matches.",
    )
    parser.add_argument("--eval-kinds", default=env_default("EVAL_KINDS", "all"))
    parser.add_argument(
        "--experiments",
        default=env_default("EXPERIMENTS", "direct"),
        help=(
            "Comma/space-separated experiment names or aliases. Supported aliases: "
            "direct, direct_core, attention, attention_core, all."
        ),
    )
    parser.add_argument("--max-excerpts", type=int, default=int(env_default("MAX_EXCERPTS", "0")))
    parser.add_argument("--max-windows-per-excerpt", type=int, default=int(env_default("MAX_WINDOWS_PER_EXCERPT", "5")))
    parser.add_argument(
        "--attention-max-windows-per-excerpt",
        type=int,
        default=int(env_default("ATTENTION_MAX_WINDOWS_PER_EXCERPT", "4")),
    )
    parser.add_argument("--window-selection", choices=["first", "uniform"], default=env_default("WINDOW_SELECTION", "uniform"))
    parser.add_argument("--sample-seed", type=int, default=int(env_default("SAMPLE_SEED", "0")))
    parser.add_argument("--batch-size", type=int, default=int(env_default("BATCH_SIZE", "512")))
    parser.add_argument("--attention-batch-size", type=int, default=int(env_default("ATTENTION_BATCH_SIZE", "16")))
    parser.add_argument("--num-gpu-workers", type=int, default=int(env_default("NUM_GPU_WORKERS", "4")))
    parser.add_argument("--gpu-ids", default=env_default("GPU_IDS", "0,1,2,3"))
    parser.add_argument("--generation-mode", choices=["auto", "none", "greedy"], default=env_default("GENERATION_MODE", "auto"))
    parser.add_argument("--walltime-minutes", type=float, default=float(env_default("WALLTIME_MINUTES", "90")))
    parser.add_argument("--stop-margin-minutes", type=float, default=float(env_default("STOP_MARGIN_MINUTES", "5")))
    parser.add_argument(
        "--min-task-start-seconds",
        type=int,
        default=int(env_default("MIN_TASK_START_SECONDS", "300")),
        help="Do not start another arm/repetition task with less than this much guarded time left.",
    )
    parser.add_argument("--prob-extraction-top-k", type=int, default=int(env_default("PROB_EXTRACTION_TOP_K", "40")))
    parser.add_argument(
        "--prob-extraction-temperature",
        type=float,
        default=float(env_default("PROB_EXTRACTION_TEMPERATURE", "1.0")),
    )
    parser.add_argument(
        "--prob-extraction-threshold",
        type=float,
        default=float(env_default("PROB_EXTRACTION_THRESHOLD", "0.001")),
    )
    parser.add_argument("--no-fim-model-path", default=env_default("NO_FIM_MODEL_PATH", f"{runs_root}/no-fim-no-dropout/hf"))
    parser.add_argument("--fim-v2-model-path", default=env_default("FIM_V2_MODEL_PATH", f"{runs_root}/fim-v2-no-dropout/hf"))
    parser.add_argument(
        "--fineweb-only-model-path",
        default=env_default("FINEWEB_ONLY_MODEL_PATH", f"{runs_root}/llama3_2_3B_pretrain_fineweb_only/hf"),
    )
    parser.add_argument("--include-token-ids", action="store_true", default=env_default("INCLUDE_TOKEN_IDS", "0") == "1")
    parser.add_argument("--include-text", action="store_true", default=env_default("INCLUDE_TEXT", "0") == "1")
    parser.add_argument(
        "--include-fim-annotations",
        action="store_true",
        default=env_default("INCLUDE_FIM_ANNOTATIONS", "0") == "1",
    )
    parser.add_argument(
        "--include-layer-metrics",
        action="store_true",
        default=env_default("INCLUDE_LAYER_METRICS", "0") == "1",
    )
    parser.add_argument("--fim-split-mode", choices=["fixed_by_excerpt", "replica_aware"], default=env_default("FIM_SPLIT_MODE", "replica_aware"))
    parser.add_argument("--no-dedupe-excerpts", dest="dedupe_excerpts", action="store_false", default=True)
    parser.add_argument("--dry-run", action="store_true", default=env_default("DRY_RUN", "0") == "1")
    args = parser.parse_args()
    if args.manifest is None:
        raise ValueError("--manifest or SUITE_MANIFEST is required")
    args.manifest = Path(args.manifest)
    if args.unique_eval_cache_dir is None:
        args.unique_eval_cache_dir = Path(args.results_root) / "verbatim_eval" / "cache" / "gutenberg_unique_repetition_buckets"
    if args.num_gpu_workers <= 0:
        raise ValueError("--num-gpu-workers must be positive")
    return args


def model_paths(config: RunnerConfig) -> dict[str, str]:
    return {
        "no_fim": config.no_fim_model_path,
        "fim_v2": config.fim_v2_model_path,
        "fineweb_only": config.fineweb_only_model_path,
    }


def task_dataset(config: RunnerConfig, repetition: int) -> Path:
    if not config.use_replica_jsonl_eval:
        return Path(config.unique_eval_cache_dir) / f"rep_{repetition}_token.jsonl"
    return Path(config.eval_replicas_dir) / f"rep_{repetition}_token.jsonl"


def build_eval_args(task: RunnerTask, config: RunnerConfig) -> argparse.Namespace:
    arm = task.arm
    eval_kind = str(arm.get("eval_kind", "direct"))
    generation_mode = config.generation_mode
    if generation_mode == "auto":
        generation_mode = "greedy" if eval_kind == "direct" and str(arm["experiment"]) == "ltr" else "none"
    return argparse.Namespace(
        dataset=str(task_dataset(config, task.repetition)),
        model_path=model_paths(config)[str(arm["model_label"])],
        model_label=str(arm["model_label"]),
        eval_kind=eval_kind,
        repetition=int(task.repetition),
        prompt_format=str(arm["prompt_format"]),
        study_name=str(arm.get("suite_name", "")) or "verbatim_debug",
        suite_name=None,
        arm_id=None,
        experiment=arm.get("experiment"),
        prefix_length=int(arm["prefix_length"]),
        middle_length=int(arm["middle_length"]),
        suffix_length=int(arm["suffix_length"]),
        context_budget=int(arm["context_budget"]),
        offset=int(arm.get("offset", 0)),
        window_stride=int(arm["window_stride"]),
        window_layout=str(arm["window_layout"]),
        max_windows_per_excerpt=(
            config.attention_max_windows_per_excerpt
            if eval_kind == "attention"
            else config.max_windows_per_excerpt
        ),
        window_selection=config.window_selection,
        max_excerpts=config.max_excerpts,
        max_samples=256,
        sample_seed=config.sample_seed,
        fim_train_split_seed=42,
        fim_train_content_length=4096,
        fim_split_mode=config.fim_split_mode,
        include_fim_annotations=config.include_fim_annotations,
        dedupe_excerpts=config.dedupe_excerpts,
        results_root=config.results_root,
        device="cuda",
        batch_size=config.attention_batch_size if eval_kind == "attention" else config.batch_size,
        generation_mode=generation_mode,
        prob_extraction_top_k=config.prob_extraction_top_k,
        prob_extraction_temperature=config.prob_extraction_temperature,
        prob_extraction_threshold=config.prob_extraction_threshold,
        shard_rank=0,
        num_shards=1,
        merge_shards=False,
        expected_shards=None,
        include_token_ids=config.include_token_ids,
        include_text=config.include_text,
        include_layer_metrics=config.include_layer_metrics,
        context_intervention=str(arm.get("context_intervention", "full")),
    )


def attach_suite_output(args: argparse.Namespace, suite_name: str, arm_id: str) -> argparse.Namespace:
    args.suite_name = suite_name
    args.arm_id = arm_id
    args.study_name = suite_name
    return args


def expected_max_excerpts(args: argparse.Namespace) -> int:
    return args.max_excerpts if args.max_excerpts is not None else args.max_samples


def task_output_paths(eval_args: argparse.Namespace) -> dict[str, Path]:
    if eval_args.eval_kind == "attention":
        return attention.output_paths(eval_args, None)
    return direct.output_paths(eval_args, None)


def task_complete(eval_args: argparse.Namespace) -> bool:
    paths = task_output_paths(eval_args)
    if not paths["jsonl"].exists() or not paths["summary"].exists():
        return False
    try:
        with paths["summary"].open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "dataset": str(Path(eval_args.dataset)),
        "suite_name": eval_args.suite_name,
        "arm_id": eval_args.arm_id,
        "experiment": eval_args.experiment,
        "repetition": eval_args.repetition,
        "max_excerpts": expected_max_excerpts(eval_args),
        "max_windows_per_excerpt": eval_args.max_windows_per_excerpt,
        "window_selection": eval_args.window_selection,
        "sample_seed": eval_args.sample_seed,
    }
    if eval_args.eval_kind == "direct":
        expected["generation_mode"] = eval_args.generation_mode
    else:
        expected["include_layer_metrics"] = eval_args.include_layer_metrics
    if not all(summary.get(key) == value for key, value in expected.items()):
        return False
    return summary.get("num_windows") == summary.get("num_selected_windows")


def build_tasks(manifest: dict[str, Any], eval_kinds: set[str], experiments: set[str]) -> list[RunnerTask]:
    tasks: list[RunnerTask] = []
    for repetition in manifest["repetitions"]:
        for arm in manifest["arms"]:
            if str(arm.get("eval_kind", "direct")) not in eval_kinds:
                continue
            if str(arm["experiment"]) not in experiments:
                continue
            tasks.append(RunnerTask(repetition=int(repetition), arm=arm))
    return tasks


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def gib(num_bytes: int | float) -> float:
    return float(num_bytes) / float(1024**3)


def reset_gpu_memory_peaks(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except RuntimeError:
        return


def log_gpu_memory(
    stage: str,
    gpu_id: int,
    device: torch.device,
    eval_kind: str,
    model_label: str,
    *,
    arm_id: str | None = None,
    repetition: int | None = None,
    batch_size: int | None = None,
) -> None:
    """Emit one parseable memory snapshot to Slurm stderr."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    device_index = device.index if device.index is not None else gpu_id
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        payload = {
            "event": "gpu_memory",
            "stage": stage,
            "gpu_id": gpu_id,
            "device": str(device),
            "eval_kind": eval_kind,
            "model_label": model_label,
            "arm_id": arm_id,
            "repetition": repetition,
            "batch_size": batch_size,
            "allocated_gib": round(gib(torch.cuda.memory_allocated(device_index)), 3),
            "reserved_gib": round(gib(torch.cuda.memory_reserved(device_index)), 3),
            "peak_allocated_gib": round(gib(torch.cuda.max_memory_allocated(device_index)), 3),
            "peak_reserved_gib": round(gib(torch.cuda.max_memory_reserved(device_index)), 3),
            "free_gib": round(gib(free_bytes), 3),
            "total_gib": round(gib(total_bytes), 3),
        }
        print("GPU_MEMORY " + json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
    except RuntimeError as exc:
        print(
            f"GPU_MEMORY_FAILED stage={stage} gpu_id={gpu_id} device={device} error={exc!r}",
            file=sys.stderr,
            flush=True,
        )


def preload_samples(config: RunnerConfig, repetitions: list[int], status_path: Path, status: dict[str, Any]) -> None:
    for repetition in repetitions:
        dataset_path = task_dataset(config, repetition)
        print(f"Loading samples for rep={repetition}: {dataset_path}", flush=True)
        SAMPLE_CACHE[repetition] = direct.load_samples(
            dataset_path,
            config.dedupe_excerpts,
            repetition=repetition,
            include_replica_metadata=config.include_fim_annotations,
        )
        samples, raw_rows, rows_after_dedupe = SAMPLE_CACHE[repetition]
        status.setdefault("sample_cache", {})[str(repetition)] = {
            "dataset": str(dataset_path),
            "raw_num_rows": raw_rows,
            "num_rows_after_dedupe": rows_after_dedupe,
            "num_cached_samples": len(samples),
        }
        write_status(status_path, status)


def worker_main(
    gpu_id: int,
    eval_kind: str,
    model_label: str,
    config: RunnerConfig,
    suite_name: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
        model_arg = argparse.Namespace(model_path=model_paths(config)[model_label])
        print(f"GPU worker {gpu_id}: loading {eval_kind}/{model_label} on {device}", flush=True)
        reset_gpu_memory_peaks(device)
        if eval_kind == "attention":
            model = attention.load_attention_model(model_arg, device)
        else:
            model = direct.load_model(model_arg, device)
        log_gpu_memory(
            "after_model_load",
            gpu_id,
            device,
            eval_kind,
            model_label,
            batch_size=config.attention_batch_size if eval_kind == "attention" else config.batch_size,
        )
        tokenizer = (
            AutoTokenizer.from_pretrained(model_arg.model_path, trust_remote_code=True)
            if eval_kind == "direct" and config.include_text
            else None
        )
        while True:
            if time.monotonic() + config.min_task_start_seconds >= config.deadline_monotonic:
                result_queue.put({"status": "time_guard", "gpu_id": gpu_id, "eval_kind": eval_kind, "model_label": model_label})
                return
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                result_queue.put({"status": "worker_done", "gpu_id": gpu_id, "eval_kind": eval_kind, "model_label": model_label})
                return
            eval_args = attach_suite_output(
                build_eval_args(task, config),
                suite_name=suite_name,
                arm_id=str(task.arm["arm_id"]),
            )
            try:
                samples, raw_rows, rows_after_dedupe = SAMPLE_CACHE[task.repetition]
                context_budget = direct.validate_args(eval_args)
                reset_gpu_memory_peaks(device)
                if eval_kind == "attention":
                    result = attention.evaluate_attention_task(
                        args=eval_args,
                        context_budget=context_budget,
                        model=model,
                        device=device,
                        raw_samples=samples,
                        raw_num_rows=raw_rows,
                        rows_after_dedupe=rows_after_dedupe,
                        show_progress=False,
                        deadline_monotonic=config.deadline_monotonic,
                    )
                else:
                    result = direct.evaluate_direct_task(
                        args=eval_args,
                        context_budget=context_budget,
                        model=model,
                        device=device,
                        tokenizer=tokenizer,
                        raw_samples=samples,
                        raw_num_rows=raw_rows,
                        rows_after_dedupe=rows_after_dedupe,
                        show_progress=False,
                        deadline_monotonic=config.deadline_monotonic,
                    )
                log_gpu_memory(
                    "after_task",
                    gpu_id,
                    device,
                    eval_kind,
                    model_label,
                    arm_id=eval_args.arm_id,
                    repetition=task.repetition,
                    batch_size=int(eval_args.batch_size),
                )
                result_queue.put(
                    {
                        "status": "completed",
                        "gpu_id": gpu_id,
                        "eval_kind": eval_kind,
                        "model_label": model_label,
                        "arm_id": eval_args.arm_id,
                        "repetition": task.repetition,
                        "num_windows": result["num_windows"],
                    }
                )
            except direct.DirectEvalTimeGuard as exc:
                result_queue.put(
                    {
                        "status": "time_guard",
                        "gpu_id": gpu_id,
                        "eval_kind": eval_kind,
                        "model_label": model_label,
                        "arm_id": eval_args.arm_id,
                        "repetition": task.repetition,
                        "message": str(exc),
                    }
                )
                return
            except Exception as exc:  # noqa: BLE001 - report and continue with remaining tasks.
                result_queue.put(
                    {
                        "status": "failed",
                        "gpu_id": gpu_id,
                        "eval_kind": eval_kind,
                        "model_label": model_label,
                        "arm_id": eval_args.arm_id,
                        "repetition": task.repetition,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    except Exception as exc:  # noqa: BLE001 - worker bootstrap failures must reach the parent.
        result_queue.put(
            {
                "status": "worker_failed",
                "gpu_id": gpu_id,
                "eval_kind": eval_kind,
                "model_label": model_label,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )


def run_model_group(
    eval_kind: str,
    model_label: str,
    tasks: list[RunnerTask],
    gpu_ids: list[int],
    config: RunnerConfig,
    suite_name: str,
    status_path: Path,
    status: dict[str, Any],
) -> bool:
    if not tasks:
        return False
    ctx = mp.get_context("fork")
    task_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()
    for task in tasks:
        task_queue.put(task)

    worker_count = min(len(gpu_ids), len(tasks))
    processes = [
        ctx.Process(
            target=worker_main,
            args=(gpu_ids[index], eval_kind, model_label, config, suite_name, task_queue, result_queue),
        )
        for index in range(worker_count)
    ]
    for process in processes:
        process.start()

    remaining_workers = worker_count
    hit_time_guard = False
    while remaining_workers:
        message = result_queue.get()
        message_status = str(message.get("status"))
        status.setdefault("events", []).append(message)
        if message_status in {"worker_done", "time_guard", "worker_failed"}:
            remaining_workers -= 1
        if message_status == "completed":
            status["completed_tasks"] = int(status.get("completed_tasks", 0)) + 1
        elif message_status == "failed":
            status["failed_tasks"] = int(status.get("failed_tasks", 0)) + 1
        elif message_status == "time_guard":
            hit_time_guard = True
        elif message_status == "worker_failed":
            status["failed_tasks"] = int(status.get("failed_tasks", 0)) + 1
        write_status(status_path, status)

    for process in processes:
        process.join()
        if process.exitcode not in {0, None}:
            status["failed_tasks"] = int(status.get("failed_tasks", 0)) + 1
            status.setdefault("events", []).append(
                {
                    "status": "worker_exit_nonzero",
                    "eval_kind": eval_kind,
                    "model_label": model_label,
                    "exitcode": process.exitcode,
                }
            )
    write_status(status_path, status)
    return hit_time_guard


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    eval_kinds = parse_eval_kinds(args.eval_kinds)
    experiments = resolve_experiments(args.experiments, manifest)
    gpu_ids = [int(value) for value in parse_csv(args.gpu_ids)][: args.num_gpu_workers]
    deadline = time.monotonic() + max(0.0, args.walltime_minutes - args.stop_margin_minutes) * 60.0
    config = RunnerConfig(
        manifest_path=str(args.manifest),
        results_root=str(args.results_root),
        eval_replicas_dir=str(args.eval_replicas_dir),
        unique_eval_token_file=str(args.unique_eval_token_file),
        replica_buckets_dir=str(args.replica_buckets_dir),
        unique_eval_cache_dir=str(args.unique_eval_cache_dir),
        use_replica_jsonl_eval=args.use_replica_jsonl_eval,
        rebuild_unique_eval_cache=args.rebuild_unique_eval_cache,
        no_fim_model_path=args.no_fim_model_path,
        fim_v2_model_path=args.fim_v2_model_path,
        fineweb_only_model_path=args.fineweb_only_model_path,
        max_excerpts=args.max_excerpts,
        max_windows_per_excerpt=args.max_windows_per_excerpt,
        attention_max_windows_per_excerpt=args.attention_max_windows_per_excerpt,
        window_selection=args.window_selection,
        sample_seed=args.sample_seed,
        batch_size=args.batch_size,
        attention_batch_size=args.attention_batch_size,
        generation_mode=args.generation_mode,
        prob_extraction_top_k=args.prob_extraction_top_k,
        prob_extraction_temperature=args.prob_extraction_temperature,
        prob_extraction_threshold=args.prob_extraction_threshold,
        include_token_ids=args.include_token_ids,
        include_text=args.include_text,
        include_fim_annotations=args.include_fim_annotations,
        include_layer_metrics=args.include_layer_metrics,
        fim_split_mode=args.fim_split_mode,
        dedupe_excerpts=args.dedupe_excerpts,
        deadline_monotonic=deadline,
        min_task_start_seconds=args.min_task_start_seconds,
    )
    all_tasks = build_tasks(manifest, eval_kinds, experiments)
    suite_name = str(manifest["suite_name"])
    pending: list[RunnerTask] = []
    skipped = 0
    for task in all_tasks:
        eval_args = attach_suite_output(
            build_eval_args(task, config),
            suite_name=suite_name,
            arm_id=str(task.arm["arm_id"]),
        )
        if task_complete(eval_args):
            skipped += 1
        else:
            pending.append(task)

    status_path = suite_root(args.results_root, suite_name) / "debug_run_status.json"
    status: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "suite_name": suite_name,
        "config": asdict(config) | {
            "gpu_ids": gpu_ids,
            "eval_kinds": sorted(eval_kinds),
            "experiments": sorted(experiments),
        },
        "total_tasks": len(all_tasks),
        "skipped_completed_tasks": skipped,
        "pending_tasks_at_start": len(pending),
        "completed_tasks": 0,
        "failed_tasks": 0,
        "events": [],
    }
    write_status(status_path, status)

    print(f"Suite: {suite_name}")
    print(f"Eval kinds: {sorted(eval_kinds)}")
    print(f"Experiments: {sorted(experiments)}")
    print(f"Tasks: {len(all_tasks)} total, {skipped} already complete, {len(pending)} pending")
    print(f"GPU workers: {gpu_ids}")
    print(f"Generation mode: {args.generation_mode}")
    if args.use_replica_jsonl_eval:
        print(f"Eval source: replica-expanded JSONLs from {args.eval_replicas_dir}")
    else:
        print(f"Eval source: compact unique cache at {args.unique_eval_cache_dir}")
        print(f"Unique token source: {args.unique_eval_token_file}")
        print(f"Bucket source: {args.replica_buckets_dir}")
    if args.dry_run or not pending:
        return

    preload_reps = sorted({task.repetition for task in pending})
    if not args.use_replica_jsonl_eval:
        print(f"Preparing compact unique eval cache for repetitions: {preload_reps}", flush=True)
        cache_results = ensure_unique_repetition_files(
            args.unique_eval_token_file,
            args.replica_buckets_dir,
            args.unique_eval_cache_dir,
            preload_reps,
            force=args.rebuild_unique_eval_cache,
        )
        status["unique_eval_cache"] = [
            {
                "repetition": result.repetition,
                "path": str(result.path),
                "status": result.status,
                "raw_replica_rows": result.raw_replica_rows,
                "num_assignments": result.num_assignments,
                "num_written": result.num_written,
            }
            for result in cache_results
        ]
        write_status(status_path, status)
        num_cached = sum(1 for result in cache_results if result.status == "cached")
        num_built = sum(1 for result in cache_results if result.status == "built")
        print(f"Unique eval cache ready: {num_cached} cached, {num_built} built", flush=True)
    preload_samples(config, preload_reps, status_path, status)

    group_order = [
        (eval_kind, model_label)
        for eval_kind in EVAL_KIND_ORDER
        for model_label in MODEL_ORDER
    ]
    pending_by_group = {
        (eval_kind, model_label): [
            task
            for task in pending
            if str(task.arm.get("eval_kind", "direct")) == eval_kind
            and str(task.arm["model_label"]) == model_label
        ]
        for eval_kind, model_label in group_order
    }
    hit_time_guard = False
    for eval_kind, model_label in group_order:
        tasks = pending_by_group[(eval_kind, model_label)]
        if not tasks:
            continue
        if time.monotonic() + args.min_task_start_seconds >= deadline:
            hit_time_guard = True
            break
        print(f"Running {len(tasks)} pending tasks for {eval_kind}/{model_label}", flush=True)
        if run_model_group(eval_kind, model_label, tasks, gpu_ids, config, suite_name, status_path, status):
            hit_time_guard = True
            break

    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    status["hit_time_guard"] = hit_time_guard
    status["remaining_estimate"] = max(0, len(pending) - int(status.get("completed_tasks", 0)))
    write_status(status_path, status)
    if int(status.get("failed_tasks", 0)) > 0:
        raise SystemExit(1)
    if hit_time_guard:
        print("Stopped cleanly at the debug time guard; resubmit to continue pending tasks.")


if __name__ == "__main__":
    main()
