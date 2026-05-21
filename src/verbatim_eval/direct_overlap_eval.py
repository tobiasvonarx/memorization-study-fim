#!/usr/bin/env python3
"""
Batched overlap-style memorization diagnostics from HF checkpoints.

The primary paper-aligned protocol evaluates FIM and non-FIM checkpoints on the
same LTR-prefix target windows. This script keeps that matched window substrate
but avoids the previous one-window-at-a-time generation/loss loop:

* reference target likelihood is computed in batched teacher-forced forwards
* greedy continuations are decoded in batches using the model KV cache
* generated-token likelihood is collected from the decode logits
* compact per-window JSONL rows are written by default
* multi-GPU Slurm runs shard windows by stable global_window_id
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from verbatim_suite import arm_root


FIM_PREFIX = 128002
FIM_MIDDLE = 128003
FIM_SUFFIX = 128005
BOS_TOKEN = 128000
EOS_TOKEN = 128001
CONTEXT_INTERVENTIONS = {
    "full",
    "suffix_distractor",
    "prefix_distractor",
    "both_distractor",
}
COOPER_PREFIX_TARGET_LENGTHS = (20, 30, 32)
COOPER_SPAN_METRIC_STEMS = (
    "cooper_log_p_z",
    "cooper_log10_p_z",
    "cooper_p_z",
    "cooper_mean_log_p_z",
    "cooper_mean_log10_p_z",
    "cooper_token_geomean_p_z",
    "cooper_extractable",
    "cooper_supported_token_rate",
    "cooper_all_tokens_in_topk",
)


def parse_cooper_prefix_target_lengths(value: str) -> tuple[int, ...]:
    lengths: list[int] = []
    seen: set[int] = set()
    for token in value.replace(",", " ").split():
        length = int(token)
        if length <= 0:
            raise ValueError("Cooper prefix target lengths must be positive integers")
        if length in seen:
            continue
        seen.add(length)
        lengths.append(length)
    if not lengths:
        raise ValueError("At least one Cooper prefix target length is required")
    return tuple(lengths)


def cooper_prefix_target_lengths(args: argparse.Namespace | None = None) -> tuple[int, ...]:
    if args is not None and hasattr(args, "cooper_prefix_target_lengths"):
        return tuple(int(length) for length in args.cooper_prefix_target_lengths)
    return COOPER_PREFIX_TARGET_LENGTHS


def metric_keys_for_spans(span_lengths: Iterable[int]) -> list[str]:
    keys = [
        "NLL",
        "NLL_sum",
        "PPL",
        "Ref_NLL",
        "Ref_NLL_sum",
        "Ref_PPL",
        "log_p_target",
        "log_p_generated",
        "memorization_score",
        "token_accuracy",
        "Rouge-L",
        "LCS",
        "TTR_ref",
        "TTR_gen",
        "exact_match",
        "greedy_exact_match",
        "cooper_log_p_z",
        "cooper_log10_p_z",
        "cooper_p_z",
        "cooper_mean_log_p_z",
        "cooper_mean_log10_p_z",
        "cooper_token_geomean_p_z",
        "cooper_extractable",
        "cooper_supported_token_rate",
        "cooper_all_tokens_in_topk",
        "match_ge_0_75",
        "match_ge_0_50",
        "match_ge_0_25",
    ]
    keys.extend(
        f"{stem}_first{span_length}"
        for span_length in span_lengths
        for stem in COOPER_SPAN_METRIC_STEMS
    )
    return keys


class DirectEvalTimeGuard(RuntimeError):
    """Raised when a long-running direct eval should stop before walltime."""

METRIC_KEYS = metric_keys_for_spans(COOPER_PREFIX_TARGET_LENGTHS)


@dataclass(frozen=True)
class EvalSample:
    excerpt_id: str
    sample_index: int
    token_ids: list[int]
    replica_metadata: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class EvalWindow:
    global_window_id: int
    excerpt_id: str
    sample_index: int
    window_index: int
    target_start: int
    candidate_window_count: int
    prefix: list[int]
    middle: list[int]
    suffix: list[int]
    true_prefix: list[int]
    true_suffix: list[int]
    context_intervention: str
    prefix_is_distractor: bool
    suffix_is_distractor: bool
    distractor_excerpt_id: str | None
    distractor_sample_index: int | None
    fim_annotation: dict[str, Any]


@dataclass(frozen=True)
class WindowBuildResult:
    windows: list[EvalWindow]
    raw_num_rows: int
    num_rows_after_dedupe: int
    num_excerpts: int
    num_excerpts_with_windows: int
    num_short_excerpts: int
    num_candidate_windows: int
    num_selected_windows: int


class MetricAccumulator:
    def __init__(self, metric_keys: Iterable[str] | None = None) -> None:
        self.metric_keys = list(metric_keys) if metric_keys is not None else list(METRIC_KEYS)
        self.values: dict[str, list[float]] = {key: [] for key in self.metric_keys}

    def update(self, row: dict[str, Any]) -> None:
        for key in self.metric_keys:
            if key in row:
                self.values[key].append(float(row[key]))

    def summary(self) -> dict[str, dict[str, float]]:
        return {key: metric_mean_std(values) for key, values in self.values.items()}

    def count(self, key: str = "Ref_NLL") -> int:
        return len(self.values.get(key, []))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched direct overlap evaluation from HF checkpoints")
    parser.add_argument("--dataset", required=True, help="Path to rep_N_token.jsonl")
    parser.add_argument("--model-path", required=True, help="HF checkpoint path")
    parser.add_argument("--model-label", required=True, help="Label used in output paths")
    parser.add_argument("--repetition", required=True, type=int, help="Repetition bucket")
    parser.add_argument(
        "--prompt-format",
        choices=["ltr_prefix", "fim_native"],
        required=True,
        help="Prompt construction strategy",
    )
    parser.add_argument(
        "--study-name",
        default="paper_aligned_prefix",
        help="Subtree below the results root",
    )
    parser.add_argument("--suite-name", default=None, help="Unified suite name for suite-mode outputs")
    parser.add_argument("--arm-id", default=None, help="Unified suite arm id for suite-mode outputs")
    parser.add_argument("--experiment", default=None, help="Experiment label recorded in suite-mode summaries")
    parser.add_argument("--prefix-length", type=int, default=50)
    parser.add_argument("--middle-length", type=int, default=50)
    parser.add_argument("--suffix-length", type=int, default=0)
    parser.add_argument(
        "--context-budget",
        type=int,
        default=None,
        help="Optional guard: require prefix_length + suffix_length to equal this value.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "Target-window phase. For matched_context the first target starts after "
            "the observed context budget; for Cooper layouts it starts after the prefix."
        ),
    )
    parser.add_argument("--window-stride", type=int, default=50)
    parser.add_argument(
        "--window-layout",
        choices=["matched_context", "cooper_nonoverlap", "cooper_sliding"],
        default="matched_context",
        help=(
            "matched_context preserves the old shared LTR/FIM target grid by reserving the "
            "full context budget on both sides. cooper_nonoverlap uses prefix+target "
            "examples with stride = prefix_length + middle_length. cooper_sliding uses "
            "the same prefix-target construction with the requested stride."
        ),
    )
    parser.add_argument(
        "--max-windows-per-excerpt",
        type=int,
        default=0,
        help="Maximum selected windows per excerpt. Use 0 for all candidate windows.",
    )
    parser.add_argument(
        "--window-selection",
        choices=["first", "uniform"],
        default="uniform",
        help="How to choose windows when max-windows-per-excerpt limits candidates.",
    )
    parser.add_argument(
        "--max-excerpts",
        type=int,
        default=None,
        help="Maximum unique excerpts to evaluate. Use 0 for all excerpts.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=256,
        help="Backward-compatible alias for --max-excerpts when --max-excerpts is omitted.",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--fim-train-split-seed",
        type=int,
        default=42,
        help="Seed used by Gutenberg FIM training preparation; used only for split annotations.",
    )
    parser.add_argument(
        "--fim-train-content-length",
        type=int,
        default=4096,
        help="Content-token cap used by Gutenberg FIM training preparation; used only for split annotations.",
    )
    parser.add_argument(
        "--fim-split-mode",
        choices=["fixed_by_excerpt", "replica_aware"],
        default="fixed_by_excerpt",
        help=(
            "How to reconstruct Gutenberg FIM training split annotations. "
            "Use replica_aware for FIM-v2 datasets whose FIM split RNG depended on replica_index."
        ),
    )
    parser.add_argument(
        "--include-fim-annotations",
        action="store_true",
        help=(
            "Include reconstructed training FIM split overlap annotations in per-window rows. "
            "Disabled by default because the maintained FIM-v2 experiments use randomized "
            "splits across repetitions."
        ),
    )
    dedupe_group = parser.add_mutually_exclusive_group()
    dedupe_group.add_argument(
        "--dedupe-excerpts",
        dest="dedupe_excerpts",
        action="store_true",
        default=True,
        help="Keep only one record per excerpt_id before selecting excerpts.",
    )
    dedupe_group.add_argument(
        "--no-dedupe-excerpts",
        dest="dedupe_excerpts",
        action="store_false",
        help="Evaluate physical rows from the replica file, including repeated excerpt_id values.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Defaults to <repo>/results",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--generation-mode",
        choices=["none", "greedy"],
        default="greedy",
        help="Use 'none' to skip greedy decoding and write only teacher-forced extraction metrics.",
    )
    parser.add_argument(
        "--prob-extraction-top-k",
        type=int,
        default=40,
        help="Top-k truncation for Cooper-style probabilistic extraction.",
    )
    parser.add_argument(
        "--prob-extraction-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for Cooper-style probabilistic extraction.",
    )
    parser.add_argument(
        "--prob-extraction-threshold",
        type=float,
        default=0.001,
        help="p_z threshold for Cooper-style extractability rate.",
    )
    parser.add_argument(
        "--cooper-prefix-target-lengths",
        type=parse_cooper_prefix_target_lengths,
        default=COOPER_PREFIX_TARGET_LENGTHS,
        help=(
            "Comma/space-separated target-prefix lengths for auxiliary Cooper p(z) summaries. "
            "Example: '20,30,40,50'."
        ),
    )
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge shard JSONL outputs and write final summary without loading a model.",
    )
    parser.add_argument(
        "--expected-shards",
        type=int,
        default=None,
        help="Number of shard files expected by --merge-shards. Defaults to --num-shards.",
    )
    parser.add_argument(
        "--include-token-ids",
        action="store_true",
        help="Include prefix/suffix/target/generated token arrays in each row.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Decode and include prefix/suffix/target/generated text in each row.",
    )
    parser.add_argument(
        "--context-intervention",
        choices=sorted(CONTEXT_INTERVENTIONS),
        default="full",
        help=(
            "Prompt-context intervention for native FIM probes. 'full' uses the true "
            "prefix and suffix. The distractor variants replace the requested context "
            "side with same-length text from another selected excerpt while keeping the "
            "target window fixed."
        ),
    )
    return parser.parse_args()


def default_results_root() -> Path:
    if os.environ.get("VERBATIM_EVAL_RESULTS_ROOT"):
        return Path(os.environ["VERBATIM_EVAL_RESULTS_ROOT"])
    if os.environ.get("RESULTS_ROOT"):
        return Path(os.environ["RESULTS_ROOT"])
    return Path(__file__).resolve().parents[2] / "results"


def resolve_device(arg_device: str) -> torch.device:
    if arg_device == "cpu":
        return torch.device("cpu")
    if arg_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_args(args: argparse.Namespace) -> int:
    context_budget = args.prefix_length + args.suffix_length
    if args.prompt_format == "ltr_prefix" and args.suffix_length != 0:
        raise ValueError("ltr_prefix format requires --suffix-length 0")
    if args.prompt_format == "ltr_prefix" and args.prefix_length <= 0:
        raise ValueError("ltr_prefix format requires --prefix-length > 0")
    if min(args.prefix_length, args.middle_length, args.suffix_length, args.offset) < 0:
        raise ValueError("prefix, middle, suffix, and offset values must be non-negative")
    if args.middle_length <= 0:
        raise ValueError("--middle-length must be positive")
    if args.window_stride <= 0:
        raise ValueError("--window-stride must be positive")
    if args.max_windows_per_excerpt < 0:
        raise ValueError("--max-windows-per-excerpt must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_rank < 0 or args.shard_rank >= args.num_shards:
        raise ValueError("--shard-rank must satisfy 0 <= shard_rank < num_shards")
    if args.expected_shards is not None and args.expected_shards <= 0:
        raise ValueError("--expected-shards must be positive")
    if args.prob_extraction_top_k <= 0:
        raise ValueError("--prob-extraction-top-k must be positive")
    if args.prob_extraction_temperature <= 0.0:
        raise ValueError("--prob-extraction-temperature must be positive")
    if not (0.0 < args.prob_extraction_threshold <= 1.0):
        raise ValueError("--prob-extraction-threshold must be in (0, 1]")
    if args.window_layout.startswith("cooper_"):
        if args.prompt_format != "ltr_prefix":
            raise ValueError("Cooper-style window layouts are prefix-only and require --prompt-format ltr_prefix")
        if args.suffix_length != 0:
            raise ValueError("Cooper-style window layouts require --suffix-length 0")
        if args.window_layout == "cooper_nonoverlap":
            expected_stride = args.prefix_length + args.middle_length
            if args.window_stride != expected_stride:
                raise ValueError(
                    "cooper_nonoverlap requires --window-stride to equal "
                    f"prefix_length + middle_length = {expected_stride}"
                )
    if args.context_budget is not None and args.context_budget != context_budget:
        raise ValueError(
            f"--context-budget={args.context_budget} does not match "
            f"prefix_length + suffix_length = {context_budget}"
        )
    if args.context_intervention != "full" and args.prompt_format != "fim_native":
        raise ValueError("--context-intervention is only supported for --prompt-format fim_native")
    return context_budget


def replica_metadata_from_row(obj: dict[str, Any], sample_index: int, fallback_repetition: int | None = None) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "excerpt_id": str(obj.get("excerpt_id", sample_index)),
        "base_excerpt_id": obj.get("base_excerpt_id", obj.get("excerpt_id", sample_index)),
        "repetition_bucket": obj.get("repetition_bucket", fallback_repetition),
        "replica_count": obj.get("replica_count", fallback_repetition),
        "replica_index": obj.get("replica_index"),
        "bucket_base_index": obj.get("bucket_base_index"),
    }


def replica_metadata_items_from_row(
    obj: dict[str, Any],
    sample_index: int,
    fallback_repetition: int | None = None,
    expand_compact: bool = False,
) -> list[dict[str, Any]]:
    explicit = obj.get("replica_metadata")
    base = replica_metadata_from_row(obj, sample_index, fallback_repetition=fallback_repetition)
    if not isinstance(explicit, list):
        if expand_compact and obj.get("unique_eval_compact") and isinstance(base.get("replica_count"), int):
            return [
                base
                | {
                    "replica_index": replica_index,
                }
                for replica_index in range(int(base["replica_count"]))
            ]
        return [base]

    items: list[dict[str, Any]] = []
    for item in explicit:
        if not isinstance(item, dict):
            continue
        metadata = dict(base)
        metadata.update(item)
        metadata["sample_index"] = sample_index
        metadata["excerpt_id"] = str(metadata.get("excerpt_id", obj.get("excerpt_id", sample_index)))
        metadata["base_excerpt_id"] = metadata.get("base_excerpt_id", obj.get("base_excerpt_id", metadata["excerpt_id"]))
        items.append(metadata)
    return items or [base]


def load_samples(
    dataset_path: Path,
    dedupe: bool,
    repetition: int | None = None,
    include_replica_metadata: bool = True,
) -> tuple[list[EvalSample], int, int]:
    raw_count = 0
    samples: list[EvalSample] = []
    by_excerpt: dict[str, EvalSample] = {}
    replica_metadata_by_excerpt: dict[str, list[dict[str, Any]]] = {}

    with dataset_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            raw_count += 1
            obj = json.loads(line)
            excerpt_id = str(obj.get("excerpt_id", index))
            replica_metadata_items = (
                replica_metadata_items_from_row(
                    obj,
                    index,
                    fallback_repetition=repetition,
                    expand_compact=True,
                )
                if include_replica_metadata
                else []
            )
            sample = EvalSample(
                excerpt_id=excerpt_id,
                sample_index=index,
                token_ids=obj["input_ids"],
                replica_metadata=tuple(replica_metadata_items),
            )
            if dedupe:
                by_excerpt.setdefault(excerpt_id, sample)
                replica_metadata_by_excerpt.setdefault(excerpt_id, []).extend(replica_metadata_items)
            else:
                samples.append(sample)

    if dedupe:
        samples = [
            EvalSample(
                excerpt_id=sample.excerpt_id,
                sample_index=sample.sample_index,
                token_ids=sample.token_ids,
                replica_metadata=tuple(replica_metadata_by_excerpt.get(sample.excerpt_id, ())),
            )
            for sample in by_excerpt.values()
        ]

    samples.sort(key=lambda sample: (sample.excerpt_id, sample.sample_index))
    return samples, raw_count, len(samples)


def select_samples(samples: list[EvalSample], max_excerpts: int | None, sample_seed: int) -> list[EvalSample]:
    if max_excerpts is None or max_excerpts <= 0 or max_excerpts >= len(samples):
        return samples

    rng = random.Random(sample_seed)
    selected = rng.sample(samples, max_excerpts)
    return sorted(selected, key=lambda sample: (sample.excerpt_id, sample.sample_index))


def stable_seed(sample_seed: int, *parts: object) -> int:
    payload = "::".join([str(sample_seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def fim_train_rng(split_seed: int, split_key: str) -> random.Random:
    digest = hashlib.sha256(f"{split_seed}:{split_key}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def strip_bos_eos_bounds(token_ids: list[int]) -> tuple[int, int]:
    start = 0
    end = len(token_ids)
    while start < end and token_ids[start] == BOS_TOKEN:
        start += 1
    while end > start and token_ids[end - 1] == EOS_TOKEN:
        end -= 1
    return start, end


def fim_train_split_bounds(
    token_ids: list[int],
    split_key: str,
    split_seed: int,
    content_length: int | None,
) -> tuple[int, int, int, int]:
    content_start, content_end = strip_bos_eos_bounds(token_ids)
    content_count = content_end - content_start
    if content_length is not None:
        content_count = min(content_count, max(content_length, 0))

    if content_count < 3:
        left_end = content_count // 3
        middle_end = left_end + (content_count // 3)
        return content_start, left_end, middle_end, content_count

    rng = fim_train_rng(split_seed, split_key)
    # Gutenberg FIM used swap-percent=100, but prepare_fim.py still consumes
    # the FIM coin before drawing split points.
    rng.random()
    left_end = rng.randint(1, content_count - 2)
    middle_end = rng.randint(left_end + 1, content_count - 1)
    return content_start, left_end, middle_end, content_count


def overlap_length(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def fim_train_segment_annotation(
    excerpt_id: str,
    token_ids: list[int],
    target_start_abs: int,
    target_length: int,
    split_seed: int,
    content_length: int | None,
    split_mode: str,
    replica_metadata: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    if split_mode == "fixed_by_excerpt":
        split_keys = [(excerpt_id, None)]
    elif split_mode == "replica_aware":
        if not replica_metadata:
            raise ValueError("replica_aware split annotations require replica metadata")
        split_keys = []
        for meta in replica_metadata:
            replica_index = meta.get("replica_index")
            if replica_index is None:
                raise ValueError(
                    f"replica_aware split annotations require replica_index for excerpt_id={excerpt_id}"
                )
            split_keys.append((f"{excerpt_id}:replica:{replica_index}", replica_index))
    else:
        raise ValueError(f"Unsupported FIM split mode: {split_mode}")

    per_split: list[dict[str, Any]] = []
    for split_key, replica_index in split_keys:
        content_start, left_end, middle_end, content_count = fim_train_split_bounds(
            token_ids,
            split_key=split_key,
            split_seed=split_seed,
            content_length=content_length,
        )
        target_start = target_start_abs - content_start
        target_end = target_start + target_length
        spans = {
            "left": (0, left_end),
            "middle": (left_end, middle_end),
            "right": (middle_end, content_count),
        }
        overlaps = {
            segment: overlap_length(target_start, target_end, segment_start, segment_end)
            for segment, (segment_start, segment_end) in spans.items()
        }
        covered_segments = [segment for segment, length in overlaps.items() if length > 0]
        middle_fraction = overlaps["middle"] / target_length if target_length else 0.0
        per_split.append(
            {
                "replica_index": replica_index,
                "content_start": content_start,
                "content_length": content_count,
                "left_end": left_end,
                "middle_start": left_end,
                "middle_end": middle_end,
                "target_start": target_start,
                "target_end": target_end,
                "target_segments": "+".join(covered_segments) if covered_segments else "outside_content",
                "left_overlap": overlaps["left"],
                "middle_overlap": overlaps["middle"],
                "right_overlap": overlaps["right"],
                "middle_fraction": middle_fraction,
                "within_middle": target_start >= left_end and target_end <= middle_end,
            }
        )

    first = per_split[0]
    middle_overlaps = [float(split["middle_overlap"]) for split in per_split]
    middle_fractions = [float(split["middle_fraction"]) for split in per_split]
    within_middle_count = sum(1 for split in per_split if split["within_middle"])
    num_splits = len(per_split)
    return {
        "train_fim_split_mode": split_mode,
        "train_fim_num_splits": num_splits,
        "train_fim_replica_indices_preview": [
            split["replica_index"] for split in per_split[:8] if split["replica_index"] is not None
        ],
        "train_fim_content_start": first["content_start"],
        "train_fim_content_length": first["content_length"],
        "train_fim_left_end": first["left_end"],
        "train_fim_middle_start": first["middle_start"],
        "train_fim_middle_end": first["middle_end"],
        "train_fim_target_start": first["target_start"],
        "train_fim_target_end": first["target_end"],
        "train_fim_target_segments": first["target_segments"],
        "train_fim_target_left_overlap": first["left_overlap"],
        "train_fim_target_middle_overlap": first["middle_overlap"],
        "train_fim_target_right_overlap": first["right_overlap"],
        "train_fim_target_middle_fraction": first["middle_fraction"],
        "train_fim_target_within_middle": first["within_middle"],
        "train_fim_target_within_middle_count": within_middle_count,
        "train_fim_target_within_middle_rate": within_middle_count / num_splits if num_splits else 0.0,
        "train_fim_target_ever_within_middle": within_middle_count > 0,
        "train_fim_target_middle_overlap_mean": sum(middle_overlaps) / num_splits if num_splits else 0.0,
        "train_fim_target_middle_overlap_max": max(middle_overlaps) if middle_overlaps else 0.0,
        "train_fim_target_middle_fraction_mean": sum(middle_fractions) / num_splits if num_splits else 0.0,
        "train_fim_target_middle_fraction_max": max(middle_fractions) if middle_fractions else 0.0,
    }


def candidate_target_starts(
    token_count: int,
    context_budget: int,
    prefix_length: int,
    middle_length: int,
    suffix_length: int,
    offset: int,
    window_stride: int,
    window_layout: str,
) -> list[int]:
    if window_layout == "matched_context":
        first_target_start = context_budget + offset
        last_target_start = token_count - middle_length - context_budget
    elif window_layout in {"cooper_nonoverlap", "cooper_sliding"}:
        first_target_start = prefix_length + offset
        last_target_start = token_count - middle_length - suffix_length
    else:
        raise ValueError(f"Unsupported window layout: {window_layout}")
    if first_target_start > last_target_start:
        return []
    return list(range(first_target_start, last_target_start + 1, window_stride))


def select_target_starts(
    starts: list[int],
    min_start_gap: int,
    max_windows_per_excerpt: int,
    window_selection: str,
    sample_seed: int,
    excerpt_id: str,
) -> list[int]:
    if min_start_gap <= 0:
        raise ValueError("min_start_gap must be positive")

    def overlaps_existing(start: int, selected: list[int]) -> bool:
        return any(abs(start - other) < min_start_gap for other in selected)

    def accept_in_order(candidates: list[int], limit: int | None = None) -> list[int]:
        selected: list[int] = []
        for start in candidates:
            if overlaps_existing(start, selected):
                continue
            selected.append(start)
            if limit is not None and len(selected) >= limit:
                break
        return sorted(selected)

    if max_windows_per_excerpt <= 0:
        return accept_in_order(starts)
    if window_selection == "first":
        return accept_in_order(starts, max_windows_per_excerpt)
    if window_selection == "uniform":
        non_overlapping_pool = accept_in_order(starts)
        if max_windows_per_excerpt >= len(non_overlapping_pool):
            return non_overlapping_pool
        rng = random.Random(stable_seed(sample_seed, excerpt_id, "windows"))
        return sorted(rng.sample(non_overlapping_pool, max_windows_per_excerpt))
    raise ValueError(f"Unsupported window selection: {window_selection}")


def build_segments_for_target(
    token_ids: list[int],
    target_start: int,
    prefix_length: int,
    middle_length: int,
    suffix_length: int,
) -> tuple[list[int], list[int], list[int]]:
    prefix_start = target_start - prefix_length
    middle_end = target_start + middle_length
    suffix_end = middle_end + suffix_length

    if prefix_start < 0 or suffix_end > len(token_ids):
        raise ValueError("Token sequence too short for requested target/lengths")

    prefix = token_ids[prefix_start:target_start]
    middle = token_ids[target_start:middle_end]
    suffix = token_ids[middle_end:suffix_end]
    return prefix, middle, suffix


def find_distractor_segments(
    samples: list[EvalSample],
    source_sample: EvalSample,
    target_start: int,
    prefix_length: int,
    middle_length: int,
    suffix_length: int,
    sample_seed: int,
) -> tuple[EvalSample, list[int], list[int], list[int]]:
    if len(samples) <= 1:
        raise ValueError("Context interventions require at least two selected excerpts")
    start_index = stable_seed(sample_seed, source_sample.excerpt_id, target_start, "distractor") % len(samples)
    for offset in range(len(samples)):
        candidate = samples[(start_index + offset) % len(samples)]
        if candidate.excerpt_id == source_sample.excerpt_id:
            continue
        try:
            prefix, middle, suffix = build_segments_for_target(
                candidate.token_ids,
                target_start=target_start,
                prefix_length=prefix_length,
                middle_length=middle_length,
                suffix_length=suffix_length,
            )
        except ValueError:
            continue
        return candidate, prefix, middle, suffix
    raise ValueError(
        f"Could not find a distractor excerpt for excerpt_id={source_sample.excerpt_id} "
        f"target_start={target_start}"
    )


def intervened_context_segments(
    args: argparse.Namespace,
    samples: list[EvalSample],
    sample: EvalSample,
    target_start: int,
    true_prefix: list[int],
    true_suffix: list[int],
) -> tuple[list[int], list[int], bool, bool, EvalSample | None]:
    intervention = getattr(args, "context_intervention", "full")
    replace_prefix = intervention in {"prefix_distractor", "both_distractor"} and args.prefix_length > 0
    replace_suffix = intervention in {"suffix_distractor", "both_distractor"} and args.suffix_length > 0
    if not replace_prefix and not replace_suffix:
        return true_prefix, true_suffix, False, False, None

    distractor, distractor_prefix, _distractor_middle, distractor_suffix = find_distractor_segments(
        samples=samples,
        source_sample=sample,
        target_start=target_start,
        prefix_length=args.prefix_length,
        middle_length=args.middle_length,
        suffix_length=args.suffix_length,
        sample_seed=args.sample_seed,
    )
    prefix = distractor_prefix if replace_prefix else true_prefix
    suffix = distractor_suffix if replace_suffix else true_suffix
    return prefix, suffix, replace_prefix, replace_suffix, distractor


def build_prompt(prefix: list[int], suffix: list[int], prompt_format: str) -> list[int]:
    if prompt_format == "ltr_prefix":
        return prefix
    if prompt_format == "fim_native":
        return [FIM_PREFIX] + prefix + [FIM_SUFFIX] + suffix + [FIM_MIDDLE]
    raise ValueError(f"Unsupported prompt format: {prompt_format}")


def build_windows_from_samples(
    args: argparse.Namespace,
    context_budget: int,
    raw_samples: list[EvalSample],
    raw_num_rows: int,
    rows_after_dedupe: int,
) -> WindowBuildResult:
    max_excerpts = args.max_excerpts if args.max_excerpts is not None else args.max_samples
    samples = select_samples(raw_samples, max_excerpts, args.sample_seed)
    if not samples:
        raise RuntimeError(f"No samples found in {Path(args.dataset)}")

    windows: list[EvalWindow] = []
    num_candidate_windows = 0
    num_selected_windows = 0
    num_excerpts_with_windows = 0
    num_short_excerpts = 0
    global_window_id = 0

    for sample in samples:
        candidate_starts = candidate_target_starts(
            token_count=len(sample.token_ids),
            context_budget=context_budget,
            prefix_length=args.prefix_length,
            middle_length=args.middle_length,
            suffix_length=args.suffix_length,
            offset=args.offset,
            window_stride=args.window_stride,
            window_layout=args.window_layout,
        )
        selected_starts = set(
            select_target_starts(
                candidate_starts,
                min_start_gap=args.prefix_length + args.middle_length + args.suffix_length,
                max_windows_per_excerpt=args.max_windows_per_excerpt,
                window_selection=args.window_selection,
                sample_seed=args.sample_seed,
                excerpt_id=sample.excerpt_id,
            )
        )
        num_candidate_windows += len(candidate_starts)
        num_selected_windows += len(selected_starts)

        if selected_starts:
            num_excerpts_with_windows += 1
        else:
            num_short_excerpts += 1

        for window_index, target_start in enumerate(candidate_starts):
            if target_start not in selected_starts:
                continue
            true_prefix, middle, true_suffix = build_segments_for_target(
                sample.token_ids,
                target_start=target_start,
                prefix_length=args.prefix_length,
                middle_length=args.middle_length,
                suffix_length=args.suffix_length,
            )
            prefix, suffix, prefix_is_distractor, suffix_is_distractor, distractor_sample = (
                intervened_context_segments(
                    args=args,
                    samples=samples,
                    sample=sample,
                    target_start=target_start,
                    true_prefix=true_prefix,
                    true_suffix=true_suffix,
                )
            )
            fim_annotation = (
                fim_train_segment_annotation(
                    excerpt_id=sample.excerpt_id,
                    token_ids=sample.token_ids,
                    target_start_abs=target_start,
                    target_length=len(middle),
                    split_seed=args.fim_train_split_seed,
                    content_length=args.fim_train_content_length,
                    split_mode=args.fim_split_mode,
                    replica_metadata=sample.replica_metadata,
                )
                if args.include_fim_annotations
                else {}
            )
            windows.append(
                EvalWindow(
                    global_window_id=global_window_id,
                    excerpt_id=sample.excerpt_id,
                    sample_index=sample.sample_index,
                    window_index=window_index,
                    target_start=target_start,
                    candidate_window_count=len(candidate_starts),
                    prefix=prefix,
                    middle=middle,
                    suffix=suffix,
                    true_prefix=true_prefix,
                    true_suffix=true_suffix,
                    context_intervention=getattr(args, "context_intervention", "full"),
                    prefix_is_distractor=prefix_is_distractor,
                    suffix_is_distractor=suffix_is_distractor,
                    distractor_excerpt_id=distractor_sample.excerpt_id if distractor_sample else None,
                    distractor_sample_index=distractor_sample.sample_index if distractor_sample else None,
                    fim_annotation=fim_annotation,
                )
            )
            global_window_id += 1

    return WindowBuildResult(
        windows=windows,
        raw_num_rows=raw_num_rows,
        num_rows_after_dedupe=rows_after_dedupe,
        num_excerpts=len(samples),
        num_excerpts_with_windows=num_excerpts_with_windows,
        num_short_excerpts=num_short_excerpts,
        num_candidate_windows=num_candidate_windows,
        num_selected_windows=num_selected_windows,
    )


def build_windows(args: argparse.Namespace, context_budget: int) -> WindowBuildResult:
    raw_samples, raw_num_rows, rows_after_dedupe = load_samples(
        Path(args.dataset),
        args.dedupe_excerpts,
        repetition=args.repetition,
        include_replica_metadata=args.include_fim_annotations,
    )
    return build_windows_from_samples(
        args=args,
        context_budget=context_budget,
        raw_samples=raw_samples,
        raw_num_rows=raw_num_rows,
        rows_after_dedupe=rows_after_dedupe,
    )


def batch_iter(items: list[EvalWindow], batch_size: int) -> Iterable[list[EvalWindow]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def safe_exp(value: float) -> float:
    if math.isfinite(value) and value < 50:
        return float(math.exp(value))
    return float("inf")


def safe_prob_from_log(value: float) -> float:
    if value == float("-inf"):
        return 0.0
    if not math.isfinite(value):
        return float("nan")
    if value < -745:
        return 0.0
    return float(math.exp(value))


def safe_log10_from_log(value: float) -> float:
    if value == float("-inf"):
        return float("-inf")
    if not math.isfinite(value):
        return float("nan")
    return value / math.log(10)


def target_shift_slice(prompt_length: int, target_length: int) -> slice:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if target_length < 0:
        raise ValueError("target_length must be non-negative")
    # In shifted LM scoring, logits[:, prompt_length - 1] predicts target token 0.
    start = prompt_length - 1
    return slice(start, start + target_length)


def empty_cooper_scores() -> dict[str, list[float]]:
    return {
        "cooper_log_p_z": [],
        "cooper_log10_p_z": [],
        "cooper_p_z": [],
        "cooper_mean_log_p_z": [],
        "cooper_mean_log10_p_z": [],
        "cooper_token_geomean_p_z": [],
        "cooper_extractable": [],
        "cooper_supported_token_rate": [],
        "cooper_all_tokens_in_topk": [],
    }


def target_position_tensors(
    prompt_lengths: list[int],
    target_lengths: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    starts = torch.tensor(
        [target_shift_slice(length, 0).start for length in prompt_lengths],
        dtype=torch.long,
        device=device,
    )
    lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
    max_target_length = int(lengths.max().item()) if lengths.numel() else 0
    offsets = torch.arange(max_target_length, dtype=torch.long, device=device).unsqueeze(0)
    valid = offsets < lengths.unsqueeze(1)
    positions = starts.unsqueeze(1) + offsets
    return positions, valid, lengths


def cooper_scores_from_target_logits(
    target_logits: torch.Tensor,
    target_labels: torch.Tensor,
    valid: torch.Tensor,
    target_lengths: torch.Tensor,
    top_k: int,
    temperature: float,
    threshold: float,
) -> dict[str, list[float]]:
    threshold_log = math.log(threshold)
    vocab_k = min(top_k, target_logits.shape[-1])
    batch_size = target_logits.shape[0]
    if batch_size == 0:
        return empty_cooper_scores()

    scaled_logits = target_logits / temperature
    top_values, top_indices = torch.topk(scaled_logits, k=vocab_k, dim=-1)
    matches = top_indices.eq(target_labels.unsqueeze(-1)) & valid.unsqueeze(-1)
    supported = matches.any(dim=-1)
    ranks = matches.to(torch.long).argmax(dim=-1)
    top_log_probs = torch.log_softmax(top_values, dim=-1)
    token_log_probs = top_log_probs.gather(-1, ranks.unsqueeze(-1)).squeeze(-1)
    token_log_probs = torch.where(supported, token_log_probs, torch.zeros_like(token_log_probs))

    supported_counts = supported.sum(dim=1)
    all_supported_tensor = supported_counts.eq(target_lengths)
    log_p_z_tensor = token_log_probs.sum(dim=1)
    neg_inf = torch.full_like(log_p_z_tensor, float("-inf"))
    log_p_z_tensor = torch.where(all_supported_tensor, log_p_z_tensor, neg_inf)
    safe_lengths = target_lengths.clamp_min(1).to(log_p_z_tensor.dtype)
    mean_log_p_z_tensor = log_p_z_tensor / safe_lengths
    supported_rate_tensor = supported_counts.to(log_p_z_tensor.dtype) / safe_lengths

    log_p_zs = [float(value) for value in log_p_z_tensor.detach().cpu().tolist()]
    mean_log_p_zs = [float(value) for value in mean_log_p_z_tensor.detach().cpu().tolist()]
    supported_rates = [float(value) for value in supported_rate_tensor.detach().cpu().tolist()]
    all_supported = [1.0 if bool(value) else 0.0 for value in all_supported_tensor.detach().cpu().tolist()]
    log10_p_zs = [safe_log10_from_log(value) for value in log_p_zs]
    p_zs = [safe_prob_from_log(value) for value in log_p_zs]
    mean_log10_p_zs = [safe_log10_from_log(value) for value in mean_log_p_zs]
    token_geomean_p_zs = [safe_prob_from_log(value) for value in mean_log_p_zs]
    extractable = [1.0 if value >= threshold_log else 0.0 for value in log_p_zs]

    return {
        "cooper_log_p_z": log_p_zs,
        "cooper_log10_p_z": log10_p_zs,
        "cooper_p_z": p_zs,
        "cooper_mean_log_p_z": mean_log_p_zs,
        "cooper_mean_log10_p_z": mean_log10_p_zs,
        "cooper_token_geomean_p_z": token_geomean_p_zs,
        "cooper_extractable": extractable,
        "cooper_supported_token_rate": supported_rates,
        "cooper_all_tokens_in_topk": all_supported,
    }


def topk_teacher_forced_target_scores(
    shift_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    prompt_lengths: list[int],
    target_lengths: list[int],
    top_k: int,
    temperature: float,
    threshold: float,
) -> dict[str, list[float]]:
    batch_size = shift_logits.shape[0]
    if batch_size == 0:
        return empty_cooper_scores()

    positions, valid, lengths = target_position_tensors(prompt_lengths, target_lengths, shift_logits.device)
    positions = positions.clamp(max=shift_logits.shape[1] - 1)
    rows = torch.arange(batch_size, dtype=torch.long, device=shift_logits.device).unsqueeze(1)
    target_logits = shift_logits[rows, positions, :]
    target_labels = shift_labels[rows, positions]
    return cooper_scores_from_target_logits(
        target_logits=target_logits,
        target_labels=target_labels,
        valid=valid,
        target_lengths=lengths,
        top_k=top_k,
        temperature=temperature,
        threshold=threshold,
    )


def decoder_backbone(model: AutoModelForCausalLM) -> torch.nn.Module | None:
    base_model_prefix = getattr(model, "base_model_prefix", None)
    if base_model_prefix:
        backbone = getattr(model, base_model_prefix, None)
        if backbone is not None and backbone is not model:
            return backbone
    backbone = getattr(model, "base_model", None)
    if backbone is not None and backbone is not model:
        return backbone
    return None


def target_only_shift_logits(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    target_positions: torch.Tensor,
) -> torch.Tensor | None:
    backbone = decoder_backbone(model)
    output_embeddings = model.get_output_embeddings()
    if backbone is None or output_embeddings is None:
        return None

    try:
        outputs = backbone(input_ids=input_ids, use_cache=False, return_dict=True)
    except TypeError:
        outputs = backbone(input_ids=input_ids, use_cache=False)
    hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    rows = torch.arange(input_ids.shape[0], dtype=torch.long, device=input_ids.device).unsqueeze(1)
    positions = target_positions.clamp(max=hidden_states.shape[1] - 1)
    target_hidden = hidden_states[rows, positions, :]
    target_logits = output_embeddings(target_hidden)
    final_logits_bias = getattr(model, "final_logits_bias", None)
    if final_logits_bias is not None:
        target_logits = target_logits + final_logits_bias.to(device=target_logits.device)
    return target_logits.float()


def teacher_forced_target_scores(
    model: AutoModelForCausalLM,
    windows: list[EvalWindow],
    prompt_format: str,
    device: torch.device,
    top_k: int,
    temperature: float,
    threshold: float,
    span_lengths: Iterable[int] | None = None,
) -> dict[str, list[float]]:
    prompts = [build_prompt(window.prefix, window.suffix, prompt_format) for window in windows]
    prompt_lengths = [len(prompt) for prompt in prompts]
    target_lengths = [len(window.middle) for window in windows]
    sequences = [prompt + window.middle for prompt, window in zip(prompts, windows)]
    sequence_lengths = {len(sequence) for sequence in sequences}
    if len(sequence_lengths) != 1:
        raise ValueError("Batched teacher-forced scoring requires equal sequence lengths")

    input_ids = torch.tensor(sequences, dtype=torch.long, device=device)
    positions, valid, lengths = target_position_tensors(prompt_lengths, target_lengths, input_ids.device)
    label_positions = (positions + 1).clamp(max=input_ids.shape[1] - 1)
    rows = torch.arange(input_ids.shape[0], dtype=torch.long, device=input_ids.device).unsqueeze(1)
    target_labels = input_ids[rows, label_positions]

    with torch.inference_mode():
        target_logits = target_only_shift_logits(model, input_ids, positions)
        if target_logits is None:
            logits = model(input_ids=input_ids, use_cache=False).logits
            shift_logits = logits[:, :-1, :].float()
            shift_labels = input_ids[:, 1:]
            positions = positions.clamp(max=shift_logits.shape[1] - 1)
            target_logits = shift_logits[rows, positions, :]
            target_labels = shift_labels[rows, positions]

    flat_nll = F.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        target_labels.reshape(-1),
        reduction="none",
    ).view(input_ids.shape[0], -1)
    per_token_nll = torch.where(valid, flat_nll, torch.zeros_like(flat_nll))
    nll_sums_tensor = per_token_nll.sum(dim=1)
    safe_lengths = lengths.clamp_min(1).to(nll_sums_tensor.dtype)
    nll_means_tensor = nll_sums_tensor / safe_lengths
    nll_means_tensor = torch.where(lengths.gt(0), nll_means_tensor, torch.full_like(nll_means_tensor, float("nan")))

    nll_sums = [float(value) for value in nll_sums_tensor.detach().cpu().tolist()]
    nll_means = [float(value) for value in nll_means_tensor.detach().cpu().tolist()]
    ppls = [safe_exp(value) for value in nll_means]
    scores = {
        "Ref_NLL": nll_means,
        "Ref_NLL_sum": nll_sums,
        "Ref_PPL": ppls,
    }
    full_cooper_scores = cooper_scores_from_target_logits(
        target_logits=target_logits,
        target_labels=target_labels,
        valid=valid,
        target_lengths=lengths,
        top_k=top_k,
        temperature=temperature,
        threshold=threshold,
    )
    scores.update(full_cooper_scores)
    min_target_length = int(lengths.min().item()) if lengths.numel() else 0
    for span_length in (tuple(int(length) for length in span_lengths) if span_lengths is not None else COOPER_PREFIX_TARGET_LENGTHS):
        if span_length > min_target_length:
            continue
        span_lengths = torch.full_like(lengths, span_length)
        span_scores = cooper_scores_from_target_logits(
            target_logits=target_logits[:, :span_length, :],
            target_labels=target_labels[:, :span_length],
            valid=valid[:, :span_length],
            target_lengths=span_lengths,
            top_k=top_k,
            temperature=temperature,
            threshold=threshold,
        )
        for key, values in span_scores.items():
            scores[f"{key}_first{span_length}"] = values
    return scores


def greedy_decode_with_scores(
    model: AutoModelForCausalLM,
    windows: list[EvalWindow],
    prompt_format: str,
    target_length: int,
    device: torch.device,
) -> tuple[list[list[int]], list[float], list[float], list[float]]:
    prompts = [build_prompt(window.prefix, window.suffix, prompt_format) for window in windows]
    prompt_lengths = {len(prompt) for prompt in prompts}
    if len(prompt_lengths) != 1:
        raise ValueError("Batched greedy decoding requires equal prompt lengths")

    input_ids = torch.tensor(prompts, dtype=torch.long, device=device)
    generated_steps: list[torch.Tensor] = []
    nll_steps: list[torch.Tensor] = []

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True)
        next_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        for step in range(target_length):
            log_probs = torch.log_softmax(next_logits.float(), dim=-1)
            next_token = torch.argmax(next_logits, dim=-1)
            token_nll = -log_probs.gather(1, next_token.unsqueeze(1)).squeeze(1)
            generated_steps.append(next_token)
            nll_steps.append(token_nll)

            if step + 1 < target_length:
                outputs = model(
                    input_ids=next_token.unsqueeze(1),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                next_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

    generated = torch.stack(generated_steps, dim=1)
    nll_tensor = torch.stack(nll_steps, dim=1)
    nll_sums_tensor = nll_tensor.sum(dim=1)
    nll_means_tensor = nll_sums_tensor / target_length

    generated_ids = generated.cpu().tolist()
    nll_sums = [float(value) for value in nll_sums_tensor.cpu().tolist()]
    nll_means = [float(value) for value in nll_means_tensor.cpu().tolist()]
    ppls = [safe_exp(value) for value in nll_means]
    return generated_ids, nll_means, nll_sums, ppls


def lcs_length(reference: list[int], prediction: list[int]) -> int:
    if not reference or not prediction:
        return 0

    prev = [0] * (len(prediction) + 1)
    curr = [0] * (len(prediction) + 1)
    for ref_token in reference:
        for j, pred_token in enumerate(prediction, start=1):
            if ref_token == pred_token:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (len(prediction) + 1)
    return prev[-1]


def rouge_l(reference: list[int], prediction: list[int]) -> float:
    if not reference or not prediction:
        return 0.0
    lcs = lcs_length(reference, prediction)
    precision = lcs / len(prediction) if prediction else 0.0
    recall = lcs / len(reference) if reference else 0.0
    if precision + recall == 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def type_token_ratio(tokens: list[int]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def metric_mean_std(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "std": float(np.std(arr)) if arr.size else float("nan"),
    }


def output_dir(args: argparse.Namespace) -> Path:
    results_root = Path(args.results_root) if args.results_root else default_results_root()
    if args.suite_name or args.arm_id:
        if not args.suite_name or not args.arm_id:
            raise ValueError("--suite-name and --arm-id must be provided together")
        return arm_root(results_root, args.suite_name, args.arm_id, args.repetition)
    return (
        results_root
        / args.study_name
        / args.prompt_format
        / args.model_label
        / f"rep_{args.repetition}"
    )


def output_stem(args: argparse.Namespace) -> str:
    if args.suite_name and args.arm_id:
        return "windows"
    window_limit = "all" if args.max_windows_per_excerpt == 0 else str(args.max_windows_per_excerpt)
    layout_prefix = "" if args.window_layout == "matched_context" else f"layout_{args.window_layout}_"
    stem = (
        f"{layout_prefix}"
        f"target_offset_{args.offset}_stride_{args.window_stride}_"
        f"windows_{window_limit}_prefix_{args.prefix_length}_"
        f"middle_{args.middle_length}_suffix_{args.suffix_length}"
    )
    intervention = getattr(args, "context_intervention", "full")
    if intervention != "full":
        stem = f"{stem}_intervention_{intervention}"
    return stem


def output_paths(args: argparse.Namespace, shard_rank: int | None = None) -> dict[str, Path]:
    out_dir = output_dir(args)
    stem = output_stem(args)
    if shard_rank is None:
        prefix = out_dir / stem
    else:
        prefix = out_dir / f"{stem}.shard_{shard_rank:03d}"
    return {
        "jsonl": Path(f"{prefix}.jsonl"),
        "summary": Path(f"{prefix}.summary.json"),
        "csv": Path(f"{prefix}.summary.csv"),
    }


def base_summary(
    args: argparse.Namespace,
    context_budget: int,
    build_result: WindowBuildResult,
    jsonl_path: Path,
    num_windows: int,
    accumulator: MetricAccumulator,
) -> dict[str, Any]:
    max_excerpts = args.max_excerpts if args.max_excerpts is not None else args.max_samples
    return {
        "dataset": str(Path(args.dataset)),
        "suite_name": args.suite_name,
        "arm_id": args.arm_id,
        "experiment": args.experiment,
        "model_path": args.model_path,
        "model_label": args.model_label,
        "prompt_format": args.prompt_format,
        "study_name": args.study_name,
        "repetition": args.repetition,
        "offset": args.offset,
        "context_budget": context_budget,
        "prefix_length": args.prefix_length,
        "middle_length": args.middle_length,
        "suffix_length": args.suffix_length,
        "window_stride": args.window_stride,
        "window_layout": args.window_layout,
        "max_windows_per_excerpt": args.max_windows_per_excerpt,
        "window_selection": args.window_selection,
        "sample_seed": args.sample_seed,
        "cooper_prefix_target_lengths": list(cooper_prefix_target_lengths(args)),
        "context_intervention": getattr(args, "context_intervention", "full"),
        "prob_extraction_top_k": args.prob_extraction_top_k,
        "prob_extraction_temperature": args.prob_extraction_temperature,
        "prob_extraction_threshold": args.prob_extraction_threshold,
        "fim_train_split_seed": args.fim_train_split_seed,
        "fim_train_content_length": args.fim_train_content_length,
        "fim_split_mode": args.fim_split_mode,
        "include_fim_annotations": args.include_fim_annotations,
        "dedupe_excerpts": args.dedupe_excerpts,
        "batch_size": args.batch_size,
        "generation_mode": getattr(args, "generation_mode", "greedy"),
        "shard_rank": args.shard_rank,
        "num_shards": args.num_shards,
        "raw_num_rows": build_result.raw_num_rows,
        "num_rows_after_dedupe": build_result.num_rows_after_dedupe,
        "max_excerpts": max_excerpts,
        "num_excerpts": build_result.num_excerpts,
        "num_excerpts_with_windows": build_result.num_excerpts_with_windows,
        "num_short_excerpts": build_result.num_short_excerpts,
        "num_candidate_windows": build_result.num_candidate_windows,
        "num_selected_windows": build_result.num_selected_windows,
        "num_windows": num_windows,
        "num_samples": num_windows,
        "metrics": accumulator.summary(),
        "match_rates": {
            "exact_match": float(np.mean(accumulator.values["exact_match"])) if num_windows else float("nan"),
            "greedy_exact_match": (
                float(np.mean(accumulator.values["greedy_exact_match"])) if num_windows else float("nan")
            ),
            "cooper_extractable": (
                float(np.mean(accumulator.values["cooper_extractable"])) if num_windows else float("nan")
            ),
            "match_ge_0_75": float(np.mean(accumulator.values["match_ge_0_75"])) if num_windows else float("nan"),
            "match_ge_0_50": float(np.mean(accumulator.values["match_ge_0_50"])) if num_windows else float("nan"),
            "match_ge_0_25": float(np.mean(accumulator.values["match_ge_0_25"])) if num_windows else float("nan"),
        },
        "per_sample_path": str(jsonl_path),
    }


def write_summary_csv(args: argparse.Namespace, summary: dict[str, Any], csv_path: Path) -> None:
    span_lengths = cooper_prefix_target_lengths(args)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "study_name",
                "prompt_format",
                "model_label",
                "repetition",
                "offset",
                "context_budget",
                "prefix_length",
                "middle_length",
                "suffix_length",
                "window_stride",
                "window_layout",
                "max_windows_per_excerpt",
                "window_selection",
                "sample_seed",
                "context_intervention",
                "prob_extraction_top_k",
                "prob_extraction_temperature",
                "prob_extraction_threshold",
                "fim_split_mode",
                "batch_size",
                "generation_mode",
                "num_shards",
                "num_excerpts",
                "num_candidate_windows",
                "num_samples",
                "memorization_score_mean",
                "token_accuracy_mean",
                "rouge_l_mean",
                "lcs_mean",
                "exact_match_rate",
                "ref_ppl_mean",
                "ppl_mean",
                "log_p_target_mean",
                "greedy_exact_match_rate",
                "cooper_extractable_rate",
                "cooper_p_z_mean",
                "cooper_token_geomean_p_z_mean",
                "cooper_mean_log10_p_z_mean",
                "cooper_log10_p_z_mean",
                "cooper_supported_token_rate_mean",
                *[
                    item
                    for span_length in span_lengths
                    for item in (
                        f"cooper_extractable_first{span_length}_rate",
                        f"cooper_p_z_first{span_length}_mean",
                    )
                ],
            ]
        )
        writer.writerow(
            [
                args.study_name,
                args.prompt_format,
                args.model_label,
                args.repetition,
                args.offset,
                summary["context_budget"],
                args.prefix_length,
                args.middle_length,
                args.suffix_length,
                args.window_stride,
                args.window_layout,
                args.max_windows_per_excerpt,
                args.window_selection,
                args.sample_seed,
                getattr(args, "context_intervention", "full"),
                args.prob_extraction_top_k,
                args.prob_extraction_temperature,
                args.prob_extraction_threshold,
                args.fim_split_mode,
                summary["batch_size"],
                summary.get("generation_mode", "greedy"),
                summary["num_shards"],
                summary["num_excerpts"],
                summary["num_candidate_windows"],
                summary["num_samples"],
                summary["metrics"]["memorization_score"]["mean"],
                summary["metrics"]["token_accuracy"]["mean"],
                summary["metrics"]["Rouge-L"]["mean"],
                summary["metrics"]["LCS"]["mean"],
                summary["match_rates"]["exact_match"],
                summary["metrics"]["Ref_PPL"]["mean"],
                summary["metrics"]["PPL"]["mean"],
                summary["metrics"]["log_p_target"]["mean"],
                summary["metrics"]["greedy_exact_match"]["mean"],
                summary["metrics"]["cooper_extractable"]["mean"],
                summary["metrics"]["cooper_p_z"]["mean"],
                summary["metrics"]["cooper_token_geomean_p_z"]["mean"],
                summary["metrics"]["cooper_mean_log10_p_z"]["mean"],
                summary["metrics"]["cooper_log10_p_z"]["mean"],
                summary["metrics"]["cooper_supported_token_rate"]["mean"],
                *[
                    value
                    for span_length in span_lengths
                    for value in (
                        summary["metrics"].get(f"cooper_extractable_first{span_length}", {}).get("mean", float("nan")),
                        summary["metrics"].get(f"cooper_p_z_first{span_length}", {}).get("mean", float("nan")),
                    )
                ],
            ]
        )


def build_row(
    args: argparse.Namespace,
    context_budget: int,
    window: EvalWindow,
    generated_middle: list[int] | None,
    ref_nll: float,
    ref_nll_sum: float,
    ref_ppl: float,
    cooper_log_p_z: float,
    cooper_log10_p_z: float,
    cooper_p_z: float,
    cooper_mean_log_p_z: float,
    cooper_mean_log10_p_z: float,
    cooper_token_geomean_p_z: float,
    cooper_extractable: float,
    cooper_supported_token_rate: float,
    cooper_all_tokens_in_topk: float,
    cooper_span_scores: dict[str, float],
    gen_nll: float,
    gen_nll_sum: float,
    gen_ppl: float,
    tokenizer: AutoTokenizer | None,
) -> dict[str, Any]:
    if generated_middle is None:
        lcs_raw = 0
        lcs_norm = float("nan")
        rouge = float("nan")
        positional_match_count = 0
        memorization_score = float("nan")
        exact_match = float("nan")
        generated_length = 0
        ttr_gen = float("nan")
        match_ge_0_75 = float("nan")
        match_ge_0_50 = float("nan")
        match_ge_0_25 = float("nan")
    else:
        if len(generated_middle) != len(window.middle):
            raise ValueError(
                "Rouge/greedy metrics require generated_middle to match the target middle length"
            )
        lcs_raw = lcs_length(window.middle, generated_middle)
        lcs_norm = lcs_raw / len(window.middle) if window.middle else 0.0
        precision = lcs_raw / len(generated_middle) if generated_middle else 0.0
        recall = lcs_raw / len(window.middle) if window.middle else 0.0
        rouge = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)
        positional_match_count = sum(
            1 for target_token, generated_token in zip(window.middle, generated_middle)
            if target_token == generated_token
        )
        memorization_score = positional_match_count / len(window.middle) if window.middle else 0.0
        exact_match = 1.0 if generated_middle == window.middle else 0.0
        generated_length = len(generated_middle)
        ttr_gen = type_token_ratio(generated_middle)
        match_ge_0_75 = 1.0 if lcs_norm >= 0.75 else 0.0
        match_ge_0_50 = 1.0 if lcs_norm >= 0.50 else 0.0
        match_ge_0_25 = 1.0 if lcs_norm >= 0.25 else 0.0

    row: dict[str, Any] = {
        "global_window_id": window.global_window_id,
        "suite_name": args.suite_name,
        "arm_id": args.arm_id,
        "experiment": args.experiment,
        "excerpt_id": window.excerpt_id,
        "sample_index": window.sample_index,
        "repetition": args.repetition,
        "offset": args.offset,
        "window_index": window.window_index,
        "target_start": window.target_start,
        "candidate_window_count": window.candidate_window_count,
        "context_budget": context_budget,
        "window_stride": args.window_stride,
        "window_layout": args.window_layout,
        "max_windows_per_excerpt": args.max_windows_per_excerpt,
        "window_selection": args.window_selection,
        "sample_seed": args.sample_seed,
        "context_intervention": window.context_intervention,
        "prefix_is_distractor": window.prefix_is_distractor,
        "suffix_is_distractor": window.suffix_is_distractor,
        "distractor_excerpt_id": window.distractor_excerpt_id,
        "distractor_sample_index": window.distractor_sample_index,
        "dedupe_excerpts": args.dedupe_excerpts,
        "prob_extraction_top_k": args.prob_extraction_top_k,
        "prob_extraction_temperature": args.prob_extraction_temperature,
        "prob_extraction_threshold": args.prob_extraction_threshold,
        "prefix_length": args.prefix_length,
        "middle_length": args.middle_length,
        "suffix_length": args.suffix_length,
        "prompt_format": args.prompt_format,
        "model_label": args.model_label,
        "true_length": len(window.middle),
        "generated_length": generated_length,
        "generation_mode": getattr(args, "generation_mode", "greedy"),
        "NLL": gen_nll,
        "NLL_sum": gen_nll_sum,
        "PPL": gen_ppl,
        "log_p_generated": -gen_nll_sum if math.isfinite(gen_nll_sum) else float("nan"),
        "Ref_NLL": ref_nll,
        "Ref_NLL_sum": ref_nll_sum,
        "Ref_PPL": ref_ppl,
        "log_p_target": -ref_nll_sum,
        "cooper_log_p_z": cooper_log_p_z,
        "cooper_log10_p_z": cooper_log10_p_z,
        "cooper_p_z": cooper_p_z,
        "cooper_mean_log_p_z": cooper_mean_log_p_z,
        "cooper_mean_log10_p_z": cooper_mean_log10_p_z,
        "cooper_token_geomean_p_z": cooper_token_geomean_p_z,
        "cooper_extractable": cooper_extractable,
        "cooper_supported_token_rate": cooper_supported_token_rate,
        "cooper_all_tokens_in_topk": cooper_all_tokens_in_topk,
        "memorization_score": memorization_score,
        "token_accuracy": memorization_score,
        "positional_match_count": positional_match_count,
        "Rouge-L": rouge,
        "LCS": lcs_norm,
        "LCS_length": lcs_raw,
        "TTR_ref": type_token_ratio(window.middle),
        "TTR_gen": ttr_gen,
        "exact_match": exact_match,
        "greedy_exact_match": exact_match,
        "match_ge_0_75": match_ge_0_75,
        "match_ge_0_50": match_ge_0_50,
        "match_ge_0_25": match_ge_0_25,
    }
    row.update(cooper_span_scores)
    row.update(window.fim_annotation)

    if args.include_token_ids:
        row.update(
            {
                "prefix_ids": window.prefix,
                "suffix_ids": window.suffix,
                "true_prefix_ids": window.true_prefix,
                "true_suffix_ids": window.true_suffix,
                "true_middle_ids": window.middle,
                "generated_middle_ids": generated_middle or [],
            }
        )
    if args.include_text:
        if tokenizer is None:
            raise RuntimeError("--include-text requires a tokenizer")
        row.update(
            {
                "prefix_text": tokenizer.decode(window.prefix, skip_special_tokens=False),
                "suffix_text": tokenizer.decode(window.suffix, skip_special_tokens=False),
                "true_prefix_text": tokenizer.decode(window.true_prefix, skip_special_tokens=False),
                "true_suffix_text": tokenizer.decode(window.true_suffix, skip_special_tokens=False),
                "true_middle_text": tokenizer.decode(window.middle, skip_special_tokens=False),
                "generated_middle_text": (
                    tokenizer.decode(generated_middle, skip_special_tokens=False)
                    if generated_middle is not None
                    else ""
                ),
            }
        )
    return row


def load_model(args: argparse.Namespace, device: torch.device) -> AutoModelForCausalLM:
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda":
        model_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.to(device)
    model.eval()
    return model


def evaluate_direct_task(
    args: argparse.Namespace,
    context_budget: int | None = None,
    model: AutoModelForCausalLM | None = None,
    device: torch.device | None = None,
    tokenizer: AutoTokenizer | None = None,
    raw_samples: list[EvalSample] | None = None,
    raw_num_rows: int | None = None,
    rows_after_dedupe: int | None = None,
    show_progress: bool = True,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    if context_budget is None:
        context_budget = validate_args(args)
    if raw_samples is None:
        build_result = build_windows(args, context_budget)
    else:
        if raw_num_rows is None or rows_after_dedupe is None:
            raise ValueError("raw_num_rows and rows_after_dedupe are required with raw_samples")
        build_result = build_windows_from_samples(
            args=args,
            context_budget=context_budget,
            raw_samples=raw_samples,
            raw_num_rows=raw_num_rows,
            rows_after_dedupe=rows_after_dedupe,
        )
    shard_windows = [
        window
        for window in build_result.windows
        if window.global_window_id % args.num_shards == args.shard_rank
    ]

    paths = output_paths(args, args.shard_rank if args.num_shards > 1 else None)
    paths["jsonl"].parent.mkdir(parents=True, exist_ok=True)
    accumulator = MetricAccumulator(metric_keys_for_spans(cooper_prefix_target_lengths(args)))

    if device is None:
        device = resolve_device(args.device)
    if model is None:
        model = load_model(args, device)
    if tokenizer is None and args.include_text:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(
        f"Evaluating shard {args.shard_rank}/{args.num_shards}: "
        f"{len(shard_windows)} of {len(build_result.windows)} selected windows on {device} "
        f"(generation_mode={getattr(args, 'generation_mode', 'greedy')})"
    )

    with paths["jsonl"].open("w", encoding="utf-8") as handle:
        num_batches = math.ceil(len(shard_windows) / args.batch_size) if shard_windows else 0
        batches = batch_iter(shard_windows, args.batch_size)
        span_lengths = cooper_prefix_target_lengths(args)
        progress = (
            tqdm(
                batches,
                total=num_batches,
                desc=f"{args.model_label} rep={args.repetition} shard={args.shard_rank}",
            )
            if show_progress
            else batches
        )
        for batch in progress:
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise DirectEvalTimeGuard("Stopped before writing a partial direct-eval summary")
            ref_scores = teacher_forced_target_scores(
                model=model,
                windows=batch,
                prompt_format=args.prompt_format,
                device=device,
                top_k=args.prob_extraction_top_k,
                temperature=args.prob_extraction_temperature,
                threshold=args.prob_extraction_threshold,
                span_lengths=span_lengths,
            )
            if getattr(args, "generation_mode", "greedy") == "greedy":
                generated_ids, gen_nlls, gen_nll_sums, gen_ppls = greedy_decode_with_scores(
                    model=model,
                    windows=batch,
                    prompt_format=args.prompt_format,
                    target_length=args.middle_length,
                    device=device,
                )
            else:
                generated_ids = [None] * len(batch)
                gen_nlls = [float("nan")] * len(batch)
                gen_nll_sums = [float("nan")] * len(batch)
                gen_ppls = [float("nan")] * len(batch)

            for index, window in enumerate(batch):
                cooper_span_scores = {
                    key: values[index]
                    for key, values in ref_scores.items()
                    if any(key.endswith(f"_first{span_length}") for span_length in span_lengths)
                }
                row = build_row(
                    args=args,
                    context_budget=context_budget,
                    window=window,
                    generated_middle=generated_ids[index],
                    ref_nll=ref_scores["Ref_NLL"][index],
                    ref_nll_sum=ref_scores["Ref_NLL_sum"][index],
                    ref_ppl=ref_scores["Ref_PPL"][index],
                    cooper_log_p_z=ref_scores["cooper_log_p_z"][index],
                    cooper_log10_p_z=ref_scores["cooper_log10_p_z"][index],
                    cooper_p_z=ref_scores["cooper_p_z"][index],
                    cooper_mean_log_p_z=ref_scores["cooper_mean_log_p_z"][index],
                    cooper_mean_log10_p_z=ref_scores["cooper_mean_log10_p_z"][index],
                    cooper_token_geomean_p_z=ref_scores["cooper_token_geomean_p_z"][index],
                    cooper_extractable=ref_scores["cooper_extractable"][index],
                    cooper_supported_token_rate=ref_scores["cooper_supported_token_rate"][index],
                    cooper_all_tokens_in_topk=ref_scores["cooper_all_tokens_in_topk"][index],
                    cooper_span_scores=cooper_span_scores,
                    gen_nll=gen_nlls[index],
                    gen_nll_sum=gen_nll_sums[index],
                    gen_ppl=gen_ppls[index],
                    tokenizer=tokenizer,
                )
                handle.write(json.dumps(row) + "\n")
                accumulator.update(row)

    summary = base_summary(
        args=args,
        context_budget=context_budget,
        build_result=build_result,
        jsonl_path=paths["jsonl"],
        num_windows=len(shard_windows),
        accumulator=accumulator,
    )
    with paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_summary_csv(args, summary, paths["csv"])

    print(f"Saved per-window rows to: {paths['jsonl']}")
    print(f"Saved summary JSON to: {paths['summary']}")
    print(f"Saved summary CSV to: {paths['csv']}")
    return {"paths": paths, "summary": summary, "num_windows": len(shard_windows)}


def run_eval(args: argparse.Namespace, context_budget: int) -> None:
    evaluate_direct_task(args=args, context_budget=context_budget)


def merge_shards(args: argparse.Namespace, context_budget: int) -> None:
    expected_shards = args.expected_shards or args.num_shards
    final_paths = output_paths(args, None)
    final_paths["jsonl"].parent.mkdir(parents=True, exist_ok=True)
    accumulator = MetricAccumulator(metric_keys_for_spans(cooper_prefix_target_lengths(args)))
    row_count = 0
    first_summary: dict[str, Any] | None = None

    with final_paths["jsonl"].open("w", encoding="utf-8") as out_handle:
        for shard_rank in range(expected_shards):
            shard_paths = output_paths(args, shard_rank)
            if not shard_paths["jsonl"].exists():
                raise FileNotFoundError(f"Missing shard JSONL: {shard_paths['jsonl']}")
            if not shard_paths["summary"].exists():
                raise FileNotFoundError(f"Missing shard summary: {shard_paths['summary']}")
            if first_summary is None:
                with shard_paths["summary"].open("r", encoding="utf-8") as handle:
                    first_summary = json.load(handle)

            with shard_paths["jsonl"].open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    out_handle.write(line)
                    row = json.loads(line)
                    accumulator.update(row)
                    row_count += 1

    if first_summary is None:
        raise RuntimeError("No shard summaries were found")

    summary = dict(first_summary)
    summary.update(
        {
            "context_budget": context_budget,
            "shard_rank": None,
            "num_shards": expected_shards,
            "num_windows": row_count,
            "num_samples": row_count,
            "metrics": accumulator.summary(),
            "match_rates": {
                "exact_match": float(np.mean(accumulator.values["exact_match"])) if row_count else float("nan"),
                "greedy_exact_match": (
                    float(np.mean(accumulator.values["greedy_exact_match"])) if row_count else float("nan")
                ),
                "cooper_extractable": (
                    float(np.mean(accumulator.values["cooper_extractable"])) if row_count else float("nan")
                ),
                "match_ge_0_75": float(np.mean(accumulator.values["match_ge_0_75"])) if row_count else float("nan"),
                "match_ge_0_50": float(np.mean(accumulator.values["match_ge_0_50"])) if row_count else float("nan"),
                "match_ge_0_25": float(np.mean(accumulator.values["match_ge_0_25"])) if row_count else float("nan"),
            },
            "per_sample_path": str(final_paths["jsonl"]),
            "merged_from_shards": expected_shards,
        }
    )

    with final_paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_summary_csv(args, summary, final_paths["csv"])

    print(f"Merged {row_count} rows into: {final_paths['jsonl']}")
    print(f"Saved merged summary JSON to: {final_paths['summary']}")
    print(f"Saved merged summary CSV to: {final_paths['csv']}")


def main() -> None:
    args = parse_args()
    context_budget = validate_args(args)
    if args.merge_shards:
        merge_shards(args, context_budget)
    else:
        run_eval(args, context_budget)


if __name__ == "__main__":
    main()
