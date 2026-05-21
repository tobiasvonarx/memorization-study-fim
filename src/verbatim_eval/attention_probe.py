#!/usr/bin/env python3
"""Attention-based memorization probes for matched Gutenberg windows.

This script reuses the same target-window construction as direct_overlap_eval,
but runs teacher-forced forwards with output attentions enabled. For each target
token, the relevant prediction-time query is the previous sequence position:
the last prompt token predicts target token 0, target token 0 predicts target
token 1, and so on. Metrics report how much attention those prediction queries
place on prefix, suffix, FIM sentinels, and already-seen target tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from direct_overlap_eval import (
    DirectEvalTimeGuard,
    FIM_MIDDLE,
    FIM_PREFIX,
    FIM_SUFFIX,
    EvalSample,
    EvalWindow,
    WindowBuildResult,
    batch_iter,
    build_prompt,
    build_windows,
    build_windows_from_samples,
    default_results_root,
    resolve_device,
    safe_exp,
    topk_teacher_forced_target_scores,
    validate_args,
)
from verbatim_suite import arm_root


ATTENTION_METRICS = [
    "attn_prefix_mass",
    "attn_suffix_mass",
    "attn_target_prev_mass",
    "attn_fim_marker_mass",
    "attn_prompt_mass",
    "attn_prefix_share_of_context",
    "attn_suffix_share_of_context",
    "attn_first_prefix_mass",
    "attn_first_suffix_mass",
    "attn_first_fim_marker_mass",
    "attn_later_prefix_mass",
    "attn_later_suffix_mass",
    "attn_later_target_prev_mass",
    "attn_later_fim_marker_mass",
    "attn_entropy_norm",
    "attn_max_weight",
    "attn_backward_distance",
    "attn_backward_distance_norm",
    "attn_early_prefix_mass",
    "attn_mid_prefix_mass",
    "attn_late_prefix_mass",
    "attn_early_suffix_mass",
    "attn_mid_suffix_mass",
    "attn_late_suffix_mass",
    "attn_early_target_prev_mass",
    "attn_mid_target_prev_mass",
    "attn_late_target_prev_mass",
    "attn_early_entropy_norm",
    "attn_mid_entropy_norm",
    "attn_late_entropy_norm",
    "Ref_NLL",
    "Ref_NLL_sum",
    "Ref_PPL",
    "cooper_log_p_z",
    "cooper_log10_p_z",
    "cooper_p_z",
    "cooper_token_geomean_p_z",
    "cooper_extractable",
    "cooper_supported_token_rate",
    "cooper_all_tokens_in_topk",
]


@dataclass(frozen=True)
class SequenceLayout:
    prompt_length: int
    target_length: int
    sequence_length: int
    prefix_mask: torch.Tensor
    suffix_mask: torch.Tensor
    target_mask: torch.Tensor
    fim_marker_mask: torch.Tensor
    query_positions: torch.Tensor


class MetricAccumulator:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {metric: [] for metric in ATTENTION_METRICS}

    def update(self, row: dict[str, Any]) -> None:
        for metric in ATTENTION_METRICS:
            if metric in row:
                self.values[metric].append(float(row[metric]))

    def summary(self) -> dict[str, dict[str, float]]:
        return {metric: finite_mean_std(values) for metric, values in self.values.items()}


def finite_mean_std(values: list[float]) -> dict[str, float]:
    arr = np.array([value for value in values if math.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attention analysis on verbatim memorization windows")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--repetition", required=True, type=int)
    parser.add_argument("--prompt-format", choices=["ltr_prefix", "fim_native"], required=True)
    parser.add_argument("--study-name", default="attention_probe")
    parser.add_argument("--suite-name", default=None, help="Suite name for unified suite output layout")
    parser.add_argument("--arm-id", default=None, help="Suite arm id for unified suite output layout")
    parser.add_argument("--experiment", default="attention", help="Logical experiment label stored in suite summaries")
    parser.add_argument("--prefix-length", type=int, default=80)
    parser.add_argument("--middle-length", type=int, default=20)
    parser.add_argument("--suffix-length", type=int, default=0)
    parser.add_argument("--context-budget", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--window-stride", type=int, default=100)
    parser.add_argument(
        "--window-layout",
        choices=["matched_context", "cooper_nonoverlap", "cooper_sliding"],
        default="cooper_nonoverlap",
    )
    parser.add_argument("--max-windows-per-excerpt", type=int, default=4)
    parser.add_argument("--window-selection", choices=["first", "uniform"], default="uniform")
    parser.add_argument("--max-excerpts", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--fim-train-split-seed", type=int, default=42)
    parser.add_argument("--fim-train-content-length", type=int, default=4096)
    parser.add_argument(
        "--fim-split-mode",
        choices=["fixed_by_excerpt", "replica_aware"],
        default="replica_aware",
    )
    parser.add_argument("--include-fim-annotations", action="store_true")
    dedupe_group = parser.add_mutually_exclusive_group()
    dedupe_group.add_argument("--dedupe-excerpts", dest="dedupe_excerpts", action="store_true", default=True)
    dedupe_group.add_argument("--no-dedupe-excerpts", dest="dedupe_excerpts", action="store_false")
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prob-extraction-top-k", type=int, default=40)
    parser.add_argument("--prob-extraction-temperature", type=float, default=1.0)
    parser.add_argument("--prob-extraction-threshold", type=float, default=0.001)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--expected-shards", type=int, default=None)
    parser.add_argument(
        "--include-layer-metrics",
        action="store_true",
        help="Store per-layer prefix/suffix/target/entropy arrays in each row.",
    )
    parser.add_argument(
        "--context-intervention",
        choices=["full", "suffix_distractor", "prefix_distractor", "both_distractor"],
        default="full",
        help="Use the same prompt-context intervention semantics as direct_overlap_eval.py.",
    )
    return parser.parse_args()


def output_dir(args: argparse.Namespace) -> Path:
    results_root = Path(args.results_root) if args.results_root else default_results_root()
    if args.suite_name and args.arm_id:
        return arm_root(results_root, args.suite_name, args.arm_id, args.repetition)
    return (
        results_root
        / "attention"
        / args.study_name
        / args.prompt_format
        / args.model_label
        / f"rep_{args.repetition}"
    )


def output_stem(args: argparse.Namespace) -> str:
    window_limit = "all" if args.max_windows_per_excerpt == 0 else str(args.max_windows_per_excerpt)
    layout_prefix = "" if args.window_layout == "matched_context" else f"layout_{args.window_layout}_"
    stem = (
        f"{layout_prefix}"
        f"target_offset_{args.offset}_stride_{args.window_stride}_"
        f"windows_{window_limit}_prefix_{args.prefix_length}_"
        f"middle_{args.middle_length}_suffix_{args.suffix_length}"
    )
    if args.context_intervention != "full":
        stem = f"{stem}_intervention_{args.context_intervention}"
    return stem


def output_paths(args: argparse.Namespace, shard_rank: int | None = None) -> dict[str, Path]:
    out_dir = output_dir(args)
    stem = "attention" if args.suite_name and args.arm_id else output_stem(args)
    suffix = "" if shard_rank is None else f".shard_{shard_rank:03d}"
    prefix = out_dir / f"{stem}{suffix}"
    if args.suite_name and args.arm_id:
        return {
            "jsonl": Path(f"{prefix}.jsonl"),
            "summary": Path(f"{prefix}.summary.json"),
            "csv": Path(f"{prefix}.summary.csv"),
        }
    return {
        "jsonl": Path(f"{prefix}.attention.jsonl"),
        "summary": Path(f"{prefix}.attention.summary.json"),
        "csv": Path(f"{prefix}.attention.summary.csv"),
    }


def build_sequence_layout(
    prompt_format: str,
    prefix_length: int,
    suffix_length: int,
    target_length: int,
    device: torch.device,
) -> SequenceLayout:
    if prompt_format == "ltr_prefix":
        prompt_length = prefix_length
        sequence_length = prompt_length + target_length
        prefix_positions = range(0, prefix_length)
        suffix_positions: range | list[int] = []
        marker_positions: list[int] = []
    elif prompt_format == "fim_native":
        prompt_length = prefix_length + suffix_length + 3
        sequence_length = prompt_length + target_length
        prefix_positions = range(1, 1 + prefix_length)
        fim_suffix_pos = 1 + prefix_length
        suffix_positions = range(fim_suffix_pos + 1, fim_suffix_pos + 1 + suffix_length)
        fim_middle_pos = prompt_length - 1
        marker_positions = [0, fim_suffix_pos, fim_middle_pos]
    else:
        raise ValueError(f"Unsupported prompt format: {prompt_format}")

    def mask_from_positions(positions: Iterable[int]) -> torch.Tensor:
        mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
        for position in positions:
            mask[position] = True
        return mask

    target_positions = range(prompt_length, sequence_length)
    query_positions = torch.arange(
        prompt_length - 1,
        prompt_length - 1 + target_length,
        dtype=torch.long,
        device=device,
    )
    return SequenceLayout(
        prompt_length=prompt_length,
        target_length=target_length,
        sequence_length=sequence_length,
        prefix_mask=mask_from_positions(prefix_positions),
        suffix_mask=mask_from_positions(suffix_positions),
        target_mask=mask_from_positions(target_positions),
        fim_marker_mask=mask_from_positions(marker_positions),
        query_positions=query_positions,
    )


def mean_segment_mass(selected: torch.Tensor, mask: torch.Tensor, query_slice: slice | None = None) -> torch.Tensor:
    if query_slice is not None:
        selected = selected[:, :, query_slice, :]
    if selected.shape[2] == 0:
        return torch.full((selected.shape[0],), float("nan"), device=selected.device)
    if not bool(mask.any().item()):
        return torch.zeros(selected.shape[0], device=selected.device)
    return selected[..., mask].sum(dim=-1).mean(dim=(1, 2))


def attention_entropy_norm(selected: torch.Tensor, query_positions: torch.Tensor) -> torch.Tensor:
    probs = selected.clamp_min(1e-30)
    entropy = -(probs * probs.log()).sum(dim=-1)
    normalizer = torch.log((query_positions + 1).float()).clamp_min(1.0)
    return (entropy / normalizer.view(1, 1, -1)).mean(dim=(1, 2))


def attention_backward_distance(selected: torch.Tensor, query_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(selected.shape[-1], device=selected.device).float()
    distances = (query_positions.float().unsqueeze(1) - positions.unsqueeze(0)).clamp_min(0.0)
    expected = (selected * distances.view(1, 1, *distances.shape)).sum(dim=-1)
    normalizer = (query_positions + 1).float().clamp_min(1.0)
    expected_norm = expected / normalizer.view(1, 1, -1)
    return expected.mean(dim=(1, 2)), expected_norm.mean(dim=(1, 2))


def stack_layer_metric(per_layer: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(per_layer, dim=1)


def band_means(values_by_layer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_layers = values_by_layer.shape[1]
    first = max(1, num_layers // 3)
    second = max(first + 1, (2 * num_layers) // 3)
    return (
        values_by_layer[:, :first].mean(dim=1),
        values_by_layer[:, first:second].mean(dim=1),
        values_by_layer[:, second:].mean(dim=1),
    )


def tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().tolist()]


def attention_metrics_for_batch(
    attentions: tuple[torch.Tensor, ...],
    layout: SequenceLayout,
    include_layer_metrics: bool,
) -> dict[str, list[Any]]:
    per_layer: dict[str, list[torch.Tensor]] = {
        "prefix": [],
        "suffix": [],
        "target_prev": [],
        "fim_marker": [],
        "prompt": [],
        "first_prefix": [],
        "first_suffix": [],
        "first_fim_marker": [],
        "later_prefix": [],
        "later_suffix": [],
        "later_target_prev": [],
        "later_fim_marker": [],
        "entropy_norm": [],
        "max_weight": [],
        "backward_distance": [],
        "backward_distance_norm": [],
    }

    prompt_mask = layout.prefix_mask | layout.suffix_mask | layout.fim_marker_mask
    later_slice = slice(1, None)

    for layer_attn in attentions:
        selected = layer_attn.float()[:, :, layout.query_positions, :]
        per_layer["prefix"].append(mean_segment_mass(selected, layout.prefix_mask))
        per_layer["suffix"].append(mean_segment_mass(selected, layout.suffix_mask))
        per_layer["target_prev"].append(mean_segment_mass(selected, layout.target_mask))
        per_layer["fim_marker"].append(mean_segment_mass(selected, layout.fim_marker_mask))
        per_layer["prompt"].append(mean_segment_mass(selected, prompt_mask))
        per_layer["first_prefix"].append(mean_segment_mass(selected, layout.prefix_mask, slice(0, 1)))
        per_layer["first_suffix"].append(mean_segment_mass(selected, layout.suffix_mask, slice(0, 1)))
        per_layer["first_fim_marker"].append(mean_segment_mass(selected, layout.fim_marker_mask, slice(0, 1)))
        per_layer["later_prefix"].append(mean_segment_mass(selected, layout.prefix_mask, later_slice))
        per_layer["later_suffix"].append(mean_segment_mass(selected, layout.suffix_mask, later_slice))
        per_layer["later_target_prev"].append(mean_segment_mass(selected, layout.target_mask, later_slice))
        per_layer["later_fim_marker"].append(mean_segment_mass(selected, layout.fim_marker_mask, later_slice))
        per_layer["entropy_norm"].append(attention_entropy_norm(selected, layout.query_positions))
        per_layer["max_weight"].append(selected.max(dim=-1).values.mean(dim=(1, 2)))
        distance, distance_norm = attention_backward_distance(selected, layout.query_positions)
        per_layer["backward_distance"].append(distance)
        per_layer["backward_distance_norm"].append(distance_norm)

    stacked = {key: stack_layer_metric(value) for key, value in per_layer.items()}
    result: dict[str, list[Any]] = {}
    mapping = {
        "prefix": "attn_prefix_mass",
        "suffix": "attn_suffix_mass",
        "target_prev": "attn_target_prev_mass",
        "fim_marker": "attn_fim_marker_mass",
        "prompt": "attn_prompt_mass",
        "first_prefix": "attn_first_prefix_mass",
        "first_suffix": "attn_first_suffix_mass",
        "first_fim_marker": "attn_first_fim_marker_mass",
        "later_prefix": "attn_later_prefix_mass",
        "later_suffix": "attn_later_suffix_mass",
        "later_target_prev": "attn_later_target_prev_mass",
        "later_fim_marker": "attn_later_fim_marker_mass",
        "entropy_norm": "attn_entropy_norm",
        "max_weight": "attn_max_weight",
        "backward_distance": "attn_backward_distance",
        "backward_distance_norm": "attn_backward_distance_norm",
    }
    for internal_key, output_key in mapping.items():
        result[output_key] = tensor_to_list(stacked[internal_key].mean(dim=1))

    context_mass = stacked["prefix"].mean(dim=1) + stacked["suffix"].mean(dim=1)
    safe_context = context_mass.clamp_min(1e-12)
    result["attn_prefix_share_of_context"] = tensor_to_list(stacked["prefix"].mean(dim=1) / safe_context)
    result["attn_suffix_share_of_context"] = tensor_to_list(stacked["suffix"].mean(dim=1) / safe_context)

    for internal_key, output_key in [
        ("prefix", "prefix_mass"),
        ("suffix", "suffix_mass"),
        ("target_prev", "target_prev_mass"),
        ("entropy_norm", "entropy_norm"),
    ]:
        early, mid, late = band_means(stacked[internal_key])
        result[f"attn_early_{output_key}"] = tensor_to_list(early)
        result[f"attn_mid_{output_key}"] = tensor_to_list(mid)
        result[f"attn_late_{output_key}"] = tensor_to_list(late)

    if include_layer_metrics:
        for internal_key in ["prefix", "suffix", "target_prev", "fim_marker", "entropy_norm"]:
            result[f"layer_{internal_key}"] = [
                tensor_to_list(stacked[internal_key][row_index, :])
                for row_index in range(stacked[internal_key].shape[0])
            ]
    return result


def load_attention_model(args: argparse.Namespace, device: torch.device) -> AutoModelForCausalLM:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda":
        kwargs["dtype"] = torch.bfloat16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            attn_implementation="eager",
            **kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    model.to(device)
    model.eval()
    model.config.output_attentions = True
    return model


def score_batch(
    model: AutoModelForCausalLM,
    windows: list[EvalWindow],
    args: argparse.Namespace,
    layout: SequenceLayout,
    device: torch.device,
) -> dict[str, list[Any]]:
    prompts = [build_prompt(window.prefix, window.suffix, args.prompt_format) for window in windows]
    sequences = [prompt + window.middle for prompt, window in zip(prompts, windows)]
    lengths = {len(sequence) for sequence in sequences}
    if lengths != {layout.sequence_length}:
        raise ValueError(f"Unexpected sequence lengths: {lengths}; expected {layout.sequence_length}")

    input_ids = torch.tensor(sequences, dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=False, output_attentions=True)
    if outputs.attentions is None:
        raise RuntimeError("Model did not return attentions; try a Transformers version with eager attention support.")

    logits = outputs.logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    per_token_nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(input_ids.shape[0], -1)

    ref_nlls: list[float] = []
    ref_nll_sums: list[float] = []
    ref_ppls: list[float] = []
    for row_index in range(input_ids.shape[0]):
        start = layout.prompt_length - 1
        end = start + layout.target_length
        nll_sum = float(per_token_nll[row_index, start:end].sum().item())
        nll_mean = nll_sum / layout.target_length
        ref_nll_sums.append(nll_sum)
        ref_nlls.append(nll_mean)
        ref_ppls.append(safe_exp(nll_mean))

    topk_scores = topk_teacher_forced_target_scores(
        shift_logits=shift_logits,
        shift_labels=shift_labels,
        prompt_lengths=[layout.prompt_length] * len(windows),
        target_lengths=[layout.target_length] * len(windows),
        top_k=args.prob_extraction_top_k,
        temperature=args.prob_extraction_temperature,
        threshold=args.prob_extraction_threshold,
    )
    attention_scores = attention_metrics_for_batch(
        attentions=outputs.attentions,
        layout=layout,
        include_layer_metrics=args.include_layer_metrics,
    )
    return {
        "Ref_NLL": ref_nlls,
        "Ref_NLL_sum": ref_nll_sums,
        "Ref_PPL": ref_ppls,
        **topk_scores,
        **attention_scores,
    }


def base_row(args: argparse.Namespace, context_budget: int, window: EvalWindow) -> dict[str, Any]:
    return {
        "global_window_id": window.global_window_id,
        "excerpt_id": window.excerpt_id,
        "sample_index": window.sample_index,
        "repetition": args.repetition,
        "target_start": window.target_start,
        "window_index": window.window_index,
        "candidate_window_count": window.candidate_window_count,
        "model_label": args.model_label,
        "prompt_format": args.prompt_format,
        "study_name": args.study_name,
        "suite_name": args.suite_name,
        "arm_id": args.arm_id,
        "experiment": args.experiment,
        "context_budget": context_budget,
        "prefix_length": args.prefix_length,
        "middle_length": args.middle_length,
        "suffix_length": args.suffix_length,
        "window_stride": args.window_stride,
        "window_layout": args.window_layout,
        "max_windows_per_excerpt": args.max_windows_per_excerpt,
        "window_selection": args.window_selection,
        "sample_seed": args.sample_seed,
        "context_intervention": args.context_intervention,
        "prefix_is_distractor": window.prefix_is_distractor,
        "suffix_is_distractor": window.suffix_is_distractor,
        "distractor_excerpt_id": window.distractor_excerpt_id,
        "distractor_sample_index": window.distractor_sample_index,
        "prob_extraction_top_k": args.prob_extraction_top_k,
        "prob_extraction_temperature": args.prob_extraction_temperature,
        "prob_extraction_threshold": args.prob_extraction_threshold,
    }


def write_summary_csv(args: argparse.Namespace, summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "study_name",
        "suite_name",
        "arm_id",
        "experiment",
        "prompt_format",
        "model_label",
        "repetition",
        "context_budget",
        "prefix_length",
        "middle_length",
        "suffix_length",
        "window_stride",
        "window_layout",
        "max_windows_per_excerpt",
        "context_intervention",
        "num_windows",
        "cooper_extractable_mean",
        "attn_prefix_mass_mean",
        "attn_suffix_mass_mean",
        "attn_target_prev_mass_mean",
        "attn_fim_marker_mass_mean",
        "attn_suffix_share_of_context_mean",
        "attn_entropy_norm_mean",
        "attn_max_weight_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerow(
            [
                args.study_name,
                args.suite_name,
                args.arm_id,
                args.experiment,
                args.prompt_format,
                args.model_label,
                args.repetition,
                summary["context_budget"],
                args.prefix_length,
                args.middle_length,
                args.suffix_length,
                args.window_stride,
                args.window_layout,
                args.max_windows_per_excerpt,
                args.context_intervention,
                summary["num_windows"],
                summary["metrics"]["cooper_extractable"]["mean"],
                summary["metrics"]["attn_prefix_mass"]["mean"],
                summary["metrics"]["attn_suffix_mass"]["mean"],
                summary["metrics"]["attn_target_prev_mass"]["mean"],
                summary["metrics"]["attn_fim_marker_mass"]["mean"],
                summary["metrics"]["attn_suffix_share_of_context"]["mean"],
                summary["metrics"]["attn_entropy_norm"]["mean"],
                summary["metrics"]["attn_max_weight"]["mean"],
            ]
        )


def make_summary(
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
        "model_path": args.model_path,
        "model_label": args.model_label,
        "prompt_format": args.prompt_format,
        "study_name": args.study_name,
        "suite_name": args.suite_name,
        "arm_id": args.arm_id,
        "experiment": args.experiment,
        "repetition": args.repetition,
        "context_budget": context_budget,
        "prefix_length": args.prefix_length,
        "middle_length": args.middle_length,
        "suffix_length": args.suffix_length,
        "window_stride": args.window_stride,
        "window_layout": args.window_layout,
        "max_windows_per_excerpt": args.max_windows_per_excerpt,
        "window_selection": args.window_selection,
        "sample_seed": args.sample_seed,
        "context_intervention": args.context_intervention,
        "prob_extraction_top_k": args.prob_extraction_top_k,
        "prob_extraction_temperature": args.prob_extraction_temperature,
        "prob_extraction_threshold": args.prob_extraction_threshold,
        "dedupe_excerpts": args.dedupe_excerpts,
        "include_layer_metrics": args.include_layer_metrics,
        "batch_size": args.batch_size,
        "shard_rank": args.shard_rank,
        "num_shards": args.num_shards,
        "raw_num_rows": build_result.raw_num_rows,
        "num_rows_after_dedupe": build_result.num_rows_after_dedupe,
        "max_excerpts": max_excerpts,
        "num_excerpts": build_result.num_excerpts,
        "num_candidate_windows": build_result.num_candidate_windows,
        "num_selected_windows": build_result.num_selected_windows,
        "num_windows": num_windows,
        "metrics": accumulator.summary(),
        "per_window_path": str(jsonl_path),
    }


def run_eval(args: argparse.Namespace, context_budget: int) -> None:
    evaluate_attention_task(args=args, context_budget=context_budget)


def evaluate_attention_task(
    args: argparse.Namespace,
    context_budget: int | None = None,
    model: AutoModelForCausalLM | None = None,
    device: torch.device | None = None,
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

    if device is None:
        device = resolve_device(args.device)
    if model is None:
        model = load_attention_model(args, device)
    layout = build_sequence_layout(
        prompt_format=args.prompt_format,
        prefix_length=args.prefix_length,
        suffix_length=args.suffix_length,
        target_length=args.middle_length,
        device=device,
    )
    accumulator = MetricAccumulator()

    print(
        f"Attention probe shard {args.shard_rank}/{args.num_shards}: "
        f"{len(shard_windows)} of {len(build_result.windows)} windows on {device}"
    )
    with paths["jsonl"].open("w", encoding="utf-8") as handle:
        total_batches = math.ceil(len(shard_windows) / args.batch_size) if shard_windows else 0
        batches = batch_iter(shard_windows, args.batch_size)
        progress = (
            tqdm(
                batches,
                total=total_batches,
                desc=f"attention {args.model_label} rep={args.repetition} shard={args.shard_rank}",
            )
            if show_progress
            else batches
        )
        for batch in progress:
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise DirectEvalTimeGuard("Stopped before writing a partial attention summary")
            scores = score_batch(model, batch, args, layout, device)
            for row_index, window in enumerate(batch):
                row = base_row(args, context_budget, window)
                for key, values in scores.items():
                    row[key] = values[row_index]
                row.update(window.fim_annotation)
                handle.write(json.dumps(row) + "\n")
                accumulator.update(row)

    summary = make_summary(args, context_budget, build_result, paths["jsonl"], len(shard_windows), accumulator)
    with paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_summary_csv(args, summary, paths["csv"])
    print(f"Saved attention rows to: {paths['jsonl']}")
    print(f"Saved attention summary JSON to: {paths['summary']}")
    print(f"Saved attention summary CSV to: {paths['csv']}")
    return {"paths": paths, "summary": summary, "num_windows": len(shard_windows)}


def merge_shards(args: argparse.Namespace, context_budget: int) -> None:
    expected_shards = args.expected_shards or args.num_shards
    final_paths = output_paths(args, None)
    final_paths["jsonl"].parent.mkdir(parents=True, exist_ok=True)
    accumulator = MetricAccumulator()
    row_count = 0
    first_summary: dict[str, Any] | None = None

    with final_paths["jsonl"].open("w", encoding="utf-8") as out_handle:
        for shard_rank in range(expected_shards):
            shard_paths = output_paths(args, shard_rank)
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
        raise RuntimeError("No shard summaries found")
    summary = dict(first_summary)
    summary.update(
        {
            "context_budget": context_budget,
            "shard_rank": None,
            "num_shards": expected_shards,
            "num_windows": row_count,
            "metrics": accumulator.summary(),
            "per_window_path": str(final_paths["jsonl"]),
            "merged_from_shards": expected_shards,
        }
    )
    with final_paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_summary_csv(args, summary, final_paths["csv"])
    print(f"Merged {row_count} rows into: {final_paths['jsonl']}")
    print(f"Saved merged attention summary JSON to: {final_paths['summary']}")


def main() -> None:
    args = parse_args()
    context_budget = validate_args(args)
    if args.merge_shards:
        merge_shards(args, context_budget)
    else:
        run_eval(args, context_budget)


if __name__ == "__main__":
    main()
