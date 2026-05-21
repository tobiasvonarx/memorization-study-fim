#!/usr/bin/env python3
"""Create appendix-ready audit figures for Cooper LTR memorization windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM


MODEL_ORDER = ["no_fim", "fim_v2", "fineweb_only"]
PAIR_MODEL_ORDER = ["no_fim", "fim_v2"]
COOPER_EXTRACTABILITY_THRESHOLD = 0.001
DEFAULT_UNIQUE_TOKEN_FILE = (
    "/capstor/store/cscs/swissai/infra01/users/tvonarx/datasets/gutenberg/"
    "4096_tokens_filtered/token.jsonl"
)


@dataclass(frozen=True)
class Candidate:
    row: dict[str, Any]
    summary: dict[str, Any]
    jsonl_path: Path


@dataclass(frozen=True)
class TokenAudit:
    token_index: int
    token_id: int
    token_text: str
    raw_logit: float
    full_logprob: float
    top40_logprob: float
    top40_prob: float
    top40_rank: int | None
    in_top40: bool
    top1_token_id: int
    top1_token_text: str
    top1_top40_prob: float


@dataclass(frozen=True)
class WordAudit:
    word_index: int
    text: str
    token_start: int
    token_end: int
    mean_raw_logit: float
    mean_full_logprob: float
    mean_top40_logprob: float
    sum_top40_logprob: float
    min_top40_rank: int | None
    all_tokens_in_top40: bool


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_results_root() -> Path:
    return repo_root() / "results"


def parse_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in value.replace(",", " ").split() if item]


def parse_repetitions(value: str | None) -> set[int] | None:
    tokens = parse_csv_arg(value)
    return {int(token) for token in tokens} if tokens else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="core_ltr")
    parser.add_argument("--results-root", type=Path, default=default_results_root())
    parser.add_argument("--unique-token-file", type=Path, default=Path(DEFAULT_UNIQUE_TOKEN_FILE))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-labels", default="fim_v2,no_fim,fineweb_only")
    parser.add_argument("--repetitions", default="128", help="Comma/space-separated repetition buckets; empty means all.")
    parser.add_argument("--examples-per-model", type=int, default=2)
    parser.add_argument(
        "--paired-cooper-scenarios",
        action="store_true",
        help=(
            "Select paired no-FIM/FIM-v2 windows covering no-FIM-only, FIM-only, "
            "and shared Cooper extractability scenarios."
        ),
    )
    parser.add_argument(
        "--examples-per-scenario",
        type=int,
        default=1,
        help="Number of paired same-window audits to select for each Cooper scenario.",
    )
    parser.add_argument("--top-n", type=int, default=0, help="If positive, select this many examples globally.")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0", help="Use cuda:N, cpu, or auto.")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--require-extractable", action="store_true", default=True)
    parser.add_argument("--allow-non-extractable", dest="require_extractable", action="store_false")
    parser.add_argument("--min-cooper-p-z", type=float, default=0.0)
    parser.add_argument("--prefix-display-chars", type=int, default=760)
    parser.add_argument("--target-color-vmin", type=float, default=0.0)
    parser.add_argument("--target-color-vmax", type=float, default=1.0)
    parser.add_argument("--max-token-label-chars", type=int, default=18)
    return parser.parse_args()


def suite_root(results_root: Path, suite: str) -> Path:
    return Path(results_root) / "verbatim_eval" / "suites" / suite


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_candidates(args: argparse.Namespace) -> list[Candidate]:
    root = suite_root(args.results_root, args.suite)
    repetitions = parse_repetitions(args.repetitions)
    model_labels = set(parse_csv_arg(args.model_labels))
    candidates: list[Candidate] = []

    for summary_path in sorted(root.glob("arms/*/rep_*/windows.summary.json")):
        summary = read_json(summary_path)
        if summary.get("experiment") != "ltr" or summary.get("prompt_format") != "ltr_prefix":
            continue
        if model_labels and str(summary.get("model_label")) not in model_labels:
            continue
        if repetitions is not None and int(summary["repetition"]) not in repetitions:
            continue
        jsonl_path = summary_path.with_name("windows.jsonl")
        if not jsonl_path.exists():
            continue
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not args.paired_cooper_scenarios:
                    if args.require_extractable and finite_float(row.get("cooper_extractable"), 0.0) < 0.5:
                        continue
                    if finite_float(row.get("cooper_p_z"), 0.0) < args.min_cooper_p_z:
                        continue
                candidates.append(Candidate(row=row, summary=summary, jsonl_path=jsonl_path))
    if not candidates:
        raise RuntimeError("No matching LTR audit candidates found")
    return candidates


def candidate_sort_key(candidate: Candidate) -> tuple[float, float, float]:
    row = candidate.row
    return (
        finite_float(row.get("cooper_p_z"), float("-inf")),
        finite_float(row.get("cooper_log_p_z"), float("-inf")),
        -finite_float(row.get("Ref_NLL"), float("inf")),
    )


def select_examples(candidates: list[Candidate], args: argparse.Namespace) -> list[Candidate]:
    if args.paired_cooper_scenarios:
        return select_paired_cooper_examples(candidates, args)

    ordered = sorted(candidates, key=candidate_sort_key, reverse=True)
    if args.top_n > 0:
        selected: list[Candidate] = []
        seen: set[tuple[str, str, int]] = set()
        for candidate in ordered:
            key = (
                str(candidate.row["model_label"]),
                str(candidate.row["excerpt_id"]),
                int(candidate.row["target_start"]),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(candidate)
            if len(selected) >= args.top_n:
                break
        return selected

    by_model = {label: [] for label in MODEL_ORDER}
    for candidate in ordered:
        by_model.setdefault(str(candidate.row["model_label"]), []).append(candidate)

    selected = []
    for model_label in MODEL_ORDER:
        seen_windows: set[tuple[str, int]] = set()
        for candidate in by_model.get(model_label, []):
            key = (str(candidate.row["excerpt_id"]), int(candidate.row["target_start"]))
            if key in seen_windows:
                continue
            seen_windows.add(key)
            selected.append(candidate)
            if len(seen_windows) >= args.examples_per_model:
                break
    return selected


def same_window_key(row: dict[str, Any]) -> tuple[int, str, int, int, int]:
    return (
        int(row["repetition"]),
        str(row["excerpt_id"]),
        int(row["sample_index"]),
        int(row["window_index"]),
        int(row["target_start"]),
    )


def scenario_label(no_fim_row: dict[str, Any], fim_row: dict[str, Any]) -> str | None:
    no_fim_extractable = finite_float(no_fim_row.get("cooper_extractable"), 0.0) >= 0.5
    fim_extractable = finite_float(fim_row.get("cooper_extractable"), 0.0) >= 0.5
    if no_fim_extractable and not fim_extractable:
        return "ltr_only"
    if fim_extractable and not no_fim_extractable:
        return "fim_only"
    if no_fim_extractable and fim_extractable:
        return "both"
    return None


def scenario_display_name(scenario: str) -> str:
    return {
        "ltr_only": "memorized by no-FIM only",
        "fim_only": "memorized by FIM-v2 only",
        "both": "memorized by both models",
    }.get(scenario, scenario.replace("_", " "))


def scenario_sort_score(scenario: str, no_fim: Candidate, fim: Candidate) -> tuple[float, float, float]:
    no_p = finite_float(no_fim.row.get("cooper_p_z"), 0.0)
    fim_p = finite_float(fim.row.get("cooper_p_z"), 0.0)
    if scenario == "ltr_only":
        return (no_p, no_p / max(fim_p, 1e-300), -finite_float(no_fim.row.get("Ref_NLL"), float("inf")))
    if scenario == "fim_only":
        return (fim_p, fim_p / max(no_p, 1e-300), -finite_float(fim.row.get("Ref_NLL"), float("inf")))
    return (min(no_p, fim_p), no_p + fim_p, -max(finite_float(no_fim.row.get("Ref_NLL"), 0.0), finite_float(fim.row.get("Ref_NLL"), 0.0)))


def pair_id_for(scenario: str, row: dict[str, Any]) -> str:
    return safe_filename(
        f"{scenario}_rep{row['repetition']}_{row['excerpt_id']}_window_{int(row['window_index']):04d}_t{row['target_start']}"
    )


def annotate_pair(scenario: str, no_fim: Candidate, fim: Candidate) -> list[Candidate]:
    pair_id = pair_id_for(scenario, no_fim.row)
    no_p = finite_float(no_fim.row.get("cooper_p_z"), 0.0)
    fim_p = finite_float(fim.row.get("cooper_p_z"), 0.0)
    for candidate, counterpart_label, counterpart_p in [
        (no_fim, "fim_v2", fim_p),
        (fim, "no_fim", no_p),
    ]:
        candidate.row["audit_scenario"] = scenario
        candidate.row["audit_scenario_display"] = scenario_display_name(scenario)
        candidate.row["audit_pair_id"] = pair_id
        candidate.row["audit_no_fim_cooper_p_z"] = no_p
        candidate.row["audit_fim_v2_cooper_p_z"] = fim_p
        candidate.row["audit_counterpart_model_label"] = counterpart_label
        candidate.row["audit_counterpart_cooper_p_z"] = counterpart_p
    return [no_fim, fim]


def select_paired_cooper_examples(candidates: list[Candidate], args: argparse.Namespace) -> list[Candidate]:
    by_window: dict[tuple[int, str, int, int, int], dict[str, Candidate]] = {}
    for candidate in candidates:
        model_label = str(candidate.row.get("model_label"))
        if model_label not in PAIR_MODEL_ORDER:
            continue
        by_window.setdefault(same_window_key(candidate.row), {})[model_label] = candidate

    by_scenario: dict[str, list[tuple[tuple[float, float, float], Candidate, Candidate]]] = {
        "ltr_only": [],
        "fim_only": [],
        "both": [],
    }
    for models in by_window.values():
        no_fim = models.get("no_fim")
        fim = models.get("fim_v2")
        if no_fim is None or fim is None:
            continue
        scenario = scenario_label(no_fim.row, fim.row)
        if scenario is None:
            continue
        by_scenario[scenario].append((scenario_sort_score(scenario, no_fim, fim), no_fim, fim))

    selected: list[Candidate] = []
    missing = [name for name, rows in by_scenario.items() if not rows]
    if missing:
        raise RuntimeError(f"No paired Cooper audit candidates found for scenarios: {', '.join(missing)}")

    for scenario in ["ltr_only", "fim_only", "both"]:
        ranked = sorted(by_scenario[scenario], key=lambda item: item[0], reverse=True)
        for _score, no_fim, fim in ranked[: args.examples_per_scenario]:
            selected.extend(annotate_pair(scenario, no_fim, fim))
    return selected


def load_unique_rows(unique_token_file: Path, excerpt_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with unique_token_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            excerpt_id = str(obj["excerpt_id"])
            if excerpt_id in excerpt_ids:
                rows[excerpt_id] = obj
                if len(rows) == len(excerpt_ids):
                    break
    missing = sorted(excerpt_ids - set(rows))
    if missing:
        raise RuntimeError(f"Unique token file is missing {len(missing)} selected excerpts: {missing[:5]}")
    return rows


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {value}, but CUDA is not available")
    return torch.device(value)


def load_model(model_path: str, device: torch.device, dtype_name: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype(dtype_name),
        trust_remote_code=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model


def tokenizer_for_model(model_path: str) -> Tokenizer:
    tokenizer_path = Path(model_path) / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer.json in {model_path}")
    return Tokenizer.from_file(str(tokenizer_path))


def decode(tokenizer: Tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def token_piece(tokenizer: Tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    return text if text else f"<{token_id}>"


def normalize_display_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def display_piece(text: str, max_chars: int) -> str:
    shown = re.sub(r"\s+", " ", text).strip() or "space"
    if len(shown) > max_chars:
        shown = shown[: max_chars - 1] + "..."
    return shown


def window_tokens(candidate: Candidate, unique_row: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    row = candidate.row
    token_ids = unique_row["input_ids"]
    target_start = int(row["target_start"])
    prefix_length = int(row["prefix_length"])
    middle_length = int(row["middle_length"])
    suffix_length = int(row["suffix_length"])
    prefix_start = target_start - prefix_length
    middle_end = target_start + middle_length
    suffix_end = middle_end + suffix_length
    if prefix_start < 0 or suffix_end > len(token_ids):
        raise ValueError(f"Window extends outside token sequence for {row['excerpt_id']}")
    return (
        token_ids[prefix_start:target_start],
        token_ids[target_start:middle_end],
        token_ids[middle_end:suffix_end],
    )


def audit_token_logits(
    model: AutoModelForCausalLM,
    tokenizer: Tokenizer,
    device: torch.device,
    prefix: list[int],
    target: list[int],
    top_k: int,
    temperature: float,
) -> list[TokenAudit]:
    input_ids = torch.tensor([prefix + target], dtype=torch.long, device=device)
    prompt_length = len(prefix)
    target_length = len(target)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0]
        target_logits = logits[prompt_length - 1 : prompt_length - 1 + target_length, :]
        scaled_logits = target_logits / temperature
        vocab_k = min(top_k, scaled_logits.shape[-1])
        top_values, top_indices = torch.topk(scaled_logits, k=vocab_k, dim=-1)
        top_log_probs = torch.log_softmax(top_values, dim=-1)
        full_log_probs = torch.log_softmax(scaled_logits, dim=-1)

    audits: list[TokenAudit] = []
    for index, token_id in enumerate(target):
        indices = top_indices[index]
        matches = (indices == int(token_id)).nonzero(as_tuple=False)
        if matches.numel():
            rank_index = int(matches[0].item())
            top40_logprob = float(top_log_probs[index, rank_index].item())
            top40_rank = rank_index + 1
            in_top40 = True
        else:
            top40_logprob = float("-inf")
            top40_rank = None
            in_top40 = False
        top1_id = int(indices[0].item())
        audits.append(
            TokenAudit(
                token_index=index,
                token_id=int(token_id),
                token_text=token_piece(tokenizer, int(token_id)),
                raw_logit=float(scaled_logits[index, int(token_id)].item()),
                full_logprob=float(full_log_probs[index, int(token_id)].item()),
                top40_logprob=top40_logprob,
                top40_prob=math.exp(top40_logprob) if math.isfinite(top40_logprob) else 0.0,
                top40_rank=top40_rank,
                in_top40=in_top40,
                top1_token_id=top1_id,
                top1_token_text=token_piece(tokenizer, top1_id),
                top1_top40_prob=float(torch.exp(top_log_probs[index, 0]).item()),
            )
        )
    return audits


def group_word_audits(token_audits: list[TokenAudit]) -> list[WordAudit]:
    groups: list[list[TokenAudit]] = []
    current: list[TokenAudit] = []
    for audit in token_audits:
        starts_new = bool(current) and bool(audit.token_text) and audit.token_text[0].isspace()
        if starts_new:
            groups.append(current)
            current = []
        current.append(audit)
    if current:
        groups.append(current)

    words: list[WordAudit] = []
    for index, group in enumerate(groups):
        text = "".join(item.token_text for item in group).strip() or "".join(item.token_text for item in group)
        top40_values = [item.top40_logprob for item in group]
        finite_top40 = [value for value in top40_values if math.isfinite(value)]
        all_in_top40 = len(finite_top40) == len(top40_values)
        mean_top40 = sum(finite_top40) / len(top40_values) if all_in_top40 else float("-inf")
        ranks = [item.top40_rank for item in group if item.top40_rank is not None]
        words.append(
            WordAudit(
                word_index=index,
                text=text,
                token_start=group[0].token_index,
                token_end=group[-1].token_index + 1,
                mean_raw_logit=sum(item.raw_logit for item in group) / len(group),
                mean_full_logprob=sum(item.full_logprob for item in group) / len(group),
                mean_top40_logprob=mean_top40,
                sum_top40_logprob=sum(finite_top40) if all_in_top40 else float("-inf"),
                min_top40_rank=min(ranks) if ranks else None,
                all_tokens_in_top40=all_in_top40,
            )
        )
    return words


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")


def luminance(rgba: tuple[float, float, float, float]) -> float:
    red, green, blue, _alpha = rgba
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def draw_token_boxes(
    ax: mpl.axes.Axes,
    token_audits: list[TokenAudit],
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    max_token_label_chars: int,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x = 0.015
    y = 0.92
    row_height = 0.31
    gap = 0.006
    for audit in token_audits:
        label = display_piece(audit.token_text, max_token_label_chars)
        token_p_z = audit.top40_prob
        score_label = f"{token_p_z:.2f}"
        width = min(0.34, max(0.075, 0.028 + 0.0118 * max(len(label), len(score_label))))
        if x + width > 0.985:
            x = 0.015
            y -= row_height
        color = cmap(norm(token_p_z))
        text_color = "black" if luminance(color) > 0.55 else "white"
        rect = mpl.patches.Rectangle(
            (x, y - 0.225),
            width,
            0.195,
            linewidth=0.55,
            edgecolor="#111111",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(
            x + width / 2,
            y - 0.095,
            label,
            ha="center",
            va="center",
            color=text_color,
            fontsize=5.7,
            family="DejaVu Sans",
        )
        ax.text(
            x + width / 2,
            y - 0.175,
            score_label,
            ha="center",
            va="center",
            color=text_color,
            fontsize=5.6,
            family="DejaVu Sans",
        )
        x += width + gap


def model_display_label(model_label: str) -> str:
    return {
        "no_fim": "no-FIM",
        "fim_v2": "FIM-v2",
        "fineweb_only": "FineWeb-only",
    }.get(model_label, model_label)


def figure_caption_text(
    candidate: Candidate,
    unique_row: dict[str, Any],
    target_length: int,
) -> str:
    row = candidate.row
    scenario = str(row.get("audit_scenario_display", "Cooper-extractable audit window"))
    model_label = str(row["model_label"])
    no_fim_p = finite_float(row.get("audit_no_fim_cooper_p_z"))
    fim_p = finite_float(row.get("audit_fim_v2_cooper_p_z"))
    return (
        f"{model_display_label(model_label)} audit for a window {scenario} under the "
        "Cooper exact-target criterion. "
        f"Repetition {row['repetition']}; source book {unique_row.get('source_book_id')}; "
        f"excerpt {row['excerpt_id']}; target start {row['target_start']}; "
        f"prefix length {row['prefix_length']} tokens; target length {target_length} tokens. "
        f"Cooper p_z values: no-FIM={no_fim_p:.6g}, FIM-v2={fim_p:.6g}; "
        f"extractability threshold t={COOPER_EXTRACTABILITY_THRESHOLD:g}. "
        "Each target-token box reports that model's top-40-renormalized ground-truth "
        "token probability; the full-window Cooper p_z is the product of these token probabilities."
    )


def make_figure(
    output_base: Path,
    candidate: Candidate,
    unique_row: dict[str, Any],
    tokenizer: Tokenizer,
    prefix: list[int],
    target: list[int],
    token_audits: list[TokenAudit],
    args: argparse.Namespace,
) -> None:
    row = candidate.row
    prefix_text = normalize_display_text(decode(tokenizer, prefix))
    if len(prefix_text) > args.prefix_display_chars:
        prefix_text = "..." + prefix_text[-args.prefix_display_chars :]

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.Normalize(vmin=args.target_color_vmin, vmax=args.target_color_vmax)

    fig = plt.figure(figsize=(3.55, 4.42), dpi=300)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1.18, 1.82],
        hspace=0.02,
    )
    ax_prefix = fig.add_subplot(grid[0])
    ax_target = fig.add_subplot(grid[1])
    ax_prefix.axis("off")

    ax_prefix.text(0.0, 1.0, "Prefix context", fontsize=7.2, color="#333333", va="top")
    wrapped_prefix = textwrap.fill(prefix_text.replace("\n", " / "), width=62)
    ax_prefix.text(
        0.0,
        0.90,
        wrapped_prefix,
        fontsize=5.45,
        family="DejaVu Sans Mono",
        color="#1f1f1f",
        va="top",
        bbox={
            "boxstyle": "square,pad=0.32",
            "facecolor": "#f7f8fa",
            "edgecolor": "#d7dbe2",
            "linewidth": 0.8,
        },
    )

    ax_target.text(
        0.015,
        0.995,
        "Target tokens",
        transform=ax_target.transAxes,
        fontsize=7.2,
        color="#333333",
        va="top",
    )
    draw_token_boxes(ax_target, token_audits, cmap, norm, args.max_token_label_chars)

    output_base.with_suffix(".caption.txt").write_text(
        figure_caption_text(candidate, unique_row, len(target)) + "\n",
        encoding="utf-8",
    )

    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def clean_csv_text(text: str) -> str:
    return normalize_display_text(text).replace("\n", "\\n")


def write_outputs(
    selected: list[Candidate],
    unique_rows: dict[str, dict[str, Any]],
    audits: dict[str, tuple[list[int], list[int], list[TokenAudit], list[WordAudit], Tokenizer]],
    output_dir: Path,
) -> None:
    example_csv = output_dir / "audit_examples.csv"
    token_csv = output_dir / "audit_token_logits.csv"
    word_csv = output_dir / "audit_word_logits.csv"
    markdown = output_dir / "audit_examples.md"

    with example_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "example_id",
                "figure_pdf",
                "figure_png",
                "caption_file",
                "audit_scenario",
                "audit_pair_id",
                "suite_name",
                "arm_id",
                "model_label",
                "repetition",
                "source_book_id",
                "excerpt_id",
                "target_start",
                "prefix_length",
                "middle_length",
                "cooper_extractable",
                "cooper_p_z",
                "no_fim_cooper_p_z",
                "fim_v2_cooper_p_z",
                "cooper_log_p_z",
                "recomputed_cooper_log_p_z",
                "cooper_log_p_z_delta",
                "Ref_NLL",
                "Ref_PPL",
                "prefix_text",
                "target_text",
            ],
        )
        writer.writeheader()
        for candidate in selected:
            row = candidate.row
            example_id = example_id_for(candidate)
            prefix, target, token_audits, _word_audits, tokenizer = audits[example_id]
            recomputed = sum(item.top40_logprob for item in token_audits)
            stored = finite_float(row.get("cooper_log_p_z"))
            unique_row = unique_rows[str(row["excerpt_id"])]
            writer.writerow(
                {
                    "example_id": example_id,
                    "figure_pdf": f"figures/{example_id}.pdf",
                    "figure_png": f"figures/{example_id}.png",
                    "caption_file": f"figures/{example_id}.caption.txt",
                    "audit_scenario": row.get("audit_scenario"),
                    "audit_pair_id": row.get("audit_pair_id"),
                    "suite_name": row.get("suite_name"),
                    "arm_id": row.get("arm_id"),
                    "model_label": row.get("model_label"),
                    "repetition": row.get("repetition"),
                    "source_book_id": unique_row.get("source_book_id"),
                    "excerpt_id": row.get("excerpt_id"),
                    "target_start": row.get("target_start"),
                    "prefix_length": row.get("prefix_length"),
                    "middle_length": row.get("middle_length"),
                    "cooper_extractable": row.get("cooper_extractable"),
                    "cooper_p_z": row.get("cooper_p_z"),
                    "no_fim_cooper_p_z": row.get("audit_no_fim_cooper_p_z"),
                    "fim_v2_cooper_p_z": row.get("audit_fim_v2_cooper_p_z"),
                    "cooper_log_p_z": row.get("cooper_log_p_z"),
                    "recomputed_cooper_log_p_z": recomputed,
                    "cooper_log_p_z_delta": recomputed - stored,
                    "Ref_NLL": row.get("Ref_NLL"),
                    "Ref_PPL": row.get("Ref_PPL"),
                    "prefix_text": clean_csv_text(decode(tokenizer, prefix)),
                    "target_text": clean_csv_text(decode(tokenizer, target)),
                }
            )

    with token_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "example_id",
                "token_index",
                "token_id",
                "token_text",
                "raw_logit",
                "full_logprob",
                "top40_logprob",
                "top40_prob",
                "top40_rank",
                "in_top40",
                "top1_token_id",
                "top1_token_text",
                "top1_top40_prob",
            ],
        )
        writer.writeheader()
        for example_id, (_prefix, _target, token_audits, _word_audits, _tokenizer) in audits.items():
            for item in token_audits:
                writer.writerow(
                    {
                        "example_id": example_id,
                        "token_index": item.token_index,
                        "token_id": item.token_id,
                        "token_text": item.token_text.replace("\n", "\\n"),
                        "raw_logit": item.raw_logit,
                        "full_logprob": item.full_logprob,
                        "top40_logprob": item.top40_logprob,
                        "top40_prob": item.top40_prob,
                        "top40_rank": item.top40_rank,
                        "in_top40": item.in_top40,
                        "top1_token_id": item.top1_token_id,
                        "top1_token_text": item.top1_token_text.replace("\n", "\\n"),
                        "top1_top40_prob": item.top1_top40_prob,
                    }
                )

    with word_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "example_id",
                "word_index",
                "word_text",
                "token_start",
                "token_end",
                "mean_raw_logit",
                "mean_full_logprob",
                "mean_top40_logprob",
                "sum_top40_logprob",
                "min_top40_rank",
                "all_tokens_in_top40",
            ],
        )
        writer.writeheader()
        for example_id, (_prefix, _target, _token_audits, word_audits, _tokenizer) in audits.items():
            for item in word_audits:
                writer.writerow(
                    {
                        "example_id": example_id,
                        "word_index": item.word_index,
                        "word_text": item.text.replace("\n", "\\n"),
                        "token_start": item.token_start,
                        "token_end": item.token_end,
                        "mean_raw_logit": item.mean_raw_logit,
                        "mean_full_logprob": item.mean_full_logprob,
                        "mean_top40_logprob": item.mean_top40_logprob,
                        "sum_top40_logprob": item.sum_top40_logprob,
                        "min_top40_rank": item.min_top40_rank,
                        "all_tokens_in_top40": item.all_tokens_in_top40,
                    }
                )

    with markdown.open("w", encoding="utf-8") as handle:
        handle.write("# LTR Cooper Audit Examples\n\n")
        handle.write(
            "Each audited window is shown twice: once under the no-FIM checkpoint and "
            "once under the FIM-v2 checkpoint. The figures are portrait-oriented for "
            "horizontal LaTeX subfigures. Figure metadata and full Cooper p_z values are "
            "stored in the per-figure `.caption.txt` files; token-level logits are in "
            "`audit_token_logits.csv`.\n\n"
        )
        previous_pair = None
        for candidate in selected:
            row = candidate.row
            example_id = example_id_for(candidate)
            unique_row = unique_rows[str(row["excerpt_id"])]
            _prefix, target, _token_audits, _word_audits, tokenizer = audits[example_id]
            pair_id = row.get("audit_pair_id", example_id)
            if pair_id != previous_pair:
                handle.write(f"## {pair_id}\n\n")
                if row.get("audit_scenario_display"):
                    handle.write(f"Scenario: {row['audit_scenario_display']}.\n\n")
                previous_pair = pair_id
            handle.write(f"### {model_display_label(str(row['model_label']))}\n\n")
            handle.write(f"![{example_id}](figures/{example_id}.png)\n\n")
            handle.write(
                f"- model: `{row['model_label']}`; repetition: `{row['repetition']}`; "
                f"book: `{unique_row.get('source_book_id')}`; excerpt: `{row['excerpt_id']}`; "
                f"target_start: `{row['target_start']}`\n"
            )
            handle.write(
                f"- Cooper p_z: `{finite_float(row.get('cooper_p_z')):.6g}`; "
                f"no-FIM p_z: `{finite_float(row.get('audit_no_fim_cooper_p_z')):.6g}`; "
                f"FIM-v2 p_z: `{finite_float(row.get('audit_fim_v2_cooper_p_z')):.6g}`; "
                f"Ref_NLL: `{finite_float(row.get('Ref_NLL')):.4f}`\n"
            )
            handle.write(f"- caption: `figures/{example_id}.caption.txt`\n")
            handle.write(f"- target: {decode(tokenizer, target).replace(chr(10), ' / ')}\n\n")


def example_id_for(candidate: Candidate) -> str:
    row = candidate.row
    if row.get("audit_pair_id"):
        return safe_filename(f"{row['audit_pair_id']}_{row['model_label']}")
    return safe_filename(
        f"{row['model_label']}_rep{row['repetition']}_{row['excerpt_id']}_t{row['target_start']}"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or suite_root(args.results_root, args.suite) / "appendix_audits"
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(args)
    selected = select_examples(candidates, args)
    if not selected:
        raise RuntimeError("No examples selected after de-duplication")

    excerpt_ids = {str(candidate.row["excerpt_id"]) for candidate in selected}
    unique_rows = load_unique_rows(args.unique_token_file, excerpt_ids)
    device = resolve_device(args.device)
    audits: dict[str, tuple[list[int], list[int], list[TokenAudit], list[WordAudit], Tokenizer]] = {}

    by_model: dict[str, list[Candidate]] = {}
    for candidate in selected:
        by_model.setdefault(str(candidate.row["model_label"]), []).append(candidate)

    for model_label in MODEL_ORDER:
        model_candidates = by_model.get(model_label, [])
        if not model_candidates:
            continue
        model_path = str(model_candidates[0].summary["model_path"])
        print(f"Loading model {model_label}: {model_path}", flush=True)
        tokenizer = tokenizer_for_model(model_path)
        model = load_model(model_path, device, args.dtype)
        for candidate in model_candidates:
            example_id = example_id_for(candidate)
            unique_row = unique_rows[str(candidate.row["excerpt_id"])]
            prefix, target, _suffix = window_tokens(candidate, unique_row)
            token_audits = audit_token_logits(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prefix=prefix,
                target=target,
                top_k=args.top_k,
                temperature=args.temperature,
            )
            word_audits = group_word_audits(token_audits)
            audits[example_id] = (prefix, target, token_audits, word_audits, tokenizer)
            output_base = figure_dir / example_id
            make_figure(
                output_base=output_base,
                candidate=candidate,
                unique_row=unique_row,
                tokenizer=tokenizer,
                prefix=prefix,
                target=target,
                token_audits=token_audits,
                args=args,
            )
            print(f"Wrote {output_base.with_suffix('.pdf')}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_outputs(selected, unique_rows, audits, output_dir)
    print(f"Wrote metadata: {output_dir / 'audit_examples.csv'}")
    print(f"Wrote token logits: {output_dir / 'audit_token_logits.csv'}")
    print(f"Wrote word logits: {output_dir / 'audit_word_logits.csv'}")
    print(f"Wrote index: {output_dir / 'audit_examples.md'}")


if __name__ == "__main__":
    main()
