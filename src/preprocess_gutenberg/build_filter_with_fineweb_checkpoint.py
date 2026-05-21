#!/usr/bin/env python3
"""Build and filter Gutenberg excerpts with a FineWeb-trained LLaMA checkpoint.

This stage combines sliding-window excerpt generation and checkpoint scoring in
one run:
1. shard source books across GPU ranks
2. generate 4096-token candidate windows on CPU worker threads
3. score each window on the local GPU in batches
4. keep the highest-PPL window per source book
5. apply global threshold filtering and write the thresholded candidate pool

The output directory contains:
- scores.jsonl: every scored sliding-window candidate with metadata and PPL
- thresholded_token.jsonl/text.jsonl: thresholded one-window-per-book excerpts
- build_filter_summary.json: run metadata and counts
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
MEGATRON_DIR = os.environ.get("MEGATRON_DIR", str(REPO_ROOT.parent / "megatron-lm"))
sys.path.insert(0, MEGATRON_DIR)

from gpt_builders import gpt_builder  # noqa: E402
from model_provider import model_provider  # noqa: E402
from megatron.training import get_args, get_model, get_tokenizer, print_rank_0  # noqa: E402
from megatron.training.checkpointing import load_checkpoint  # noqa: E402
from megatron.training.initialize import initialize_megatron  # noqa: E402
from megatron.training.utils import get_ltor_masks_and_position_ids  # noqa: E402


TEXT_COLUMN = "text"
BOOK_ID_COLUMN = "id"
EXCERPT_ID_COLUMN = "excerpt_id"
INPUT_IDS_COLUMN = "input_ids"
SOURCE_BOOK_ID_COLUMN = "source_book_id"
TEXT_LENGTH_COLUMN = "text_length"
TOKEN_LENGTH_COLUMN = "token_length"
WINDOW_INDEX_COLUMN = "window_index"
WINDOW_START_TOKEN_COLUMN = "window_start_token"

GUTENBERG_START_PATTERNS = [
    re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL),
    re.compile(r"START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE),
]

GUTENBERG_END_PATTERNS = [
    re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL),
    re.compile(r"END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE),
    re.compile(r"End of the Project Gutenberg EBook", re.IGNORECASE),
    re.compile(r"Project Gutenberg(?:'s)?\s+(?:License|Literary Archive Foundation)", re.IGNORECASE),
]


@dataclass
class BookTask:
    book_id: str
    raw_text: str


@dataclass
class PreparedBook:
    book_id: str
    token_ids: List[int]


def add_extra_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("Gutenberg build + filter")
    group.add_argument("--output-dir", type=str, required=True, help="Directory for score/filter outputs")
    group.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer path or HF id. Defaults to $LLAMA_TOKENIZER_PATH or $TOKENIZER_ROOT/llama3_2_3B_tokenizer.",
    )
    group.add_argument("--num-books", type=int, default=34000, help="Maximum number of source books to process.")
    group.add_argument("--num-tokens", type=int, default=4096, help="Exact token length for each sliding window.")
    group.add_argument("--char-pos-start", type=int, default=10000, help="Start character offset for the source slice.")
    group.add_argument("--char-pos-end", type=int, default=80000, help="End character offset for the source slice.")
    group.add_argument(
        "--min-char-length",
        type=int,
        default=80000,
        help="Minimum raw text length required before processing a source book.",
    )
    group.add_argument(
        "--window-stride-tokens",
        type=int,
        default=4096,
        help="Stride, in tokens, for the sliding window over each source book.",
    )
    group.add_argument(
        "--max-windows-per-book",
        type=int,
        default=0,
        help="Optional cap on candidate windows per book. Use 0 for no cap.",
    )
    group.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HF datasets cache dir. Defaults under $DATASET_ROOT/gutenberg/cache.",
    )
    group.add_argument(
        "--preprocess-workers",
        type=int,
        default=8,
        help="CPU worker threads per GPU rank for tokenization and window extraction.",
    )
    group.add_argument(
        "--score-batch-size",
        type=int,
        default=16,
        help="Number of sliding windows to score per GPU forward pass.",
    )
    group.add_argument(
        "--log-every-books",
        type=int,
        default=25,
        help="Emit a local progress log after every N processed books.",
    )
    group.add_argument(
        "--memorized-max-ppl",
        type=float,
        default=None,
        help="Drop excerpts whose perplexity is at or below this threshold.",
    )
    group.add_argument(
        "--max-allowed-ppl",
        type=float,
        default=None,
        help="Drop excerpts whose perplexity is above this threshold.",
    )
    group.add_argument(
        "--drop-low-ppl-fraction",
        type=float,
        default=None,
        help="Drop this lowest-PPL fraction after scoring the best excerpt from each book.",
    )
    return parser


def rank_prefix() -> str:
    if not dist.is_initialized():
        return "rank0"
    return "rank{}".format(dist.get_rank())


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print("[{}][{}] {}".format(timestamp, rank_prefix(), message), flush=True)


def ensure_checkpoint_is_read_only(load_path: str, output_dir: Path) -> Path:
    checkpoint_dir = Path(load_path).resolve()
    output_dir = output_dir.resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError("Checkpoint load path does not exist: {}".format(checkpoint_dir))
    if not checkpoint_dir.is_dir():
        raise ValueError("Checkpoint load path must be a directory: {}".format(checkpoint_dir))
    if output_dir == checkpoint_dir or output_dir.is_relative_to(checkpoint_dir):
        raise ValueError(
            "Refusing to write outputs inside the checkpoint tree. "
            "output_dir={} checkpoint_dir={}".format(output_dir, checkpoint_dir)
        )
    log("Using checkpoint in read-only mode from {}".format(checkpoint_dir))
    return checkpoint_dir


def resolve_tokenizer_path(cli_value: Optional[str]) -> str:
    if cli_value:
        return cli_value
    user = os.environ["USER"]
    tokenizer_root = os.getenv("TOKENIZER_ROOT", "/iopsstor/scratch/cscs/{}/tokenizer".format(user))
    return os.getenv("LLAMA_TOKENIZER_PATH", "{}/llama3_2_3B_tokenizer".format(tokenizer_root))


def resolve_cache_dir(cli_value: Optional[str]) -> str:
    if cli_value:
        return cli_value
    user = os.environ["USER"]
    persistent_root = os.getenv(
        "PERSISTENT_ROOT",
        "/capstor/store/cscs/swissai/infra01/users/{}".format(user),
    )
    dataset_root = Path(os.getenv("DATASET_ROOT", "{}/datasets".format(persistent_root)))
    return str(dataset_root / "gutenberg" / "cache")


def load_cached_gutenberg(cache_dir: str) -> Dataset:
    base = Path(cache_dir) / "manu___project_gutenberg" / "default" / "0.0.0"
    if not base.exists():
        raise FileNotFoundError("No cached Gutenberg dataset root found under {}".format(base))

    dataset_roots = sorted(path for path in base.iterdir() if path.is_dir())
    if not dataset_roots:
        raise FileNotFoundError("No cached Gutenberg dataset root found under {}".format(base))

    arrow_files = sorted(dataset_roots[-1].glob("project_gutenberg-en-*.arrow"))
    if not arrow_files:
        raise FileNotFoundError("No Gutenberg arrow shards found under {}".format(dataset_roots[-1]))

    shards = [Dataset.from_file(str(path)) for path in arrow_files]
    return shards[0] if len(shards) == 1 else concatenate_datasets(shards)


def load_gutenberg_dataset(cache_dir: str) -> Dataset:
    try:
        dataset = load_cached_gutenberg(cache_dir)
        print_rank_0("Loaded cached Gutenberg Arrow shards directly from {}".format(cache_dir))
        return dataset
    except FileNotFoundError:
        print_rank_0("Cached Gutenberg Arrow shards not found under {}; falling back to load_dataset".format(cache_dir))

    return load_dataset(
        "manu/project_gutenberg",
        split="en",
        cache_dir=cache_dir,
        download_mode="reuse_cache_if_exists",
    )


def select_book_indices(dataset: Dataset, num_books: int, min_char_length: int, char_pos_end: int) -> List[int]:
    max_required_chars = max(min_char_length, char_pos_end)
    selected: List[int] = []
    seen = set()

    for index, example in enumerate(dataset):
        book_id = str(example[BOOK_ID_COLUMN])
        if book_id in seen:
            continue
        if len(example[TEXT_COLUMN]) < max_required_chars:
            continue
        selected.append(index)
        seen.add(book_id)
        if len(selected) % 1000 == 0:
            print_rank_0("Selected {:,} source books for processing".format(len(selected)))
        if len(selected) >= num_books:
            break
    return selected


def broadcast_object(obj):
    payload = [obj] if dist.get_rank() == 0 else [None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def strip_gutenberg_boilerplate(text: str) -> str:
    start = 0
    end = len(text)

    for pattern in GUTENBERG_START_PATTERNS:
        match = pattern.search(text)
        if match:
            start = max(start, match.end())
            break

    trailing_candidates = []
    for pattern in GUTENBERG_END_PATTERNS:
        match = pattern.search(text, pos=start)
        if match:
            trailing_candidates.append(match.start())
    if trailing_candidates:
        end = min(end, min(trailing_candidates))

    stripped = text[start:end].strip()
    return stripped if stripped else text


def prepare_book_tokens(
    tokenizer: AutoTokenizer,
    task: BookTask,
    char_pos_start: int,
    char_pos_end: int,
    num_tokens: int,
):
    cleaned_text = strip_gutenberg_boilerplate(task.raw_text)
    window_text = cleaned_text[char_pos_start:char_pos_end]
    token_ids = tokenizer.encode(window_text, add_special_tokens=False)
    if len(token_ids) < num_tokens:
        return None
    return PreparedBook(book_id=task.book_id, token_ids=token_ids)


def iter_window_positions(
    token_count: int,
    num_tokens: int,
    window_stride_tokens: int,
    max_windows_per_book: int,
) -> Iterable[Tuple[int, int]]:
    stride = max(1, window_stride_tokens)
    last_start = token_count - num_tokens
    window_index = 0
    for start in range(0, last_start + 1, stride):
        yield window_index, start
        window_index += 1
        if max_windows_per_book > 0 and window_index >= max_windows_per_book:
            break


def score_candidate_batch(
    model,
    batch_rows: Sequence[dict],
    eod_token: int,
    pad_token: Optional[int],
) -> List[float]:
    sequence = torch.tensor(
        [row[INPUT_IDS_COLUMN] for row in batch_rows],
        dtype=torch.long,
        device=torch.cuda.current_device(),
    )
    tokens = sequence[:, :-1].contiguous()
    labels = sequence[:, 1:].contiguous()

    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=eod_token,
        pad_token=pad_token if pad_token is not None else eod_token,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        pad_mask_loss=pad_token is not None,
    )
    losses = model(tokens, position_ids, attention_mask, labels=labels)
    valid_mask = torch.ones_like(labels, dtype=torch.float32)
    if pad_token is not None:
        valid_mask *= (labels != pad_token).float()
    valid_mask *= (labels != eod_token).float()
    token_loss_sum = (losses * valid_mask).sum(dim=1)
    token_count = valid_mask.sum(dim=1).clamp_min(1.0)
    mean_nll = token_loss_sum / token_count
    ppls = torch.exp(mean_nll.clamp(max=50.0)).tolist()
    return [float(value) if value < math.exp(50.0) else float("inf") for value in ppls]


def build_model():
    args = get_args()
    model = get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)
    load_checkpoint(model, None, None, strict=False)
    assert len(model) == 1, "Expected one model chunk per rank"
    model = model[0]
    model.eval()
    log("Loaded checkpoint from {}".format(args.load))
    return model


def apply_threshold_filters(
    rows: List[dict],
    memorized_max_ppl: Optional[float],
    max_allowed_ppl: Optional[float],
    drop_low_ppl_fraction: Optional[float],
) -> List[dict]:
    kept = []
    for row in rows:
        if memorized_max_ppl is not None and row["ppl"] <= memorized_max_ppl:
            continue
        if max_allowed_ppl is not None and row["ppl"] > max_allowed_ppl:
            continue
        kept.append(row)

    kept.sort(key=lambda row: (row["ppl"], row[EXCERPT_ID_COLUMN]))
    if drop_low_ppl_fraction:
        drop_count = int(len(kept) * drop_low_ppl_fraction)
        kept = kept[drop_count:]
    return kept


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def merge_rank_outputs(
    tmp_dir: Path,
    output_dir: Path,
    checkpoint_path: str,
    args,
) -> None:
    stale_outputs = [
        output_dir / "token.jsonl",
        output_dir / "text.jsonl",
        output_dir / "summary.json",
        output_dir / "semantic_dedup_removed_token.jsonl",
        output_dir / "semantic_dedup_removed_text.jsonl",
        output_dir / "semantic_dedup_clusters.jsonl",
        output_dir / "semantic_dedup_pca.jsonl",
        output_dir / "semantic_dedup_embeddings.pt",
    ]
    for path in stale_outputs:
        if path.exists():
            path.unlink()

    best_rows: List[dict] = []
    for path in sorted(tmp_dir.glob("rank*_best_rows.jsonl")):
        best_rows.extend(load_jsonl(path))

    best_rows.sort(key=lambda row: (row["ppl"], row[EXCERPT_ID_COLUMN]))
    threshold_rows = apply_threshold_filters(
        rows=best_rows,
        memorized_max_ppl=args.memorized_max_ppl,
        max_allowed_ppl=args.max_allowed_ppl,
        drop_low_ppl_fraction=args.drop_low_ppl_fraction,
    )
    decode_tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer_path(args.tokenizer_path), trust_remote_code=True)
    decode_tokenizer.model_max_length = 200_000
    for row in threshold_rows:
        row[TEXT_COLUMN] = decode_tokenizer.decode(row[INPUT_IDS_COLUMN], skip_special_tokens=False)

    candidate_count = 0
    with (output_dir / "scores.jsonl").open("w", encoding="utf-8") as score_handle:
        for path in sorted(tmp_dir.glob("rank*_candidate_scores.jsonl")):
            for row in iter_jsonl(path):
                score_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                candidate_count += 1

    thresholded_token_rows = []
    thresholded_text_rows = []
    for row in threshold_rows:
        thresholded_token_rows.append(
            {
                EXCERPT_ID_COLUMN: row[EXCERPT_ID_COLUMN],
                SOURCE_BOOK_ID_COLUMN: row[SOURCE_BOOK_ID_COLUMN],
                INPUT_IDS_COLUMN: row[INPUT_IDS_COLUMN],
                "ppl": row["ppl"],
                WINDOW_INDEX_COLUMN: row[WINDOW_INDEX_COLUMN],
                WINDOW_START_TOKEN_COLUMN: row[WINDOW_START_TOKEN_COLUMN],
            }
        )
        thresholded_text_rows.append(
            {
                EXCERPT_ID_COLUMN: row[EXCERPT_ID_COLUMN],
                SOURCE_BOOK_ID_COLUMN: row[SOURCE_BOOK_ID_COLUMN],
                TEXT_COLUMN: row[TEXT_COLUMN],
                "ppl": row["ppl"],
                WINDOW_INDEX_COLUMN: row[WINDOW_INDEX_COLUMN],
                WINDOW_START_TOKEN_COLUMN: row[WINDOW_START_TOKEN_COLUMN],
            }
        )

    write_jsonl(output_dir / "thresholded_token.jsonl", thresholded_token_rows)
    write_jsonl(output_dir / "thresholded_text.jsonl", thresholded_text_rows)

    summary = {
        "output_dir": str(output_dir),
        "checkpoint": checkpoint_path,
        "num_candidate_windows": candidate_count,
        "num_best_per_book": len(best_rows),
        "num_after_threshold_filters": len(threshold_rows),
        "num_books_requested": args.num_books,
        "preprocess_workers_per_rank": args.preprocess_workers,
        "score_batch_size": args.score_batch_size,
        "window_stride_tokens": args.window_stride_tokens,
        "memorized_max_ppl": args.memorized_max_ppl,
        "max_allowed_ppl": args.max_allowed_ppl,
        "drop_low_ppl_fraction": args.drop_low_ppl_fraction,
        "thresholded_excerpt_ids": sorted(row[EXCERPT_ID_COLUMN] for row in threshold_rows),
    }
    with (output_dir / "build_filter_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    log(
        "Merged {:,} candidate windows into {:,} thresholded excerpts under {}".format(
            candidate_count,
            len(threshold_rows),
            output_dir,
        )
    )


@torch.inference_mode()
def main() -> None:
    initialize_megatron(
        extra_args_provider=add_extra_args,
        args_defaults={
            "no_load_rng": True,
            "no_load_optim": True,
            "micro_batch_size": 1,
            "exit_on_missing_checkpoint": True,
        },
    )
    args = get_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the combined Gutenberg build/filter stage.")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    output_dir = Path(args.output_dir)
    checkpoint_dir = ensure_checkpoint_is_read_only(args.load, output_dir)
    tmp_dir = output_dir / ".tmp_build_filter"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    cache_dir = resolve_cache_dir(args.cache_dir)
    dataset = load_gutenberg_dataset(cache_dir)
    if rank == 0:
        selected_indices = select_book_indices(
            dataset=dataset,
            num_books=args.num_books,
            min_char_length=args.min_char_length,
            char_pos_end=args.char_pos_end,
        )
        print_rank_0("Selected {:,} source books for distributed processing".format(len(selected_indices)))
    else:
        selected_indices = None
    selected_indices = broadcast_object(selected_indices)
    assigned_indices = selected_indices[rank::world_size]

    tokenizer_path = resolve_tokenizer_path(args.tokenizer_path)
    cpu_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    cpu_tokenizer.model_max_length = 200_000
    model = build_model()
    model_tokenizer = get_tokenizer()

    candidate_tmp_path = tmp_dir / "rank{:02d}_candidate_scores.jsonl".format(rank)
    best_tmp_path = tmp_dir / "rank{:02d}_best_rows.jsonl".format(rank)

    pending = {}
    task_iter = iter(assigned_indices)
    local_books = 0
    local_windows = 0
    local_best = 0
    start_time = time.time()
    log(
        "Starting build_filter with {:,} assigned books | {} CPU workers | batch_size={} | stride={}".format(
            len(assigned_indices),
            max(1, args.preprocess_workers),
            max(1, args.score_batch_size),
            args.window_stride_tokens,
        )
    )

    def submit_until_full(executor: ThreadPoolExecutor) -> None:
        while len(pending) < max(1, args.preprocess_workers * 2):
            try:
                dataset_index = next(task_iter)
            except StopIteration:
                break
            example = dataset[dataset_index]
            task = BookTask(book_id=str(example[BOOK_ID_COLUMN]), raw_text=example[TEXT_COLUMN])
            future = executor.submit(
                prepare_book_tokens,
                cpu_tokenizer,
                task,
                args.char_pos_start,
                args.char_pos_end,
                args.num_tokens,
            )
            pending[future] = task.book_id

    with candidate_tmp_path.open("w", encoding="utf-8") as candidate_handle, best_tmp_path.open(
        "w", encoding="utf-8"
    ) as best_handle:
        with ThreadPoolExecutor(max_workers=max(1, args.preprocess_workers)) as executor:
            submit_until_full(executor)
            while pending:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    book_id = pending.pop(future)
                    prepared_book = future.result()
                    submit_until_full(executor)

                    local_books += 1
                    if prepared_book is None:
                        if args.log_every_books > 0 and local_books % args.log_every_books == 0:
                            elapsed = max(time.time() - start_time, 1e-6)
                            log(
                                "progress books={:,}/{:,} windows_scored={:,} best_books={:,} books_per_s={:.2f} windows_per_s={:.2f}".format(
                                    local_books,
                                    len(assigned_indices),
                                    local_windows,
                                    local_best,
                                    local_books / elapsed,
                                    local_windows / elapsed,
                                )
                            )
                        continue

                    best_row = None
                    batch_rows: List[dict] = []
                    for window_index, start in iter_window_positions(
                        token_count=len(prepared_book.token_ids),
                        num_tokens=args.num_tokens,
                        window_stride_tokens=args.window_stride_tokens,
                        max_windows_per_book=args.max_windows_per_book,
                    ):
                        selected_ids = prepared_book.token_ids[start : start + args.num_tokens]
                        if len(selected_ids) != args.num_tokens:
                            continue
                        batch_rows.append(
                            {
                                EXCERPT_ID_COLUMN: "{}::window_{:04d}".format(prepared_book.book_id, window_index),
                                SOURCE_BOOK_ID_COLUMN: prepared_book.book_id,
                                INPUT_IDS_COLUMN: selected_ids,
                                TOKEN_LENGTH_COLUMN: len(selected_ids),
                                WINDOW_INDEX_COLUMN: window_index,
                                WINDOW_START_TOKEN_COLUMN: start,
                            }
                        )
                        if len(batch_rows) < max(1, args.score_batch_size):
                            continue

                        ppls = score_candidate_batch(
                            model=model,
                            batch_rows=batch_rows,
                            eod_token=model_tokenizer.eod,
                            pad_token=getattr(model_tokenizer, "pad", None),
                        )
                        for candidate, ppl in zip(batch_rows, ppls):
                            candidate["ppl"] = ppl
                            candidate_handle.write(
                                json.dumps(
                                    {
                                        EXCERPT_ID_COLUMN: candidate[EXCERPT_ID_COLUMN],
                                        SOURCE_BOOK_ID_COLUMN: candidate[SOURCE_BOOK_ID_COLUMN],
                                        WINDOW_INDEX_COLUMN: candidate[WINDOW_INDEX_COLUMN],
                                        WINDOW_START_TOKEN_COLUMN: candidate[WINDOW_START_TOKEN_COLUMN],
                                        TOKEN_LENGTH_COLUMN: candidate[TOKEN_LENGTH_COLUMN],
                                        "ppl": candidate["ppl"],
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            if best_row is None or candidate["ppl"] > best_row["ppl"] or (
                                candidate["ppl"] == best_row["ppl"]
                                and candidate[EXCERPT_ID_COLUMN] < best_row[EXCERPT_ID_COLUMN]
                            ):
                                best_row = {
                                    EXCERPT_ID_COLUMN: candidate[EXCERPT_ID_COLUMN],
                                    SOURCE_BOOK_ID_COLUMN: candidate[SOURCE_BOOK_ID_COLUMN],
                                    INPUT_IDS_COLUMN: candidate[INPUT_IDS_COLUMN],
                                    TOKEN_LENGTH_COLUMN: candidate[TOKEN_LENGTH_COLUMN],
                                    WINDOW_INDEX_COLUMN: candidate[WINDOW_INDEX_COLUMN],
                                    WINDOW_START_TOKEN_COLUMN: candidate[WINDOW_START_TOKEN_COLUMN],
                                    "ppl": candidate["ppl"],
                                }
                            local_windows += 1
                        batch_rows = []

                    if batch_rows:
                        ppls = score_candidate_batch(
                            model=model,
                            batch_rows=batch_rows,
                            eod_token=model_tokenizer.eod,
                            pad_token=getattr(model_tokenizer, "pad", None),
                        )
                        for candidate, ppl in zip(batch_rows, ppls):
                            candidate["ppl"] = ppl
                            candidate_handle.write(
                                json.dumps(
                                    {
                                        EXCERPT_ID_COLUMN: candidate[EXCERPT_ID_COLUMN],
                                        SOURCE_BOOK_ID_COLUMN: candidate[SOURCE_BOOK_ID_COLUMN],
                                        WINDOW_INDEX_COLUMN: candidate[WINDOW_INDEX_COLUMN],
                                        WINDOW_START_TOKEN_COLUMN: candidate[WINDOW_START_TOKEN_COLUMN],
                                        TOKEN_LENGTH_COLUMN: candidate[TOKEN_LENGTH_COLUMN],
                                        "ppl": candidate["ppl"],
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            if best_row is None or candidate["ppl"] > best_row["ppl"] or (
                                candidate["ppl"] == best_row["ppl"]
                                and candidate[EXCERPT_ID_COLUMN] < best_row[EXCERPT_ID_COLUMN]
                            ):
                                best_row = {
                                    EXCERPT_ID_COLUMN: candidate[EXCERPT_ID_COLUMN],
                                    SOURCE_BOOK_ID_COLUMN: candidate[SOURCE_BOOK_ID_COLUMN],
                                    INPUT_IDS_COLUMN: candidate[INPUT_IDS_COLUMN],
                                    TOKEN_LENGTH_COLUMN: candidate[TOKEN_LENGTH_COLUMN],
                                    WINDOW_INDEX_COLUMN: candidate[WINDOW_INDEX_COLUMN],
                                    WINDOW_START_TOKEN_COLUMN: candidate[WINDOW_START_TOKEN_COLUMN],
                                    "ppl": candidate["ppl"],
                                }
                            local_windows += 1

                    if best_row is None:
                        if args.log_every_books > 0 and local_books % args.log_every_books == 0:
                            elapsed = max(time.time() - start_time, 1e-6)
                            log(
                                "progress books={:,}/{:,} windows_scored={:,} best_books={:,} books_per_s={:.2f} windows_per_s={:.2f}".format(
                                    local_books,
                                    len(assigned_indices),
                                    local_windows,
                                    local_best,
                                    local_books / elapsed,
                                    local_windows / elapsed,
                                )
                            )
                        continue

                    best_handle.write(json.dumps(best_row, ensure_ascii=False) + "\n")
                    local_best += 1

                    if args.log_every_books > 0 and local_books % args.log_every_books == 0:
                        elapsed = max(time.time() - start_time, 1e-6)
                        log(
                            "progress books={:,}/{:,} windows_scored={:,} best_books={:,} books_per_s={:.2f} windows_per_s={:.2f}".format(
                                local_books,
                                len(assigned_indices),
                                local_windows,
                                local_best,
                                local_books / elapsed,
                                local_windows / elapsed,
                            )
                        )

    # Release the checkpoint-scoring model before the semantic dedup stage loads its own encoder.
    del model
    torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        merge_rank_outputs(
            tmp_dir=tmp_dir,
            output_dir=output_dir,
            checkpoint_path=str(checkpoint_dir),
            args=args,
        )
    dist.barrier()


if __name__ == "__main__":
    main()
