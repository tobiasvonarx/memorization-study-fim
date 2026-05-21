#!/usr/bin/env python3
"""Replicate filtered Gutenberg excerpts into memorization buckets.

The active pipeline expects a filtered LLaMA-tokenized base set. If score fields
are present, we can optionally rebalance ordering before slicing contiguous
repetition buckets.
"""

import argparse
import json
import time
import heapq
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_REPETITIONS = [1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128]
SEMANTIC_DEDUP_REQUIRED_FILES = [
    "token.jsonl",
    "text.jsonl",
    "semantic_dedup_removed_token.jsonl",
    "semantic_dedup_removed_text.jsonl",
    "semantic_dedup_clusters.jsonl",
]
SEMANTIC_DEDUP_REQUIRED_COLUMNS = [
    "semantic_dedup_cluster_id",
    "semantic_dedup_cluster_size",
    "semantic_dedup_representative_excerpt_id",
    "semantic_dedup_similarity_to_representative",
    "semantic_dedup_jaccard_to_representative",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replicate Gutenberg excerpts by repetition bucket.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with filtered token/text JSONL files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for replicated rep_* JSONL files")
    parser.add_argument(
        "--bucket-size",
        type=str,
        default="auto",
        help="Number of base excerpts per repetition bucket, or 'auto' to use floor(num_filtered_excerpts / num_buckets).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        nargs="+",
        default=DEFAULT_REPETITIONS,
        help="Repetition schedule.",
    )
    parser.add_argument(
        "--score-field",
        type=str,
        default="ppl",
        help="Optional score field used for balanced ordering when present.",
    )
    parser.add_argument(
        "--balance-mode",
        choices=["none", "snake_by_score", "equalize_bucket_mean"],
        default="equalize_bucket_mean",
        help="How to reorder the filtered base set before bucket slicing.",
    )
    parser.add_argument(
        "--max-score-mean-spread-ratio",
        type=float,
        default=0.01,
        help=(
            "Fail if max(bucket mean score) - min(bucket mean score), divided by the global mean, "
            "exceeds this ratio. Set negative to disable."
        ),
    )
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print("[{}][create_replicas] {}".format(timestamp, message), flush=True)


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSONL file: {path}")
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ensure_semantic_dedup_inputs(input_dir: Path, token_rows: List[dict], text_rows: List[dict]) -> None:
    missing_paths = [str(input_dir / name) for name in SEMANTIC_DEDUP_REQUIRED_FILES if not (input_dir / name).exists()]
    if missing_paths:
        raise FileNotFoundError(
            "create_replicas must run after semantic_dedup; missing required semantic-dedup outputs: {}".format(
                ", ".join(missing_paths)
            )
        )

    thresholded_paths = [input_dir / "thresholded_token.jsonl", input_dir / "thresholded_text.jsonl"]
    existing_thresholded_paths = [path for path in thresholded_paths if path.exists()]
    if existing_thresholded_paths:
        dedup_mtime = min((input_dir / "token.jsonl").stat().st_mtime, (input_dir / "text.jsonl").stat().st_mtime)
        newest_thresholded_mtime = max(path.stat().st_mtime for path in existing_thresholded_paths)
        if dedup_mtime < newest_thresholded_mtime:
            raise RuntimeError(
                "create_replicas would use stale semantic-dedup outputs: token.jsonl/text.jsonl are older than "
                "thresholded_token.jsonl/thresholded_text.jsonl. Rerun PIPELINE_STAGE=semantic_dedup first."
            )

    for file_label, rows in [("token.jsonl", token_rows), ("text.jsonl", text_rows)]:
        if not rows:
            raise ValueError(f"{file_label} is empty; cannot create repetition buckets")
        missing_columns = [column for column in SEMANTIC_DEDUP_REQUIRED_COLUMNS if column not in rows[0]]
        if missing_columns:
            raise ValueError(
                "{} does not look like post-semantic-dedup output; missing columns: {}. "
                "create_replicas intentionally consumes token.jsonl/text.jsonl, not thresholded_*.jsonl.".format(
                    file_label,
                    ", ".join(missing_columns),
                )
            )


def snake_order(rows: List[dict], score_field: str) -> List[dict]:
    sortable = sorted(rows, key=lambda row: (row.get(score_field, 0.0), row["excerpt_id"]))
    ordered: List[dict] = []
    left = 0
    right = len(sortable) - 1
    while left <= right:
        ordered.append(sortable[left])
        left += 1
        if left <= right:
            ordered.append(sortable[right])
            right -= 1
    return ordered


def equalize_bucket_mean(rows: List[dict], score_field: str, num_buckets: int, bucket_size: int) -> List[List[dict]]:
    """Partition rows into equal-size buckets with similar average score.

    We sort by descending score and greedily place each excerpt into the
    currently lightest bucket that still has capacity. Because every bucket has
    the same target size, equalizing score totals also equalizes the average
    score per bucket.
    """
    sortable = sorted(rows, key=lambda row: (-float(row.get(score_field, 0.0)), row["excerpt_id"]))
    buckets: List[List[dict]] = [[] for _ in range(num_buckets)]
    bucket_totals = [0.0 for _ in range(num_buckets)]

    # Heap entries: (current_total, current_count, bucket_index)
    heap: List[Tuple[float, int, int]] = [(0.0, 0, index) for index in range(num_buckets)]
    heapq.heapify(heap)

    for row in sortable:
        skipped: List[Tuple[float, int, int]] = []
        chosen_total = None
        chosen_count = None
        chosen_index = None

        while heap:
            current_total, current_count, bucket_index = heapq.heappop(heap)
            if current_count < bucket_size:
                chosen_total = current_total
                chosen_count = current_count
                chosen_index = bucket_index
                break
            skipped.append((current_total, current_count, bucket_index))

        for entry in skipped:
            heapq.heappush(heap, entry)

        if chosen_index is None:
            raise ValueError("Ran out of bucket capacity while balancing by score")

        buckets[chosen_index].append(row)
        new_total = chosen_total + float(row.get(score_field, 0.0))
        bucket_totals[chosen_index] = new_total
        heapq.heappush(heap, (new_total, chosen_count + 1, chosen_index))

    for bucket in buckets:
        bucket.sort(key=lambda row: row["excerpt_id"])

    return buckets


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def expand_replicas(rows: List[dict], repetition: int) -> List[dict]:
    """Duplicate rows while preserving per-copy metadata for FIM-v2 splits."""
    replicated_rows: List[dict] = []
    for replica_index in range(repetition):
        for bucket_base_index, row in enumerate(rows):
            copied = dict(row)
            copied.setdefault("base_excerpt_id", row["excerpt_id"])
            copied["repetition_bucket"] = repetition
            copied["replica_count"] = repetition
            copied["replica_index"] = replica_index
            copied["bucket_base_index"] = bucket_base_index
            replicated_rows.append(copied)
    return replicated_rows


def resolve_bucket_size(raw_bucket_size: str, num_rows: int, num_buckets: int) -> int:
    if raw_bucket_size.casefold() == "auto":
        bucket_size = num_rows // num_buckets
        if bucket_size < 1:
            raise ValueError(
                f"Cannot auto-size {num_buckets} repetition buckets from only {num_rows} filtered excerpts"
            )
        return bucket_size

    try:
        bucket_size = int(raw_bucket_size)
    except ValueError as error:
        raise ValueError("--bucket-size must be a positive integer or 'auto'") from error
    if bucket_size < 1:
        raise ValueError("--bucket-size must be positive")
    return bucket_size


def ensure_matching_text_rows(token_rows: List[dict], text_rows: List[dict]) -> Dict[str, dict]:
    token_ids = [row["excerpt_id"] for row in token_rows]
    text_by_id = {row["excerpt_id"]: row for row in text_rows}
    if len(text_by_id) != len(text_rows):
        raise ValueError("text.jsonl contains duplicate excerpt_id values")

    missing_text_ids = [excerpt_id for excerpt_id in token_ids if excerpt_id not in text_by_id]
    if missing_text_ids:
        preview = ", ".join(missing_text_ids[:5])
        raise ValueError(f"text.jsonl is missing {len(missing_text_ids)} token excerpt ids; first few: {preview}")

    extra_text_ids = sorted(set(text_by_id).difference(token_ids))
    if extra_text_ids:
        preview = ", ".join(extra_text_ids[:5])
        raise ValueError(f"text.jsonl has {len(extra_text_ids)} excerpt ids absent from token.jsonl; first few: {preview}")

    return text_by_id


def require_score_field(rows: List[dict], score_field: str) -> None:
    missing = [row["excerpt_id"] for row in rows if score_field not in row]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Cannot balance buckets by {score_field}; missing in {len(missing)} rows. First few: {preview}")


def bucket_score_stats(buckets: List[List[dict]], score_field: str) -> Dict[str, Any]:
    bucket_means = []
    bucket_sizes = []
    for bucket in buckets:
        bucket_sizes.append(len(bucket))
        scores = [float(row[score_field]) for row in bucket]
        bucket_means.append(sum(scores) / len(scores) if scores else 0.0)

    global_mean = sum(bucket_means) / len(bucket_means) if bucket_means else 0.0
    mean_min = min(bucket_means) if bucket_means else None
    mean_max = max(bucket_means) if bucket_means else None
    mean_spread = (mean_max - mean_min) if mean_min is not None and mean_max is not None else None
    mean_spread_ratio = (mean_spread / global_mean) if mean_spread is not None and global_mean else 0.0
    return {
        "bucket_sizes": bucket_sizes,
        "score_field": score_field,
        "score_mean_global": global_mean,
        "score_mean_min": mean_min,
        "score_mean_max": mean_max,
        "score_mean_spread": mean_spread,
        "score_mean_spread_ratio": mean_spread_ratio,
    }


def validate_buckets(
    buckets: List[List[dict]],
    repetitions: List[int],
    bucket_size: int,
    score_field: str,
    max_score_mean_spread_ratio: float,
) -> Dict[str, Any]:
    bucket_sizes = [len(bucket) for bucket in buckets]
    if bucket_sizes != [bucket_size] * len(repetitions):
        raise ValueError(f"Replica buckets are not equal sized: expected {bucket_size}, got {bucket_sizes}")

    excerpt_ids = [row["excerpt_id"] for bucket in buckets for row in bucket]
    if len(set(excerpt_ids)) != len(excerpt_ids):
        raise ValueError("Replica buckets are not disjoint; at least one base excerpt appears in multiple buckets")

    require_score_field([row for bucket in buckets for row in bucket], score_field)
    stats = bucket_score_stats(buckets, score_field)
    spread_ratio = float(stats["score_mean_spread_ratio"])
    if max_score_mean_spread_ratio >= 0 and spread_ratio > max_score_mean_spread_ratio:
        raise ValueError(
            "Bucket mean {} spread ratio {:.6f} exceeds allowed {:.6f}".format(
                score_field,
                spread_ratio,
                max_score_mean_spread_ratio,
            )
        )
    return stats


def write_bucket_summary(
    output_dir: Path,
    repetitions: List[int],
    buckets: List[List[dict]],
    score_field: str,
    balance_stats: Dict[str, Any],
) -> None:
    summary_rows = []
    for repetition, bucket in zip(repetitions, buckets):
        scores = [float(row[score_field]) for row in bucket]
        score_sum = sum(scores)
        score_mean = score_sum / len(scores) if scores else 0.0
        summary_rows.append(
            {
                "repetition": repetition,
                "num_base_excerpts": len(bucket),
                "score_field": score_field,
                "score_sum": score_sum,
                "score_mean": score_mean,
                "score_min": min(scores) if scores else None,
                "score_max": max(scores) if scores else None,
                "excerpt_ids_preview": [row["excerpt_id"] for row in bucket[:5]],
            }
        )

    with (output_dir / "bucket_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)
    with (output_dir / "bucket_balance_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(balance_stats, handle, indent=2)
    log(
        "Wrote bucket summary for {:,} repetition buckets to {}".format(
            len(summary_rows),
            output_dir / "bucket_summary.json",
        )
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(
        "Starting create_replicas with input_dir={} output_dir={} bucket_size={} repetitions={} balance_mode={} score_field={}".format(
            input_dir,
            output_dir,
            args.bucket_size,
            args.repetitions,
            args.balance_mode,
            args.score_field,
        )
    )

    token_rows = load_jsonl(input_dir / "token.jsonl")
    text_rows = load_jsonl(input_dir / "text.jsonl")
    ensure_semantic_dedup_inputs(input_dir, token_rows, text_rows)
    log(
        "Loaded {:,} post-dedup token rows and {:,} post-dedup text rows from {}".format(
            len(token_rows),
            len(text_rows),
            input_dir,
        )
    )
    text_by_id = ensure_matching_text_rows(token_rows, text_rows)

    bucket_size = resolve_bucket_size(
        raw_bucket_size=args.bucket_size,
        num_rows=len(token_rows),
        num_buckets=len(args.repetitions),
    )
    if args.bucket_size.casefold() == "auto":
        log("Resolved auto bucket_size={} from {:,} filtered excerpts".format(bucket_size, len(token_rows)))

    required = len(args.repetitions) * bucket_size
    log(
        "Replica capacity check: need {:,} base excerpts for {:,} buckets, have {:,}".format(
            required,
            len(args.repetitions),
            len(token_rows),
        )
    )
    if len(token_rows) < required:
        raise ValueError(
            f"Need at least {required} filtered excerpts for bucket_size={bucket_size} "
            f"and repetitions={args.repetitions}, but only found {len(token_rows)}"
        )

    unused_rows = len(token_rows) - required
    if unused_rows:
        log("Leaving {:,} filtered excerpts unused so all {:,} buckets have exactly {:,} excerpts".format(
            unused_rows,
            len(args.repetitions),
            bucket_size,
        ))

    usable_rows = token_rows[:required]
    if args.balance_mode == "equalize_bucket_mean":
        require_score_field(usable_rows, args.score_field)
        log("Balancing usable excerpts with equalize_bucket_mean over {}".format(args.score_field))
        buckets = equalize_bucket_mean(
            rows=usable_rows,
            score_field=args.score_field,
            num_buckets=len(args.repetitions),
            bucket_size=bucket_size,
        )
    else:
        if args.balance_mode == "snake_by_score":
            require_score_field(usable_rows, args.score_field)
            log("Balancing usable excerpts with snake_by_score over {}".format(args.score_field))
            ordered_rows = snake_order(usable_rows, args.score_field)
        else:
            log("Using excerpt_id ordering without score balancing")
            ordered_rows = sorted(usable_rows, key=lambda row: row["excerpt_id"])
        buckets = []
        for bucket_index in range(len(args.repetitions)):
            start = bucket_index * bucket_size
            end = start + bucket_size
            buckets.append(ordered_rows[start:end])

    balance_stats = validate_buckets(
        buckets=buckets,
        repetitions=args.repetitions,
        bucket_size=bucket_size,
        score_field=args.score_field,
        max_score_mean_spread_ratio=args.max_score_mean_spread_ratio,
    )
    log(
        "Bucket balance: size={} mean {} min={:.6f} max={:.6f} spread={:.6f} spread_ratio={:.6f}".format(
            bucket_size,
            args.score_field,
            balance_stats["score_mean_min"],
            balance_stats["score_mean_max"],
            balance_stats["score_mean_spread"],
            balance_stats["score_mean_spread_ratio"],
        )
    )

    for repetition, token_slice in zip(args.repetitions, buckets):
        text_slice = [text_by_id[row["excerpt_id"]] for row in token_slice]

        replicated_token_rows = expand_replicas(token_slice, repetition)
        replicated_text_rows = expand_replicas(text_slice, repetition)

        write_jsonl(output_dir / f"rep_{repetition}_token.jsonl", replicated_token_rows)
        write_jsonl(output_dir / f"rep_{repetition}_text.jsonl", replicated_text_rows)
        score_values = [float(row.get(args.score_field, 0.0)) for row in token_slice if args.score_field in row]
        score_mean = (sum(score_values) / len(score_values)) if score_values else None
        log(
            "Wrote rep_{} files with {:,} base excerpts and {:,} replicated rows{}".format(
                repetition,
                len(token_slice),
                len(replicated_token_rows),
                "" if score_mean is None else " (mean {}={:.4f})".format(args.score_field, score_mean),
            )
        )

    write_bucket_summary(
        output_dir=output_dir,
        repetitions=args.repetitions,
        buckets=buckets,
        score_field=args.score_field,
        balance_stats=balance_stats,
    )

    log(
        "Completed replica creation in {} using {:,} filtered base excerpts".format(
            output_dir,
            len(token_rows),
        )
    )


if __name__ == "__main__":
    main()
