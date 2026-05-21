"""Build compact unique-excerpt eval files for repetition buckets.

The replica-expanded Gutenberg files encode bucket membership but repeat every
base excerpt `N` times. Evaluation should probe each unique excerpt once, so
this module creates compact `rep_N_token.jsonl` files from the final unique
token source. When `bucket_summary.json` is available, bucket membership is
reconstructed from the original balancing algorithm instead of scanning the
large replica-expanded JSONLs.
"""

from __future__ import annotations

import json
import os
import re
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VERSION = 2
DEFAULT_BUCKET_REPETITIONS = [1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128]
DEFAULT_SCORE_FIELD = "ppl"
STRING_FIELD_RE = {
    "excerpt_id": re.compile(r'"excerpt_id"\s*:\s*"([^"]*)"'),
    "base_excerpt_id": re.compile(r'"base_excerpt_id"\s*:\s*"([^"]*)"'),
}
INT_FIELD_RE = {
    "repetition_bucket": re.compile(r'"repetition_bucket"\s*:\s*(-?\d+|null)'),
    "replica_count": re.compile(r'"replica_count"\s*:\s*(-?\d+|null)'),
    "bucket_base_index": re.compile(r'"bucket_base_index"\s*:\s*(-?\d+|null)'),
}


@dataclass(frozen=True)
class CacheBuildResult:
    repetition: int
    path: Path
    status: str
    raw_replica_rows: int
    num_assignments: int
    num_written: int


def source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def optional_source_signature(path: Path) -> dict[str, Any] | None:
    return source_signature(path) if path.exists() else None


def output_path(cache_dir: Path, repetition: int) -> Path:
    return cache_dir / f"rep_{repetition}_token.jsonl"


def metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def _extract_string(line: str, field: str) -> str | None:
    match = STRING_FIELD_RE[field].search(line)
    return match.group(1) if match else None


def _extract_int(line: str, field: str) -> int | None:
    match = INT_FIELD_RE[field].search(line)
    if not match or match.group(1) == "null":
        return None
    return int(match.group(1))


def _assignment_from_line(line: str, repetition: int, default_bucket_index: int) -> dict[str, Any]:
    base_excerpt_id = _extract_string(line, "base_excerpt_id") or _extract_string(line, "excerpt_id")
    repetition_bucket = _extract_int(line, "repetition_bucket")
    replica_count = _extract_int(line, "replica_count")
    bucket_base_index = _extract_int(line, "bucket_base_index")

    if base_excerpt_id is None:
        obj = json.loads(line)
        base_excerpt_id = str(obj.get("base_excerpt_id", obj["excerpt_id"]))
        repetition_bucket = obj.get("repetition_bucket", repetition)
        replica_count = obj.get("replica_count", repetition)
        bucket_base_index = obj.get("bucket_base_index", default_bucket_index)

    repetition_bucket = int(repetition if repetition_bucket is None else repetition_bucket)
    replica_count = int(repetition if replica_count is None else replica_count)
    bucket_base_index = int(default_bucket_index if bucket_base_index is None else bucket_base_index)

    return {
        "base_excerpt_id": str(base_excerpt_id),
        "repetition_bucket": repetition_bucket,
        "replica_count": replica_count,
        "bucket_base_index": bucket_base_index,
    }


def load_bucket_assignments(replica_bucket_file: Path, repetition: int) -> tuple[dict[str, dict[str, Any]], int]:
    assignments: dict[str, dict[str, Any]] = {}
    raw_rows = 0
    with replica_bucket_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw_rows += 1
            if '"replica_index"' in line and '"replica_index": 0' not in line and '"replica_index":0' not in line:
                continue
            assignment = _assignment_from_line(line, repetition, len(assignments))
            base_excerpt_id = assignment["base_excerpt_id"]
            assignments.setdefault(base_excerpt_id, assignment)
    if not assignments:
        raise RuntimeError(f"No unique assignments found in {replica_bucket_file}")
    return assignments, raw_rows


def bucket_summary_path(replica_buckets_dir: Path) -> Path:
    return Path(replica_buckets_dir) / "bucket_summary.json"


def read_bucket_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bucket_repetitions_from_summary(summary: list[dict[str, Any]], requested: list[int]) -> list[int]:
    if summary:
        return [int(row["repetition"]) for row in summary]
    return sorted(set(requested)) or DEFAULT_BUCKET_REPETITIONS


def load_unique_bucket_rows(unique_token_file: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with unique_token_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            rows.append(
                {
                    "excerpt_id": str(obj["excerpt_id"]),
                    "score": float(obj.get(DEFAULT_SCORE_FIELD, 0.0)),
                }
            )
    return rows


def reconstruct_equalized_bucket_assignments(
    unique_token_file: Path,
    bucket_repetitions: list[int],
) -> dict[int, dict[str, dict[str, Any]]]:
    rows = load_unique_bucket_rows(unique_token_file)
    bucket_count = len(bucket_repetitions)
    bucket_size = len(rows) // bucket_count
    if bucket_size < 1:
        raise ValueError(f"Cannot build {bucket_count} repetition buckets from {len(rows)} unique rows")
    usable_rows = rows[: bucket_size * bucket_count]
    sortable = sorted(usable_rows, key=lambda row: (-float(row["score"]), row["excerpt_id"]))

    buckets: list[list[dict[str, Any]]] = [[] for _ in bucket_repetitions]
    heap: list[tuple[float, int, int]] = [(0.0, 0, index) for index in range(bucket_count)]
    heapq.heapify(heap)

    for row in sortable:
        skipped: list[tuple[float, int, int]] = []
        chosen: tuple[float, int, int] | None = None
        while heap:
            current_total, current_count, bucket_index = heapq.heappop(heap)
            if current_count < bucket_size:
                chosen = (current_total, current_count, bucket_index)
                break
            skipped.append((current_total, current_count, bucket_index))
        for entry in skipped:
            heapq.heappush(heap, entry)
        if chosen is None:
            raise ValueError("Ran out of bucket capacity while reconstructing repetition buckets")
        current_total, current_count, bucket_index = chosen
        buckets[bucket_index].append(row)
        heapq.heappush(heap, (current_total + float(row["score"]), current_count + 1, bucket_index))

    assignments_by_repetition: dict[int, dict[str, dict[str, Any]]] = {}
    for repetition, bucket in zip(bucket_repetitions, buckets):
        bucket.sort(key=lambda row: row["excerpt_id"])
        assignments_by_repetition[repetition] = {
            str(row["excerpt_id"]): {
                "base_excerpt_id": str(row["excerpt_id"]),
                "repetition_bucket": repetition,
                "replica_count": repetition,
                "bucket_base_index": bucket_base_index,
            }
            for bucket_base_index, row in enumerate(bucket)
        }
    return assignments_by_repetition


def validate_reconstructed_summary(
    summary: list[dict[str, Any]],
    assignments_by_repetition: dict[int, dict[str, dict[str, Any]]],
) -> None:
    for row in summary:
        repetition = int(row["repetition"])
        if repetition not in assignments_by_repetition:
            continue
        preview = [str(value) for value in row.get("excerpt_ids_preview", [])]
        actual = list(assignments_by_repetition[repetition])[: len(preview)]
        if preview and actual != preview:
            raise RuntimeError(
                "Reconstructed bucket assignment does not match bucket_summary.json "
                f"for rep={repetition}: expected preview {preview}, got {actual}"
            )


def expected_metadata(
    unique_token_file: Path,
    assignment_source_file: Path,
    repetition: int,
    assignment_mode: str,
    bucket_repetitions: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "repetition": repetition,
        "unique_token_file": source_signature(unique_token_file),
        "assignment_mode": assignment_mode,
        "assignment_source_file": optional_source_signature(assignment_source_file),
        "bucket_repetitions": bucket_repetitions,
        "score_field": DEFAULT_SCORE_FIELD,
    }


def cache_file_valid(
    output_file: Path,
    unique_token_file: Path,
    assignment_source_file: Path,
    repetition: int,
    assignment_mode: str,
    bucket_repetitions: list[int] | None = None,
) -> bool:
    meta_file = metadata_path(output_file)
    if not output_file.exists() or output_file.stat().st_size == 0 or not meta_file.exists():
        return False
    try:
        metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        expected = expected_metadata(
            unique_token_file,
            assignment_source_file,
            repetition,
            assignment_mode,
            bucket_repetitions,
        )
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("version") == expected["version"]
        and metadata.get("repetition") == expected["repetition"]
        and metadata.get("unique_token_file") == expected["unique_token_file"]
        and metadata.get("assignment_mode") == expected["assignment_mode"]
        and metadata.get("assignment_source_file") == expected["assignment_source_file"]
        and metadata.get("bucket_repetitions") == expected["bucket_repetitions"]
        and metadata.get("score_field") == expected["score_field"]
    )


def write_compact_files(
    unique_token_file: Path,
    cache_dir: Path,
    assignments_by_repetition: dict[int, dict[str, dict[str, Any]]],
    raw_rows_by_repetition: dict[int, int],
) -> list[CacheBuildResult]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    assignment_by_excerpt: dict[str, dict[str, Any]] = {}
    for repetition, assignments in assignments_by_repetition.items():
        for excerpt_id, assignment in assignments.items():
            if excerpt_id in assignment_by_excerpt:
                raise ValueError(f"Excerpt assigned to multiple repetition buckets: {excerpt_id}")
            assignment_by_excerpt[excerpt_id] = assignment | {"_output_repetition": repetition}

    temp_paths: dict[int, Path] = {}
    handles: dict[int, Any] = {}
    counts = {repetition: 0 for repetition in assignments_by_repetition}
    written_excerpts: dict[int, set[str]] = {repetition: set() for repetition in assignments_by_repetition}
    try:
        for repetition in assignments_by_repetition:
            target = output_path(cache_dir, repetition)
            temp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
            temp_paths[repetition] = temp
            handles[repetition] = temp.open("w", encoding="utf-8")

        with unique_token_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                excerpt_id = _extract_string(line, "excerpt_id")
                if excerpt_id is None or excerpt_id not in assignment_by_excerpt:
                    continue
                assignment = assignment_by_excerpt[excerpt_id]
                repetition = int(assignment["_output_repetition"])
                row = json.loads(line)
                row["base_excerpt_id"] = assignment["base_excerpt_id"]
                row["repetition_bucket"] = assignment["repetition_bucket"]
                row["replica_count"] = assignment["replica_count"]
                row["replica_index"] = 0
                row["bucket_base_index"] = assignment["bucket_base_index"]
                row["unique_eval_compact"] = True
                handles[repetition].write(json.dumps(row, separators=(",", ":")) + "\n")
                counts[repetition] += 1
                written_excerpts[repetition].add(excerpt_id)
    finally:
        for handle in handles.values():
            handle.close()

    results: list[CacheBuildResult] = []
    for repetition, assignments in assignments_by_repetition.items():
        expected_count = len(assignments)
        written = counts[repetition]
        target = output_path(cache_dir, repetition)
        temp = temp_paths[repetition]
        if written != expected_count:
            temp.unlink(missing_ok=True)
            missing = sorted(set(assignments) - written_excerpts[repetition])
            preview = ", ".join(missing[:5])
            raise RuntimeError(
                f"Unique source wrote {written}/{expected_count} rows for rep={repetition}. "
                f"Check that {unique_token_file} matches the bucket source. Missing preview: {preview}"
            )
        temp.replace(target)
        results.append(
            CacheBuildResult(
                repetition=repetition,
                path=target,
                status="built",
                raw_replica_rows=raw_rows_by_repetition[repetition],
                num_assignments=expected_count,
                num_written=written,
            )
        )
    return results


def write_metadata(
    output_file: Path,
    unique_token_file: Path,
    assignment_source_file: Path,
    repetition: int,
    result: CacheBuildResult,
    assignment_mode: str,
    bucket_repetitions: list[int] | None = None,
) -> None:
    payload = expected_metadata(
        unique_token_file,
        assignment_source_file,
        repetition,
        assignment_mode,
        bucket_repetitions,
    )
    payload.update(
        {
            "compact_token_file": str(output_file),
            "raw_replica_rows": result.raw_replica_rows,
            "num_assignments": result.num_assignments,
            "num_written": result.num_written,
        }
    )
    meta_file = metadata_path(output_file)
    tmp_file = meta_file.with_name(f"{meta_file.name}.tmp.{os.getpid()}")
    tmp_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_file.replace(meta_file)


def ensure_unique_repetition_files(
    unique_token_file: Path,
    replica_buckets_dir: Path,
    cache_dir: Path,
    repetitions: list[int],
    *,
    force: bool = False,
) -> list[CacheBuildResult]:
    unique_token_file = Path(unique_token_file)
    replica_buckets_dir = Path(replica_buckets_dir)
    cache_dir = Path(cache_dir)
    summary_file = bucket_summary_path(replica_buckets_dir)
    summary = read_bucket_summary(summary_file)
    bucket_repetitions = bucket_repetitions_from_summary(summary, repetitions)
    assignment_mode = "reconstruct_equalize_bucket_mean" if summary else "scan_replica_jsonl"
    results: list[CacheBuildResult] = []
    missing: list[int] = []
    for repetition in sorted(set(int(value) for value in repetitions)):
        per_rep_assignment_source = (
            summary_file if summary else replica_buckets_dir / f"rep_{repetition}_token.jsonl"
        )
        target = output_path(cache_dir, repetition)
        if not force and cache_file_valid(
            target,
            unique_token_file,
            per_rep_assignment_source,
            repetition,
            assignment_mode,
            bucket_repetitions if summary else None,
        ):
            results.append(
                CacheBuildResult(
                    repetition=repetition,
                    path=target,
                    status="cached",
                    raw_replica_rows=0,
                    num_assignments=0,
                    num_written=0,
                )
            )
        else:
            missing.append(repetition)

    if not missing:
        return results

    assignments_by_repetition: dict[int, dict[str, dict[str, Any]]] = {}
    raw_rows_by_repetition: dict[int, int] = {}
    if summary:
        reconstructed = reconstruct_equalized_bucket_assignments(unique_token_file, bucket_repetitions)
        validate_reconstructed_summary(summary, reconstructed)
        assignments_by_repetition = {
            repetition: reconstructed[repetition]
            for repetition in missing
        }
        raw_rows_by_repetition = {repetition: 0 for repetition in missing}
    else:
        for repetition in missing:
            replica_file = replica_buckets_dir / f"rep_{repetition}_token.jsonl"
            assignments, raw_rows = load_bucket_assignments(replica_file, repetition)
            assignments_by_repetition[repetition] = assignments
            raw_rows_by_repetition[repetition] = raw_rows

    built = write_compact_files(unique_token_file, cache_dir, assignments_by_repetition, raw_rows_by_repetition)
    for result in built:
        per_rep_assignment_source = (
            summary_file if summary else replica_buckets_dir / f"rep_{result.repetition}_token.jsonl"
        )
        write_metadata(
            result.path,
            unique_token_file,
            per_rep_assignment_source,
            result.repetition,
            result,
            assignment_mode,
            bucket_repetitions if summary else None,
        )
    return results + built
