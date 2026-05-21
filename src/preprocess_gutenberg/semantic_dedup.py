#!/usr/bin/env python3
"""Run semantic deduplication on thresholded Gutenberg excerpts."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer


TEXT_COLUMN = "text"
EXCERPT_ID_COLUMN = "excerpt_id"
INPUT_IDS_COLUMN = "input_ids"
SOURCE_BOOK_ID_COLUMN = "source_book_id"
WINDOW_INDEX_COLUMN = "window_index"
WINDOW_START_TOKEN_COLUMN = "window_start_token"
SEMANTIC_CLUSTER_ID_COLUMN = "semantic_dedup_cluster_id"
SEMANTIC_CLUSTER_SIZE_COLUMN = "semantic_dedup_cluster_size"
SEMANTIC_STATUS_COLUMN = "semantic_dedup_status"
SEMANTIC_SIMILARITY_COLUMN = "semantic_dedup_similarity_to_representative"
SEMANTIC_REPRESENTATIVE_COLUMN = "semantic_dedup_representative_excerpt_id"
SEMANTIC_JACCARD_COLUMN = "semantic_dedup_jaccard_to_representative"
SEMANTIC_JACCARD_KIND = "token_5gram_set"
SEMANTIC_JACCARD_SHINGLE_SIZE = 5
DEFAULT_SEMANTIC_DEDUP_SIMILARITY_THRESHOLD = 0.96
DEFAULT_SEMANTIC_DEDUP_JACCARD_THRESHOLD = 0.20
DEFAULT_FINAL_COUNT_DIVISOR = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic deduplication for thresholded Gutenberg excerpts.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing thresholded_* JSONL files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for final deduplicated outputs.")
    parser.add_argument(
        "--semantic-dedup-model",
        type=str,
        default="nomic-ai/nomic-embed-text-v1.5",
        help="HF model id or local path used for semantic deduplication embeddings.",
    )
    parser.add_argument(
        "--semantic-dedup-cache-dir",
        type=str,
        default=None,
        help="HF model cache dir for semantic deduplication. Defaults under $DATASET_ROOT/gutenberg/model_cache.",
    )
    parser.add_argument(
        "--semantic-dedup-batch-size",
        type=int,
        default=8,
        help="Batch size used when embedding thresholded excerpts for semantic deduplication.",
    )
    parser.add_argument(
        "--semantic-dedup-max-length",
        type=int,
        default=4096,
        help="Maximum tokenizer length used by the semantic embedding model.",
    )
    parser.add_argument(
        "--semantic-dedup-embedding-dim",
        type=int,
        default=256,
        help="Final embedding dimension retained after Matryoshka truncation.",
    )
    parser.add_argument(
        "--semantic-dedup-similarity-threshold",
        type=float,
        default=DEFAULT_SEMANTIC_DEDUP_SIMILARITY_THRESHOLD,
        help="Cosine similarity threshold required for duplicate removal.",
    )
    parser.add_argument(
        "--semantic-dedup-jaccard-threshold",
        type=float,
        default=DEFAULT_SEMANTIC_DEDUP_JACCARD_THRESHOLD,
        help="Token-shingle Jaccard threshold required, in addition to cosine similarity, for duplicate removal.",
    )
    parser.add_argument(
        "--semantic-dedup-similarity-block-size",
        type=int,
        default=512,
        help="Query block size used for blockwise cosine-similarity search during semantic deduplication.",
    )
    parser.add_argument(
        "--keep-top-k-by-ppl",
        type=int,
        default=None,
        help="Keep only the K highest-PPL excerpts after semantic deduplication.",
    )
    parser.add_argument(
        "--final-count-divisor",
        type=int,
        default=DEFAULT_FINAL_COUNT_DIVISOR,
        help="After deduplication, trim the final kept set down to a multiple of this value. Use 1 to disable.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print("[{}][semantic_dedup] {}".format(timestamp, message), flush=True)


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


def resolve_semantic_dedup_cache_dir(cli_value: Optional[str]) -> str:
    if cli_value:
        return cli_value
    user = os.environ["USER"]
    persistent_root = os.getenv(
        "PERSISTENT_ROOT",
        "/capstor/store/cscs/swissai/infra01/users/{}".format(user),
    )
    dataset_root = Path(os.getenv("DATASET_ROOT", "{}/datasets".format(persistent_root)))
    return str(dataset_root / "gutenberg" / "model_cache")


def validate_semantic_dedup_args(args) -> None:
    if args.semantic_dedup_batch_size < 1:
        raise ValueError("--semantic-dedup-batch-size must be >= 1.")
    if args.semantic_dedup_max_length < 8:
        raise ValueError("--semantic-dedup-max-length must be >= 8.")
    if args.semantic_dedup_embedding_dim < 2:
        raise ValueError("--semantic-dedup-embedding-dim must be >= 2.")
    if not 0.0 < args.semantic_dedup_similarity_threshold < 1.0:
        raise ValueError("--semantic-dedup-similarity-threshold must be in (0, 1).")
    if not 0.0 <= args.semantic_dedup_jaccard_threshold <= 1.0:
        raise ValueError("--semantic-dedup-jaccard-threshold must be in [0, 1].")
    if args.semantic_dedup_similarity_block_size < 1:
        raise ValueError("--semantic-dedup-similarity-block-size must be >= 1.")
    if args.keep_top_k_by_ppl is not None and args.keep_top_k_by_ppl < 1:
        raise ValueError("--keep-top-k-by-ppl must be >= 1 when provided.")
    if args.final_count_divisor < 1:
        raise ValueError("--final-count-divisor must be >= 1.")


def row_preference_key(row: dict) -> Tuple[float, str]:
    return (-float(row["ppl"]), row[EXCERPT_ID_COLUMN])


def choose_preferred_row(rows: Sequence[dict]) -> dict:
    return sorted(rows, key=row_preference_key)[0]


def select_top_k_by_ppl(rows: List[dict], keep_top_k_by_ppl: Optional[int]) -> List[dict]:
    if keep_top_k_by_ppl is not None:
        rows = sorted(rows, key=lambda row: (-row["ppl"], row[EXCERPT_ID_COLUMN]))[:keep_top_k_by_ppl]
        rows.sort(key=lambda row: row[EXCERPT_ID_COLUMN])
    return rows


def trim_to_count_divisor(rows: List[dict], final_count_divisor: int) -> Tuple[List[dict], List[dict]]:
    if final_count_divisor <= 1:
        return rows, []
    if len(rows) < final_count_divisor:
        raise ValueError(
            f"Cannot trim {len(rows)} rows to a positive multiple of final_count_divisor={final_count_divisor}"
        )
    remainder = len(rows) % final_count_divisor
    if remainder == 0:
        return rows, []
    keep_count = len(rows) - remainder
    kept_rows = sorted(rows, key=lambda row: (-row["ppl"], row[EXCERPT_ID_COLUMN]))[:keep_count]
    kept_ids = {row[EXCERPT_ID_COLUMN] for row in kept_rows}
    trimmed_rows = [row for row in rows if row[EXCERPT_ID_COLUMN] not in kept_ids]
    kept_rows.sort(key=lambda row: row[EXCERPT_ID_COLUMN])
    trimmed_rows.sort(key=lambda row: row[EXCERPT_ID_COLUMN])
    return kept_rows, trimmed_rows


def resolve_semantic_dedup_model_path(model_name_or_path: str, cache_dir: str) -> str:
    local_path = Path(model_name_or_path)
    if local_path.exists():
        return str(local_path.resolve())
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return snapshot_download(repo_id=model_name_or_path, cache_dir=str(cache_path))


def get_semantic_dedup_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")


def mean_pool_embeddings(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    numerator = torch.sum(hidden_state * mask, dim=1)
    denominator = torch.clamp(mask.sum(dim=1), min=1e-9)
    return numerator / denominator


def load_semantic_dedup_encoder(args):
    model_path = resolve_semantic_dedup_model_path(
        model_name_or_path=args.semantic_dedup_model,
        cache_dir=resolve_semantic_dedup_cache_dir(args.semantic_dedup_cache_dir),
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            model_max_length=args.semantic_dedup_max_length,
        )
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            model_max_length=args.semantic_dedup_max_length,
            cache_dir=resolve_semantic_dedup_cache_dir(args.semantic_dedup_cache_dir),
        )
    rope_parameters = None
    if args.semantic_dedup_max_length > 2048:
        rope_parameters = {
            "rope_theta": 1000.0,
            "rope_type": "dynamic",
            "factor": max(2.0, float(args.semantic_dedup_max_length) / 2048.0),
        }
    device = get_semantic_dedup_device()
    model_kwargs = {"trust_remote_code": True}
    if rope_parameters is not None:
        model_kwargs["rope_parameters"] = rope_parameters
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModel.from_pretrained(model_path, **model_kwargs)
    model.to(device)
    model.eval()
    log(
        "Loaded semantic dedup encoder {} on {} with max_length={} dim={}".format(
            args.semantic_dedup_model,
            device,
            args.semantic_dedup_max_length,
            args.semantic_dedup_embedding_dim,
        )
    )
    return tokenizer, model, device, model_path


@torch.inference_mode()
def encode_semantic_dedup_embeddings(rows: Sequence[dict], args) -> Tuple[torch.Tensor, str]:
    tokenizer, model, device, model_path = load_semantic_dedup_encoder(args)
    all_embeddings: List[torch.Tensor] = []
    batch_size = max(1, args.semantic_dedup_batch_size)
    prefix = "clustering: "
    total_batches = (len(rows) + batch_size - 1) // batch_size
    for start in range(0, len(rows), batch_size):
        batch_index = start // batch_size + 1
        if batch_index == 1 or batch_index == total_batches or batch_index % 25 == 0:
            log("embedding batch {:,}/{:,}".format(batch_index, total_batches))
        batch_texts = [prefix + rows[index][TEXT_COLUMN] for index in range(start, min(start + batch_size, len(rows)))]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.semantic_dedup_max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded)
        pooled = mean_pool_embeddings(outputs[0], encoded["attention_mask"])
        pooled = torch.nn.functional.layer_norm(pooled, normalized_shape=(pooled.shape[1],))
        pooled = pooled[:, : args.semantic_dedup_embedding_dim]
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embeddings.append(pooled.float().cpu())
    embeddings = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty((0, 0), dtype=torch.float32)
    if device.type == "cuda":
        del model
        torch.cuda.empty_cache()
    return embeddings, model_path


def compute_pca_coordinates(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32)
    if embeddings.shape[0] == 1:
        return torch.zeros((1, 2), dtype=torch.float32)
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    if eigenvectors.shape[1] == 1:
        coordinates = centered @ eigenvectors[:, -1:]
        return torch.cat([coordinates, torch.zeros_like(coordinates)], dim=1)
    principal_axes = eigenvectors[:, -2:]
    return centered @ principal_axes


def build_token_shingle_set(input_ids: Sequence[int], shingle_size: int = SEMANTIC_JACCARD_SHINGLE_SIZE) -> Set[Tuple[int, ...]]:
    if not input_ids:
        return set()
    if len(input_ids) < shingle_size:
        return {tuple(int(token_id) for token_id in input_ids)}
    return {
        tuple(int(token_id) for token_id in input_ids[start : start + shingle_size])
        for start in range(len(input_ids) - shingle_size + 1)
    }


def compute_jaccard_similarity(left: Set[Tuple[int, ...]], right: Set[Tuple[int, ...]]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right) / len(union))


def semantic_dedup_rows(
    rows: List[dict],
    args,
) -> Tuple[List[dict], List[dict], List[dict], List[dict], Dict[str, object], torch.Tensor, str]:
    stats: Dict[str, object] = {
        "dedup_enabled": True,
        "dedup_method": "semantic_embedding_and_token_jaccard_direct_representative_threshold",
        "semantic_dedup_model": args.semantic_dedup_model,
        "semantic_dedup_similarity_threshold": args.semantic_dedup_similarity_threshold,
        "semantic_dedup_jaccard_threshold": args.semantic_dedup_jaccard_threshold,
        "semantic_dedup_embedding_dim": args.semantic_dedup_embedding_dim,
        "semantic_dedup_max_length": args.semantic_dedup_max_length,
        "semantic_dedup_jaccard_kind": SEMANTIC_JACCARD_KIND,
        "num_rows_before_dedup": len(rows),
        "num_removed_rows_total": 0,
        "num_rows_after_dedup": len(rows),
        "num_semantic_candidate_pairs_above_threshold": 0,
        "num_semantic_candidate_pairs_above_cosine_threshold": 0,
        "num_candidate_representative_matches_passing_jaccard_threshold": 0,
        "num_rows_assigned_to_existing_representative": 0,
        "num_semantic_clusters": len(rows),
        "largest_cluster_size": 1,
    }
    if not rows:
        return rows, [], [], [], stats, torch.empty((0, 0), dtype=torch.float32), ""

    log("encoding {:,} thresholded excerpts".format(len(rows)))
    embeddings, model_path = encode_semantic_dedup_embeddings(rows, args)
    pca_coordinates = compute_pca_coordinates(embeddings)
    log("computed embeddings and PCA coordinates for {:,} excerpts".format(len(rows)))

    processing_order = sorted(range(len(rows)), key=lambda index: row_preference_key(rows[index]))
    preference_rank = {index: rank for rank, index in enumerate(processing_order)}
    candidate_matches: List[List[Tuple[int, float]]] = [[] for _ in rows]
    similarity_device = get_semantic_dedup_device()
    similarity_dtype = torch.float16 if similarity_device.type == "cuda" else torch.float32
    search_embeddings = embeddings.to(similarity_device, dtype=similarity_dtype)
    block_size = max(1, args.semantic_dedup_similarity_block_size)
    threshold = float(args.semantic_dedup_similarity_threshold)
    total_blocks = (len(rows) + block_size - 1) // block_size
    for start in range(0, len(rows), block_size):
        end = min(start + block_size, len(rows))
        block_index = start // block_size + 1
        if block_index == 1 or block_index == total_blocks or block_index % 10 == 0:
            log(
                "similarity block {:,}/{:,} | rows {:,}:{:,} | threshold={:.4f}".format(
                    block_index,
                    total_blocks,
                    start,
                    end,
                    threshold,
                )
            )
        block = search_embeddings[start:end]
        similarities = torch.matmul(block, search_embeddings.T)
        for local_row, global_row in enumerate(range(start, end)):
            similarities[local_row, : global_row + 1] = -1.0
        matched_pairs = torch.nonzero(similarities >= threshold, as_tuple=False)
        stats["num_semantic_candidate_pairs_above_threshold"] += int(matched_pairs.shape[0])
        stats["num_semantic_candidate_pairs_above_cosine_threshold"] += int(matched_pairs.shape[0])
        for local_row, match_index in matched_pairs.tolist():
            left_index = start + int(local_row)
            right_index = int(match_index)
            if preference_rank[left_index] < preference_rank[right_index]:
                better_index = left_index
                worse_index = right_index
            else:
                better_index = right_index
                worse_index = left_index
            similarity = float(similarities[local_row, match_index].item())
            candidate_matches[worse_index].append((better_index, similarity))
        if similarity_device.type == "cuda":
            torch.cuda.empty_cache()

    representative_by_index: Dict[int, int] = {}
    similarity_to_representative: Dict[int, float] = {}
    jaccard_to_representative: Dict[int, float] = {}
    grouped_rows: Dict[int, List[int]] = {}
    representative_indices = set()
    shingle_cache: Dict[int, Set[Tuple[int, ...]]] = {}

    def get_shingles(row_index: int) -> Set[Tuple[int, ...]]:
        cached = shingle_cache.get(row_index)
        if cached is None:
            cached = build_token_shingle_set(rows[row_index][INPUT_IDS_COLUMN])
            shingle_cache[row_index] = cached
        return cached

    for index in processing_order:
        best_representative = None
        best_similarity = -1.0
        best_jaccard = -1.0
        if candidate_matches[index]:
            candidate_matches[index].sort(
                key=lambda item: (
                    -item[1],
                    preference_rank[item[0]],
                    rows[item[0]][EXCERPT_ID_COLUMN],
                )
            )
            for candidate_index, similarity in candidate_matches[index]:
                if candidate_index in representative_indices:
                    jaccard = compute_jaccard_similarity(get_shingles(index), get_shingles(candidate_index))
                    if jaccard >= args.semantic_dedup_jaccard_threshold:
                        best_representative = candidate_index
                        best_similarity = similarity
                        best_jaccard = jaccard
                        stats["num_candidate_representative_matches_passing_jaccard_threshold"] += 1
                        break
        if best_representative is None:
            representative_by_index[index] = index
            similarity_to_representative[index] = 1.0
            jaccard_to_representative[index] = 1.0
            representative_indices.add(index)
            grouped_rows[index] = [index]
        else:
            representative_by_index[index] = best_representative
            similarity_to_representative[index] = best_similarity
            jaccard_to_representative[index] = best_jaccard
            grouped_rows[best_representative].append(index)
            stats["num_rows_assigned_to_existing_representative"] += 1

    deduped_rows = []
    removed_rows = []
    cluster_rows = []
    pca_rows = []
    removed_jaccard_values = []
    sorted_clusters = sorted(
        grouped_rows.values(),
        key=lambda indices: row_preference_key(choose_preferred_row([rows[index] for index in indices])),
    )
    for cluster_number, cluster_indices in enumerate(sorted_clusters):
        cluster_members = [rows[index] for index in cluster_indices]
        representative = choose_preferred_row(cluster_members)
        cluster_id = "cluster_{:05d}".format(cluster_number)
        deduped_rows.append(representative)
        member_excerpt_ids = []
        removed_excerpt_ids = []
        cluster_removed_jaccard_values = []
        for index in sorted(cluster_indices, key=lambda item: row_preference_key(rows[item])):
            row = rows[index]
            similarity = similarity_to_representative[index]
            if row[EXCERPT_ID_COLUMN] == representative[EXCERPT_ID_COLUMN]:
                jaccard = 1.0
            else:
                jaccard = jaccard_to_representative[index]
                removed_jaccard_values.append(jaccard)
                cluster_removed_jaccard_values.append(jaccard)
            member_excerpt_ids.append(row[EXCERPT_ID_COLUMN])
            annotated_row = dict(row)
            annotated_row[SEMANTIC_CLUSTER_ID_COLUMN] = cluster_id
            annotated_row[SEMANTIC_CLUSTER_SIZE_COLUMN] = len(cluster_indices)
            annotated_row[SEMANTIC_REPRESENTATIVE_COLUMN] = representative[EXCERPT_ID_COLUMN]
            annotated_row[SEMANTIC_SIMILARITY_COLUMN] = similarity
            annotated_row[SEMANTIC_JACCARD_COLUMN] = jaccard
            annotated_row["semantic_dedup_representative_ppl"] = representative["ppl"]
            annotated_row[SEMANTIC_STATUS_COLUMN] = (
                "kept" if row[EXCERPT_ID_COLUMN] == representative[EXCERPT_ID_COLUMN] else "removed"
            )
            if row[EXCERPT_ID_COLUMN] != representative[EXCERPT_ID_COLUMN]:
                removed_rows.append(annotated_row)
                removed_excerpt_ids.append(row[EXCERPT_ID_COLUMN])
            else:
                deduped_rows[-1] = annotated_row
            pca_rows.append(
                {
                    EXCERPT_ID_COLUMN: row[EXCERPT_ID_COLUMN],
                    SOURCE_BOOK_ID_COLUMN: row[SOURCE_BOOK_ID_COLUMN],
                    "ppl": row["ppl"],
                    "pca_x": float(pca_coordinates[index, 0].item()) if pca_coordinates.numel() else 0.0,
                    "pca_y": float(pca_coordinates[index, 1].item()) if pca_coordinates.numel() else 0.0,
                    SEMANTIC_CLUSTER_ID_COLUMN: cluster_id,
                    SEMANTIC_CLUSTER_SIZE_COLUMN: len(cluster_indices),
                    SEMANTIC_STATUS_COLUMN: annotated_row[SEMANTIC_STATUS_COLUMN],
                    SEMANTIC_REPRESENTATIVE_COLUMN: representative[EXCERPT_ID_COLUMN],
                    SEMANTIC_SIMILARITY_COLUMN: similarity,
                    SEMANTIC_JACCARD_COLUMN: jaccard,
                }
            )
        cluster_rows.append(
            {
                SEMANTIC_CLUSTER_ID_COLUMN: cluster_id,
                "representative_excerpt_id": representative[EXCERPT_ID_COLUMN],
                "representative_source_book_id": representative[SOURCE_BOOK_ID_COLUMN],
                "representative_ppl": representative["ppl"],
                "representative_text": representative[TEXT_COLUMN],
                "cluster_size": len(cluster_indices),
                "member_excerpt_ids": member_excerpt_ids,
                "removed_excerpt_ids": removed_excerpt_ids,
                "removed_jaccard_mean": (
                    float(sum(cluster_removed_jaccard_values) / len(cluster_removed_jaccard_values))
                    if removed_excerpt_ids
                    else None
                ),
            }
        )

    stats["num_rows_after_dedup"] = len(deduped_rows)
    stats["num_semantic_clusters"] = len(grouped_rows)
    stats["num_removed_rows_total"] = len(removed_rows)
    stats["largest_cluster_size"] = max((len(cluster_indices) for cluster_indices in grouped_rows.values()), default=0)
    if removed_jaccard_values:
        sorted_removed_jaccard = sorted(removed_jaccard_values)
        stats["removed_jaccard_mean"] = float(sum(sorted_removed_jaccard) / len(sorted_removed_jaccard))
        stats["removed_jaccard_median"] = float(sorted_removed_jaccard[len(sorted_removed_jaccard) // 2])
        stats["removed_jaccard_p95"] = float(sorted_removed_jaccard[min(len(sorted_removed_jaccard) - 1, int(0.95 * len(sorted_removed_jaccard)))])
    else:
        stats["removed_jaccard_mean"] = None
        stats["removed_jaccard_median"] = None
        stats["removed_jaccard_p95"] = None
    log(
        "semantic clustering complete | clusters={:,} kept={:,} removed={:,} candidate_pairs_above_threshold={:,} largest_cluster_size={:,}".format(
            len(grouped_rows),
            len(deduped_rows),
            len(removed_rows),
            stats["num_semantic_candidate_pairs_above_threshold"],
            stats["largest_cluster_size"],
        )
    )
    deduped_rows.sort(key=lambda row: row[EXCERPT_ID_COLUMN])
    removed_rows.sort(key=lambda row: row[EXCERPT_ID_COLUMN])
    cluster_rows.sort(key=lambda row: row["representative_excerpt_id"])
    pca_rows.sort(key=lambda row: row[EXCERPT_ID_COLUMN])
    return deduped_rows, removed_rows, cluster_rows, pca_rows, stats, embeddings, model_path


def merge_thresholded_rows(input_dir: Path) -> List[dict]:
    log("loading thresholded rows from {}".format(input_dir))
    token_rows = load_jsonl(input_dir / "thresholded_token.jsonl")
    text_rows = load_jsonl(input_dir / "thresholded_text.jsonl")
    text_by_id = {row[EXCERPT_ID_COLUMN]: row for row in text_rows}
    merged_rows = []
    for row in token_rows:
        text_row = text_by_id.get(row[EXCERPT_ID_COLUMN])
        if text_row is None:
            raise ValueError("Missing thresholded text row for excerpt_id={}".format(row[EXCERPT_ID_COLUMN]))
        merged_row = dict(row)
        merged_row[TEXT_COLUMN] = text_row[TEXT_COLUMN]
        merged_rows.append(merged_row)
    log(
        "loaded {:,} thresholded token rows and {:,} thresholded text rows".format(
            len(token_rows),
            len(text_rows),
        )
    )
    return merged_rows


def main() -> None:
    args = parse_args()
    validate_semantic_dedup_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for semantic deduplication.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(
        "starting semantic_dedup | input_dir={} output_dir={} batch_size={} dim={} cosine_threshold={:.4f} jaccard_threshold={:.4f} block_size={} keep_top_k_by_ppl={} final_count_divisor={}".format(
            input_dir,
            output_dir,
            args.semantic_dedup_batch_size,
            args.semantic_dedup_embedding_dim,
            args.semantic_dedup_similarity_threshold,
            args.semantic_dedup_jaccard_threshold,
            args.semantic_dedup_similarity_block_size,
            args.keep_top_k_by_ppl,
            args.final_count_divisor,
        )
    )

    threshold_rows = merge_thresholded_rows(input_dir)
    deduped_rows, removed_rows, cluster_rows, pca_rows, dedup_stats, embeddings, semantic_model_path = (
        semantic_dedup_rows(threshold_rows, args)
    )
    rows_after_top_k = select_top_k_by_ppl(deduped_rows, args.keep_top_k_by_ppl)
    kept_rows, divisibility_trimmed_rows = trim_to_count_divisor(rows_after_top_k, args.final_count_divisor)
    kept_ids = {row[EXCERPT_ID_COLUMN] for row in kept_rows}
    if args.keep_top_k_by_ppl is not None:
        log(
            "applied keep_top_k_by_ppl={} | rows after top-k={:,}".format(
                args.keep_top_k_by_ppl,
                len(rows_after_top_k),
            )
        )
    if divisibility_trimmed_rows:
        log(
            "trimmed {:,} low-PPL kept rows so final kept rows={:,} is divisible by {}".format(
                len(divisibility_trimmed_rows),
                len(kept_rows),
                args.final_count_divisor,
            )
        )

    token_rows = []
    text_rows = []
    semantic_removed_token_rows = []
    semantic_removed_text_rows = []
    divisibility_trimmed_token_rows = []
    divisibility_trimmed_text_rows = []
    for row in kept_rows:
        token_rows.append(
            {
                EXCERPT_ID_COLUMN: row[EXCERPT_ID_COLUMN],
                SOURCE_BOOK_ID_COLUMN: row[SOURCE_BOOK_ID_COLUMN],
                INPUT_IDS_COLUMN: row[INPUT_IDS_COLUMN],
                "ppl": row["ppl"],
                WINDOW_INDEX_COLUMN: row[WINDOW_INDEX_COLUMN],
                WINDOW_START_TOKEN_COLUMN: row[WINDOW_START_TOKEN_COLUMN],
                SEMANTIC_CLUSTER_ID_COLUMN: row[SEMANTIC_CLUSTER_ID_COLUMN],
                SEMANTIC_CLUSTER_SIZE_COLUMN: row[SEMANTIC_CLUSTER_SIZE_COLUMN],
                SEMANTIC_REPRESENTATIVE_COLUMN: row[SEMANTIC_REPRESENTATIVE_COLUMN],
                SEMANTIC_SIMILARITY_COLUMN: row[SEMANTIC_SIMILARITY_COLUMN],
                SEMANTIC_JACCARD_COLUMN: row[SEMANTIC_JACCARD_COLUMN],
            }
        )
        text_rows.append(
            {
                EXCERPT_ID_COLUMN: row[EXCERPT_ID_COLUMN],
                SOURCE_BOOK_ID_COLUMN: row[SOURCE_BOOK_ID_COLUMN],
                TEXT_COLUMN: row[TEXT_COLUMN],
                "ppl": row["ppl"],
                WINDOW_INDEX_COLUMN: row[WINDOW_INDEX_COLUMN],
                WINDOW_START_TOKEN_COLUMN: row[WINDOW_START_TOKEN_COLUMN],
                SEMANTIC_CLUSTER_ID_COLUMN: row[SEMANTIC_CLUSTER_ID_COLUMN],
                SEMANTIC_CLUSTER_SIZE_COLUMN: row[SEMANTIC_CLUSTER_SIZE_COLUMN],
                SEMANTIC_REPRESENTATIVE_COLUMN: row[SEMANTIC_REPRESENTATIVE_COLUMN],
                SEMANTIC_SIMILARITY_COLUMN: row[SEMANTIC_SIMILARITY_COLUMN],
                SEMANTIC_JACCARD_COLUMN: row[SEMANTIC_JACCARD_COLUMN],
            }
        )
    for row in divisibility_trimmed_rows:
        trim_reason = "final_count_not_divisible_by_{}".format(args.final_count_divisor)
        divisibility_trimmed_token_rows.append(
            {
                EXCERPT_ID_COLUMN: row[EXCERPT_ID_COLUMN],
                SOURCE_BOOK_ID_COLUMN: row[SOURCE_BOOK_ID_COLUMN],
                INPUT_IDS_COLUMN: row[INPUT_IDS_COLUMN],
                "ppl": row["ppl"],
                WINDOW_INDEX_COLUMN: row[WINDOW_INDEX_COLUMN],
                WINDOW_START_TOKEN_COLUMN: row[WINDOW_START_TOKEN_COLUMN],
                SEMANTIC_CLUSTER_ID_COLUMN: row[SEMANTIC_CLUSTER_ID_COLUMN],
                SEMANTIC_CLUSTER_SIZE_COLUMN: row[SEMANTIC_CLUSTER_SIZE_COLUMN],
                SEMANTIC_REPRESENTATIVE_COLUMN: row[SEMANTIC_REPRESENTATIVE_COLUMN],
                SEMANTIC_SIMILARITY_COLUMN: row[SEMANTIC_SIMILARITY_COLUMN],
                SEMANTIC_JACCARD_COLUMN: row[SEMANTIC_JACCARD_COLUMN],
                "final_count_trim_reason": trim_reason,
            }
        )
        divisibility_trimmed_text_rows.append(
            {
                EXCERPT_ID_COLUMN: row[EXCERPT_ID_COLUMN],
                SOURCE_BOOK_ID_COLUMN: row[SOURCE_BOOK_ID_COLUMN],
                TEXT_COLUMN: row[TEXT_COLUMN],
                "ppl": row["ppl"],
                WINDOW_INDEX_COLUMN: row[WINDOW_INDEX_COLUMN],
                WINDOW_START_TOKEN_COLUMN: row[WINDOW_START_TOKEN_COLUMN],
                SEMANTIC_CLUSTER_ID_COLUMN: row[SEMANTIC_CLUSTER_ID_COLUMN],
                SEMANTIC_CLUSTER_SIZE_COLUMN: row[SEMANTIC_CLUSTER_SIZE_COLUMN],
                SEMANTIC_REPRESENTATIVE_COLUMN: row[SEMANTIC_REPRESENTATIVE_COLUMN],
                SEMANTIC_SIMILARITY_COLUMN: row[SEMANTIC_SIMILARITY_COLUMN],
                SEMANTIC_JACCARD_COLUMN: row[SEMANTIC_JACCARD_COLUMN],
                "final_count_trim_reason": trim_reason,
            }
        )
    for row in removed_rows:
        semantic_removed_token_rows.append(
            {
                EXCERPT_ID_COLUMN: row[EXCERPT_ID_COLUMN],
                SOURCE_BOOK_ID_COLUMN: row[SOURCE_BOOK_ID_COLUMN],
                INPUT_IDS_COLUMN: row[INPUT_IDS_COLUMN],
                "ppl": row["ppl"],
                WINDOW_INDEX_COLUMN: row[WINDOW_INDEX_COLUMN],
                WINDOW_START_TOKEN_COLUMN: row[WINDOW_START_TOKEN_COLUMN],
                SEMANTIC_CLUSTER_ID_COLUMN: row[SEMANTIC_CLUSTER_ID_COLUMN],
                SEMANTIC_CLUSTER_SIZE_COLUMN: row[SEMANTIC_CLUSTER_SIZE_COLUMN],
                SEMANTIC_REPRESENTATIVE_COLUMN: row[SEMANTIC_REPRESENTATIVE_COLUMN],
                SEMANTIC_SIMILARITY_COLUMN: row[SEMANTIC_SIMILARITY_COLUMN],
                SEMANTIC_JACCARD_COLUMN: row[SEMANTIC_JACCARD_COLUMN],
                "semantic_dedup_representative_ppl": row["semantic_dedup_representative_ppl"],
            }
        )
        semantic_removed_text_rows.append(
            {
                EXCERPT_ID_COLUMN: row[EXCERPT_ID_COLUMN],
                SOURCE_BOOK_ID_COLUMN: row[SOURCE_BOOK_ID_COLUMN],
                TEXT_COLUMN: row[TEXT_COLUMN],
                "ppl": row["ppl"],
                WINDOW_INDEX_COLUMN: row[WINDOW_INDEX_COLUMN],
                WINDOW_START_TOKEN_COLUMN: row[WINDOW_START_TOKEN_COLUMN],
                SEMANTIC_CLUSTER_ID_COLUMN: row[SEMANTIC_CLUSTER_ID_COLUMN],
                SEMANTIC_CLUSTER_SIZE_COLUMN: row[SEMANTIC_CLUSTER_SIZE_COLUMN],
                SEMANTIC_REPRESENTATIVE_COLUMN: row[SEMANTIC_REPRESENTATIVE_COLUMN],
                SEMANTIC_SIMILARITY_COLUMN: row[SEMANTIC_SIMILARITY_COLUMN],
                SEMANTIC_JACCARD_COLUMN: row[SEMANTIC_JACCARD_COLUMN],
                "semantic_dedup_representative_ppl": row["semantic_dedup_representative_ppl"],
            }
        )

    write_jsonl(output_dir / "token.jsonl", token_rows)
    write_jsonl(output_dir / "text.jsonl", text_rows)
    write_jsonl(output_dir / "semantic_dedup_removed_token.jsonl", semantic_removed_token_rows)
    write_jsonl(output_dir / "semantic_dedup_removed_text.jsonl", semantic_removed_text_rows)
    write_jsonl(output_dir / "divisibility_trimmed_token.jsonl", divisibility_trimmed_token_rows)
    write_jsonl(output_dir / "divisibility_trimmed_text.jsonl", divisibility_trimmed_text_rows)
    write_jsonl(output_dir / "semantic_dedup_clusters.jsonl", cluster_rows)
    write_jsonl(output_dir / "semantic_dedup_pca.jsonl", pca_rows)
    log(
        "wrote semantic dedup manifests | kept_text={:,} removed_text={:,} clusters={:,} pca_points={:,}".format(
            len(text_rows),
            len(semantic_removed_text_rows),
            len(cluster_rows),
            len(pca_rows),
        )
    )
    torch.save(
        {
            "excerpt_ids": [row[EXCERPT_ID_COLUMN] for row in threshold_rows],
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "embeddings": embeddings.half(),
        },
        output_dir / "semantic_dedup_embeddings.pt",
    )
    log(
        "saved semantic embedding tensor | excerpts={:,} embedding_dim={}".format(
            len(threshold_rows),
            int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        )
    )

    summary = {}
    build_filter_summary_path = input_dir / "build_filter_summary.json"
    if build_filter_summary_path.exists():
        summary = json.loads(build_filter_summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "output_dir": str(output_dir),
            "semantic_dedup_model_path": semantic_model_path,
            "num_after_dedup": len(deduped_rows),
            "num_after_top_k_before_divisibility_trim": len(rows_after_top_k),
            "num_kept": len(kept_rows),
            "keep_top_k_by_ppl": args.keep_top_k_by_ppl,
            "final_count_divisor": args.final_count_divisor,
            "num_trimmed_for_count_divisibility": len(divisibility_trimmed_rows),
            "divisibility_trimmed_excerpt_ids": [row[EXCERPT_ID_COLUMN] for row in divisibility_trimmed_rows],
            "semantic_dedup_stats": dedup_stats,
            "kept_excerpt_ids": sorted(kept_ids),
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log("wrote summary to {}".format(output_dir / "summary.json"))

    log(
        "Wrote {:,} kept excerpts, {:,} removed excerpts, and {:,} semantic clusters to {}".format(
            len(kept_rows),
            len(removed_rows),
            len(cluster_rows),
            output_dir,
        )
    )


if __name__ == "__main__":
    main()
