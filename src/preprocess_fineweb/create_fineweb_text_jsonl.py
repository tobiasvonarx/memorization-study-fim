"""Convert FineWeb parquet shards to text-only JSONL for Megatron preprocessing.

Reads raw parquet files from FineWeb, extracts the 'text' field, filters short
documents, and writes {"text": "..."} JSONL suitable for Megatron's
tools/preprocess_data.py.

Usage:
    python create_fineweb_text_jsonl.py \
        --dataset /path/to/fineweb/sample/100BT \
        --output_dir /path/to/output/jsonl \
        --min-chars 200 \
        --num-workers 32
"""

import argparse
import json
import os
from multiprocessing import Pool

from datatrove.pipeline.readers import ParquetReader
from tqdm import tqdm


def process_shard(args):
    shard_path, min_chars, output_file = args
    reader = ParquetReader(shard_path)

    written = 0
    skipped = 0
    try:
        with open(output_file, "w", encoding="utf-8") as fout:
            for doc in reader():
                if not hasattr(doc, "text") or not doc.text:
                    skipped += 1
                    continue
                if len(doc.text) < min_chars:
                    skipped += 1
                    continue
                fout.write(json.dumps({"text": doc.text}, ensure_ascii=False) + "\n")
                written += 1
    except OSError as e:
        # Corrupted parquet file — log and skip rather than crashing the whole job.
        print(
            f"WARNING: skipping corrupted shard {os.path.basename(shard_path)}: {e}",
            flush=True,
        )
        # Remove any partial output so the shard is cleanly absent.
        if os.path.exists(output_file):
            os.remove(output_file)
        return os.path.basename(shard_path), 0, -1  # -1 skipped signals corruption

    return os.path.basename(shard_path), written, skipped


def main():
    p = argparse.ArgumentParser(
        description="Convert FineWeb parquet shards to text JSONL"
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Directory containing parquet shards (e.g. fineweb/sample/100BT)",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for JSONL files",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Minimum document length in characters (default: 200, ~50 tokens)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=32,
        help="Number of parallel workers for shard processing",
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    shards = sorted(
        [
            os.path.join(args.dataset, f)
            for f in os.listdir(args.dataset)
            if f.endswith(".parquet")
        ]
    )
    if not shards:
        print(f"No .parquet files found in {args.dataset}")
        return

    print(f"Found {len(shards)} parquet shards")

    tasks = []
    for shard in shards:
        output_file = os.path.join(
            args.output_dir, os.path.basename(shard).replace(".parquet", ".jsonl")
        )
        tasks.append((shard, args.min_chars, output_file))

    total_written = 0
    total_skipped = 0
    corrupted = []
    with Pool(args.num_workers) as pool:
        for shard_name, written, skipped in tqdm(
            pool.imap_unordered(process_shard, tasks),
            total=len(tasks),
            desc="Shards",
            unit="shard",
        ):
            if skipped == -1:
                corrupted.append(shard_name)
                tqdm.write(f"  {shard_name}: CORRUPTED (skipped)")
            else:
                total_written += written
                total_skipped += skipped
                tqdm.write(f"  {shard_name}: {written} written, {skipped} skipped")

    print(f"\nDone. Total: {total_written} documents written, {total_skipped} skipped")
    if corrupted:
        print(f"CORRUPTED SHARDS SKIPPED ({len(corrupted)}):")
        for s in corrupted:
            print(f"  {s}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
