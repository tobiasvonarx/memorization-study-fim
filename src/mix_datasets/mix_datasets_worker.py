#!/usr/bin/env python3
import os
from glob import glob

import numpy as np

from pretrain_dataset_builder import IndexedDatasetBuilder


def _resolve_layout(dataset_root: str, mode: str):
    mode = mode.lower()
    if mode == "fim":
        fineweb_glob = os.path.join(dataset_root, "fineweb", "fineweb_100BT_llama_fim", "*.bin")
        gutenberg_prefix = os.path.join(dataset_root, "gutenberg", "4096_tokens_with_fim", "tokens")
        shard_dir = os.path.join(dataset_root, "combined_fim_shards")
        output_prefix = os.path.join(dataset_root, "combined_fim")
    elif mode == "fim_v2":
        fineweb_glob = os.path.join(dataset_root, "fineweb", "fineweb_100BT_llama_fim", "*.bin")
        gutenberg_prefix = os.path.join(dataset_root, "gutenberg", "4096_tokens_with_fim_v2", "tokens")
        shard_dir = os.path.join(dataset_root, "combined_fim_v2_shards")
        output_prefix = os.path.join(dataset_root, "combined_fim_v2")
    elif mode == "hybrid":
        fineweb_glob = os.path.join(dataset_root, "fineweb", "fineweb_100BT_llama_fim", "*.bin")
        gutenberg_prefix = os.path.join(dataset_root, "gutenberg", "4096_tokens_hybrid", "tokens")
        shard_dir = os.path.join(dataset_root, "combined_hybrid_shards")
        output_prefix = os.path.join(dataset_root, "combined_hybrid")
    else:
        fineweb_glob = os.path.join(dataset_root, "fineweb", "fineweb_100BT_llama_no_fim", "*.bin")
        gutenberg_prefix = os.path.join(dataset_root, "gutenberg", "4096_tokens_no_fim", "tokens")
        shard_dir = os.path.join(dataset_root, "combined_no_fim_shards")
        output_prefix = os.path.join(dataset_root, "combined_no_fim")

    # Allow explicit overrides.
    fineweb_glob = os.getenv("MIX_FINEWEB_GLOB", fineweb_glob)
    gutenberg_prefix = os.getenv("MIX_GUTENBERG_PREFIX", gutenberg_prefix)
    shard_dir = os.getenv("MIX_SHARD_DIR", shard_dir)
    output_prefix = os.getenv("MIX_OUTPUT_PREFIX", output_prefix)
    return fineweb_glob, gutenberg_prefix, shard_dir, output_prefix


def get_sources(dataset_root: str, mode: str):
    fineweb_glob, gutenberg_prefix, _, _ = _resolve_layout(dataset_root, mode)
    fineweb = [path[:-4] for path in glob(fineweb_glob)]
    sources = [gutenberg_prefix] + fineweb
    return [source for source in sources if os.path.exists(f"{source}.bin") and os.path.exists(f"{source}.idx")]


def main():
    user = os.environ["USER"]
    persistent_root = os.getenv(
        "PERSISTENT_ROOT",
        f"/capstor/store/cscs/swissai/infra01/users/{user}",
    )
    dataset_root = os.getenv("DATASET_ROOT", f"{persistent_root}/datasets")
    mix_mode = os.getenv("MIX_MODE", "no_fim")
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    fineweb_glob, gutenberg_prefix, shard_dir, output_prefix = _resolve_layout(dataset_root, mix_mode)
    sources = get_sources(dataset_root, mix_mode)

    if task_id >= len(sources):
        print(f"Task {task_id} has no dataset (available={len(sources)}).")
        return

    src = sources[task_id]
    print(f"[Task {task_id}] MIX_MODE={mix_mode}")
    print(f"[Task {task_id}] Fineweb glob: {fineweb_glob}")
    print(f"[Task {task_id}] Gutenberg prefix: {gutenberg_prefix}")
    print(f"[Task {task_id}] Output shards: {shard_dir}")
    print(f"[Task {task_id}] Final output prefix: {output_prefix}")
    print(f"[Task {task_id}] Processing dataset: {src}")

    out_prefix = os.path.join(shard_dir, f"shard_{task_id}")
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    builder = IndexedDatasetBuilder(f"{out_prefix}.bin", dtype=np.int32, multimodal=False)
    builder.add_index(src)
    builder.finalize(f"{out_prefix}.idx")
    print(f"[Task {task_id}] Finished writing shard {out_prefix}")


if __name__ == "__main__":
    main()
