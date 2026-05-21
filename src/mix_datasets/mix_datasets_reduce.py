#!/usr/bin/env python3
import os
from glob import glob

import numpy as np
from tqdm import tqdm

from pretrain_dataset_builder import IndexedDatasetBuilder


def _resolve_layout(dataset_root: str, mode: str):
    mode = mode.lower()
    if mode == "fim":
        shard_dir = os.path.join(dataset_root, "combined_fim_shards")
        output_prefix = os.path.join(dataset_root, "combined_fim")
    elif mode == "fim_v2":
        shard_dir = os.path.join(dataset_root, "combined_fim_v2_shards")
        output_prefix = os.path.join(dataset_root, "combined_fim_v2")
    elif mode == "hybrid":
        shard_dir = os.path.join(dataset_root, "combined_hybrid_shards")
        output_prefix = os.path.join(dataset_root, "combined_hybrid")
    else:
        shard_dir = os.path.join(dataset_root, "combined_no_fim_shards")
        output_prefix = os.path.join(dataset_root, "combined_no_fim")

    shard_dir = os.getenv("MIX_SHARD_DIR", shard_dir)
    output_prefix = os.getenv("MIX_OUTPUT_PREFIX", output_prefix)
    return shard_dir, output_prefix


def main():
    user = os.environ["USER"]
    persistent_root = os.getenv(
        "PERSISTENT_ROOT",
        f"/capstor/store/cscs/swissai/infra01/users/{user}",
    )
    dataset_root = os.getenv("DATASET_ROOT", f"{persistent_root}/datasets")
    mix_mode = os.getenv("MIX_MODE", "no_fim")
    shard_dir, output_prefix = _resolve_layout(dataset_root, mix_mode)

    shard_bins = sorted(glob(os.path.join(shard_dir, "shard_*.bin")))
    if not shard_bins:
        raise RuntimeError(f"No .bin shards found in {shard_dir}")

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    builder = IndexedDatasetBuilder(f"{output_prefix}.bin", dtype=np.int32)

    print(f"MIX_MODE={mix_mode}")
    print(f"Merging {len(shard_bins)} shards from {shard_dir} -> {output_prefix}")

    for shard_bin in tqdm(shard_bins):
        prefix = shard_bin[:-4]
        builder.add_index(prefix)

    builder.finalize(f"{output_prefix}.idx")
    print("DONE: merged dataset at:", output_prefix)


if __name__ == "__main__":
    main()
