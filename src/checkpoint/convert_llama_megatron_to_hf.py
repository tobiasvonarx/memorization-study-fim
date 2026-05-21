#!/usr/bin/env python3
"""Convert the LLaMA 3.2 Megatron DCP checkpoints to Hugging Face format.

The paired LLaMA runs save Megatron distributed checkpoints with stacked layer
tensors, for example ``decoder.layers.self_attention.linear_qkv.weight`` has
shape ``[num_layers, qkv_dim, hidden_size]``. This script loads those tensors
directly with ``torch.distributed.checkpoint`` and writes a standard
``LlamaForCausalLM`` Hugging Face checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Dict

import torch
from huggingface_hub import save_torch_state_dict
from torch.distributed.checkpoint import load as dcp_load
from transformers import AutoTokenizer, LlamaConfig


FIM_TOKENS = {
    "fim_prefix": 128002,
    "fim_middle": 128003,
    "fim_suffix": 128005,
}

LLAMA_ARCHITECTURES = {
    "1B": {
        "num_layers": 16,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "pad_token_id": 128011,
    },
    "3B": {
        "num_layers": 28,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "pad_token_id": 128011,
    },
}


def apply_architecture_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = LLAMA_ARCHITECTURES[args.architecture]
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load-dir",
        required=True,
        help=(
            "Megatron checkpoint root containing latest_checkpointed_iteration.txt "
            "or a specific iter_XXXXXXX directory containing .metadata."
        ),
    )
    parser.add_argument("--save-dir", required=True, help="Output Hugging Face checkpoint directory.")
    parser.add_argument("--tokenizer-dir", required=True, help="Local LLaMA tokenizer directory.")
    parser.add_argument("--max-shard-size", default="5GB", help="HF safetensors shard size.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing save dir.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths and validate metadata only.")
    parser.add_argument(
        "--include-lm-head",
        action="store_true",
        help="Also write lm_head.weight. By default the HF config ties it to the embedding.",
    )
    parser.add_argument(
        "--architecture",
        choices=sorted(LLAMA_ARCHITECTURES),
        default="3B",
        help="LLaMA 3.2 architecture preset. Explicit shape flags override the preset.",
    )

    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--num-attention-heads", type=int, default=None)
    parser.add_argument("--num-key-value-heads", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--max-position-embeddings", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=None)
    parser.add_argument("--rms-norm-eps", type=float, default=None)
    parser.add_argument("--bos-token-id", type=int, default=None)
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument("--pad-token-id", type=int, default=None)
    return apply_architecture_defaults(parser.parse_args())


def resolve_iteration_dir(load_dir: Path) -> Path:
    load_dir = load_dir.resolve()
    if (load_dir / ".metadata").is_file():
        return load_dir

    latest_path = load_dir / "latest_checkpointed_iteration.txt"
    if latest_path.is_file():
        latest = latest_path.read_text(encoding="utf-8").strip()
        if latest == "release":
            release_dir = load_dir / "release"
            if release_dir.is_dir():
                return release_dir
            raise FileNotFoundError(f"{latest_path} points to release, but {release_dir} is missing")
        if latest.isdigit():
            iter_dir = load_dir / f"iter_{int(latest):07d}"
            if iter_dir.is_dir():
                return iter_dir
            raise FileNotFoundError(f"{latest_path} points to {iter_dir}, but it is missing")
        raise ValueError(f"Unsupported latest checkpoint marker in {latest_path}: {latest!r}")

    candidates = sorted(path for path in load_dir.glob("iter_*") if path.is_dir())
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"Could not find .metadata, latest_checkpointed_iteration.txt, or iter_* under {load_dir}"
    )


def model_tensor_shapes(args: argparse.Namespace) -> Dict[str, tuple[int, ...]]:
    head_dim = args.hidden_size // args.num_attention_heads
    qkv_dim = args.hidden_size + 2 * args.num_key_value_heads * head_dim
    return {
        "embedding.word_embeddings.weight": (args.vocab_size, args.hidden_size),
        "decoder.layers.self_attention.linear_qkv.weight": (
            args.num_layers,
            qkv_dim,
            args.hidden_size,
        ),
        "decoder.layers.self_attention.linear_qkv.layer_norm_weight": (
            args.num_layers,
            args.hidden_size,
        ),
        "decoder.layers.self_attention.linear_proj.weight": (
            args.num_layers,
            args.hidden_size,
            args.hidden_size,
        ),
        "decoder.layers.mlp.linear_fc1.weight": (
            args.num_layers,
            2 * args.intermediate_size,
            args.hidden_size,
        ),
        "decoder.layers.mlp.linear_fc1.layer_norm_weight": (
            args.num_layers,
            args.hidden_size,
        ),
        "decoder.layers.mlp.linear_fc2.weight": (
            args.num_layers,
            args.hidden_size,
            args.intermediate_size,
        ),
        "decoder.final_layernorm.weight": (args.hidden_size,),
    }


def load_tensor(checkpoint_dir: Path, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    state = {name: torch.empty(shape, dtype=torch.bfloat16)}
    dcp_load(state, checkpoint_id=str(checkpoint_dir))
    return state[name]


def own(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().contiguous()


def split_qkv(
    qkv: torch.Tensor,
    *,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    head_dim = hidden_size // num_attention_heads
    query_heads_per_group = num_attention_heads // num_key_value_heads
    grouped = qkv.view(
        num_key_value_heads,
        query_heads_per_group + 2,
        head_dim,
        hidden_size,
    )
    q = grouped[:, :query_heads_per_group].reshape(num_attention_heads * head_dim, hidden_size)
    k = grouped[:, query_heads_per_group].reshape(num_key_value_heads * head_dim, hidden_size)
    v = grouped[:, query_heads_per_group + 1].reshape(num_key_value_heads * head_dim, hidden_size)
    return q, k, v


def build_hf_config(args: argparse.Namespace) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        hidden_act="silu",
        max_position_embeddings=args.max_position_embeddings,
        initializer_range=0.02,
        rms_norm_eps=args.rms_norm_eps,
        use_cache=True,
        rope_theta=args.rope_theta,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
        pad_token_id=args.pad_token_id,
        torch_dtype="bfloat16",
    )


def write_tokenizer(tokenizer_dir: Path, save_dir: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    tokenizer.save_pretrained(save_dir)
    (save_dir / "fim_tokens.json").write_text(json.dumps(FIM_TOKENS, indent=2) + "\n")


def validate_metadata(checkpoint_dir: Path, args: argparse.Namespace) -> None:
    import pickle

    with (checkpoint_dir / ".metadata").open("rb") as handle:
        metadata = pickle.load(handle)

    expected = model_tensor_shapes(args)
    missing = [name for name in expected if name not in metadata.state_dict_metadata]
    if missing:
        raise KeyError(f"Checkpoint is missing expected model tensors: {missing}")

    mismatches = []
    for name, shape in expected.items():
        actual = tuple(metadata.state_dict_metadata[name].size)
        if actual != shape:
            mismatches.append((name, actual, shape))
    if mismatches:
        details = "\n".join(f"{name}: actual={actual} expected={shape}" for name, actual, shape in mismatches)
        raise ValueError(f"Checkpoint tensor shape mismatch:\n{details}")


def convert(args: argparse.Namespace) -> None:
    checkpoint_dir = resolve_iteration_dir(Path(args.load_dir))
    save_dir = Path(args.save_dir).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()

    validate_metadata(checkpoint_dir, args)
    if args.dry_run:
        print(f"Resolved checkpoint: {checkpoint_dir}")
        print(f"Output directory: {save_dir}")
        print("Metadata validation passed.")
        return

    if save_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Save directory already exists: {save_dir} (pass --overwrite)")
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True)

    print(f"Loading Megatron DCP checkpoint from {checkpoint_dir}", flush=True)
    print(f"Writing Hugging Face checkpoint to {save_dir}", flush=True)

    shapes = model_tensor_shapes(args)
    hf_state: dict[str, torch.Tensor] = {}

    embedding = load_tensor(checkpoint_dir, "embedding.word_embeddings.weight", shapes["embedding.word_embeddings.weight"])
    hf_state["model.embed_tokens.weight"] = embedding
    if args.include_lm_head:
        hf_state["lm_head.weight"] = own(embedding)
    del embedding
    gc.collect()

    input_norm = load_tensor(
        checkpoint_dir,
        "decoder.layers.self_attention.linear_qkv.layer_norm_weight",
        shapes["decoder.layers.self_attention.linear_qkv.layer_norm_weight"],
    )
    post_norm = load_tensor(
        checkpoint_dir,
        "decoder.layers.mlp.linear_fc1.layer_norm_weight",
        shapes["decoder.layers.mlp.linear_fc1.layer_norm_weight"],
    )
    for layer_idx in range(args.num_layers):
        hf_state[f"model.layers.{layer_idx}.input_layernorm.weight"] = own(input_norm[layer_idx])
        hf_state[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = own(post_norm[layer_idx])
    del input_norm, post_norm
    gc.collect()

    qkv_all = load_tensor(
        checkpoint_dir,
        "decoder.layers.self_attention.linear_qkv.weight",
        shapes["decoder.layers.self_attention.linear_qkv.weight"],
    )
    for layer_idx in range(args.num_layers):
        q, k, v = split_qkv(
            qkv_all[layer_idx],
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
        )
        hf_state[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = own(q)
        hf_state[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = own(k)
        hf_state[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = own(v)
    del qkv_all
    gc.collect()

    o_proj_all = load_tensor(
        checkpoint_dir,
        "decoder.layers.self_attention.linear_proj.weight",
        shapes["decoder.layers.self_attention.linear_proj.weight"],
    )
    for layer_idx in range(args.num_layers):
        hf_state[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = own(o_proj_all[layer_idx])
    del o_proj_all
    gc.collect()

    fc1_all = load_tensor(
        checkpoint_dir,
        "decoder.layers.mlp.linear_fc1.weight",
        shapes["decoder.layers.mlp.linear_fc1.weight"],
    )
    for layer_idx in range(args.num_layers):
        gate, up = torch.chunk(fc1_all[layer_idx], 2, dim=0)
        hf_state[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = own(gate)
        hf_state[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = own(up)
    del fc1_all
    gc.collect()

    fc2_all = load_tensor(
        checkpoint_dir,
        "decoder.layers.mlp.linear_fc2.weight",
        shapes["decoder.layers.mlp.linear_fc2.weight"],
    )
    for layer_idx in range(args.num_layers):
        hf_state[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = own(fc2_all[layer_idx])
    del fc2_all
    gc.collect()

    final_norm = load_tensor(
        checkpoint_dir,
        "decoder.final_layernorm.weight",
        shapes["decoder.final_layernorm.weight"],
    )
    hf_state["model.norm.weight"] = final_norm

    config = build_hf_config(args)
    config.save_pretrained(save_dir)
    write_tokenizer(tokenizer_dir, save_dir)

    save_torch_state_dict(
        hf_state,
        save_dir,
        max_shard_size=args.max_shard_size,
        safe_serialization=True,
        metadata={"format": "pt"},
    )

    metadata = {
        "source_checkpoint": str(checkpoint_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "fim_tokens": FIM_TOKENS,
        "architecture_preset": args.architecture,
        "architecture": {
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "num_attention_heads": args.num_attention_heads,
            "num_key_value_heads": args.num_key_value_heads,
            "vocab_size": args.vocab_size,
            "rope_theta": args.rope_theta,
        },
        "include_lm_head": args.include_lm_head,
    }
    (save_dir / "conversion_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved HF checkpoint to {save_dir}", flush=True)


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
