#!/usr/bin/env python3
"""
Construct tokenized datasets for clean LTR and FIM variants (LLaMA 3.2 tokenizer).

FIM sentinels are mapped to LLaMA reserved slots that already exist in the
vocabulary, so no embedding resize is needed. The assignment is documented in
the top-level README:

    FIM_PREFIX = 128002  (<|reserved_special_token_0|>)
    FIM_MIDDLE = 128003  (<|reserved_special_token_1|>)
    FIM_SUFFIX = 128005  (<|reserved_special_token_2|>)
    PAD_TOKEN  = 128011  (<|reserved_special_token_3|>; not written by default)
    BOS_TOKEN  = 128000  (<|begin_of_text|>)
    EOS / EOD  = 128001  (<|end_of_text|>)

Modes:
  - legacy: original behavior (FIM examples + sentinel-LTR examples)
  - ltr_clean: no FIM sentinels anywhere
  - fim_standard: FIM examples, non-FIM fallback uses clean LTR
  - fim_hybrid: same formatting as fim_standard; intended to pair with swap-percent 50

Input sources (choose one):
  --input-glob          JSONL shards. Each line is either {"input_ids": [...]} (default)
                        or {"text": "..."} when --text is set.
  --bin-prefix          Megatron MMAP dataset prefix (reads path.bin + path.idx). Each
                        sequence is treated as one document; leading BOS and trailing
                        EOS are stripped before FIM rearrangement.
  --seq-length          Maximum number of source content tokens to keep before adding
                        EOS and any FIM sentinels. It is not an output-length cap.
"""

import argparse
import glob
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pretrain_dataset_builder import IndexedDataset, IndexedDatasetBuilder


FIM_PREFIX = 128002   # <|reserved_special_token_0|> repurposed as <|fim_prefix|>
FIM_MIDDLE = 128003   # <|reserved_special_token_1|> repurposed as <|fim_middle|>
FIM_SUFFIX = 128005   # <|reserved_special_token_2|> repurposed as <|fim_suffix|>
PAD_TOKEN = 128011    # <|reserved_special_token_3|>; verified against tokenizer, not written by default
BOS_TOKEN = 128000    # <|begin_of_text|>
EOS_TOKEN = 128001    # <|end_of_text|> — also serves as EOD for Megatron

EXPECTED_LLAMA_BOS_TOKEN = 128000
EXPECTED_LLAMA_EOS_EOD_TOKEN = 128001
EXPECTED_LLAMA_PAD_TOKEN = 128011

# Kept as a module global for the helper functions below; overwritten at runtime
# if a non-standard tokenizer is passed via --tokenizer (see construct_dataset).


def _extract_tokens_from_token_json(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("tokens", "ids", "input_ids", "token_ids"):
            if key in obj:
                return obj[key]
        for value in obj.values():
            if isinstance(value, list):
                return value
    raise ValueError("Could not extract token list from JSON line")


def _extract_tokens_from_text_json(obj, tokenizer):
    if isinstance(obj, dict) and "text" in obj:
        return tokenizer.encode(obj["text"], add_special_tokens=False)
    raise ValueError("Could not extract text from JSON line")


def _extract_excerpt_id(obj, field):
    if isinstance(obj, dict):
        return obj.get(field)
    return None


def _stable_rng(base_seed, excerpt_id):
    digest = hashlib.sha256(f"{base_seed}:{excerpt_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _strip_bos_eos(doc):
    while doc.shape[0] > 0 and int(doc[0]) == BOS_TOKEN:
        doc = doc[1:]
    while doc.shape[0] > 0 and int(doc[-1]) == EOS_TOKEN:
        doc = doc[:-1]
    return doc


def _split_even(doc):
    n = doc.shape[0]
    left_len = n // 3
    middle_len = n // 3
    left = doc[:left_len]
    middle = doc[left_len : left_len + middle_len]
    right = doc[left_len + middle_len :]
    return left, middle, right


def _split_random(doc, rng, min_span=1):
    n = doc.shape[0]
    if n < 3 * min_span:
        return _split_even(doc)
    left_end = rng.randint(min_span, n - (2 * min_span))
    middle_end = rng.randint(left_end + min_span, n - min_span)
    left = doc[:left_end]
    middle = doc[left_end:middle_end]
    right = doc[middle_end:]
    return left, middle, right


def _cap_content(doc, content_length):
    if content_length is None:
        return doc
    return doc[: max(content_length, 0)]


def _build_clean_ltr(doc, content_length):
    trimmed = _cap_content(doc, content_length)
    seq = np.concatenate([trimmed.astype(np.int32), np.array([EOS_TOKEN], dtype=np.int32)])
    return seq.astype(np.int32)


def _build_fim(doc, content_length, rng=None):
    doc = _cap_content(doc, content_length)
    if rng is None:
        left, middle, right = _split_even(doc)
    else:
        left, middle, right = _split_random(doc, rng)

    parts = [
        np.array([FIM_PREFIX], dtype=np.int32),
        left.astype(np.int32),
        np.array([FIM_SUFFIX], dtype=np.int32),
        right.astype(np.int32),
        np.array([FIM_MIDDLE], dtype=np.int32),
        middle.astype(np.int32),
        np.array([EOS_TOKEN], dtype=np.int32),
    ]
    seq = np.concatenate(parts)
    return seq.astype(np.int32)


def _build_legacy_ltr(doc, content_length, rng=None):
    doc = _cap_content(doc, content_length)
    if rng is None:
        left, middle, right = _split_even(doc)
    else:
        left, middle, right = _split_random(doc, rng)

    parts = [
        np.array([FIM_PREFIX], dtype=np.int32),
        left.astype(np.int32),
        np.array([FIM_MIDDLE], dtype=np.int32),
        middle.astype(np.int32),
        np.array([FIM_SUFFIX], dtype=np.int32),
        right.astype(np.int32),
        np.array([EOS_TOKEN], dtype=np.int32),
    ]
    seq = np.concatenate(parts)
    return seq.astype(np.int32)


def _extract_replica_index(obj, field):
    if isinstance(obj, dict):
        return obj.get(field)
    return None


def _iter_jsonl(files, use_text, tokenizer, excerpt_id_field, replica_index_field):
    """Yield (tokens, excerpt_id_or_None, replica_index_or_None) from JSONL shards."""
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    yield None, None, None
                    continue
                try:
                    obj = json.loads(line)
                    if use_text:
                        tokens = _extract_tokens_from_text_json(obj, tokenizer)
                    else:
                        tokens = _extract_tokens_from_token_json(obj)
                    excerpt_id = _extract_excerpt_id(obj, excerpt_id_field)
                    replica_index = _extract_replica_index(obj, replica_index_field)
                except Exception:
                    yield None, None, None
                    continue
                yield tokens, excerpt_id, replica_index


def _iter_bin(bin_prefix):
    """Yield (tokens, excerpt_id=None, replica_index=None) from a Megatron MMAP .bin/.idx."""
    ds = IndexedDataset(bin_prefix, multimodal=False, mmap=True)
    for i in range(len(ds)):
        yield ds[i], None, None


def construct_dataset(
    output_prefix,
    input_glob=None,
    bin_prefix=None,
    min_doc_length=50,
    swap_fraction=1.0,
    content_length=None,
    seed=None,
    use_text=False,
    tokenizer_name="/iopsstor/scratch/cscs/tvonarx/tokenizer/llama3_2_3B_tokenizer",
    mode="legacy",
    deterministic_splits=False,
    split_seed=42,
    excerpt_id_field="excerpt_id",
    replica_aware_splits=False,
    replica_index_field="replica_index",
    max_docs=None,
):
    if (input_glob is None) == (bin_prefix is None):
        print("Exactly one of --input-glob or --bin-prefix must be provided.", file=sys.stderr)
        sys.exit(1)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    global EOS_TOKEN, PAD_TOKEN
    tokenizer = None
    # Load the tokenizer for all paths: text JSONL needs it for encoding, while
    # pre-tokenized JSONL / bin-prefix runs use it as a fail-fast protocol check
    # that Megatron's EOD alias and our embedded EOS token are the same id.
    if tokenizer_name:
        from transformers import AutoTokenizer

        print(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        tok_bos = tokenizer.bos_token_id
        if tok_bos != EXPECTED_LLAMA_BOS_TOKEN:
            raise ValueError(
                f"Tokenizer BOS mismatch: expected {EXPECTED_LLAMA_BOS_TOKEN} "
                f"(<|begin_of_text|>), got {tok_bos}"
            )
        tok_eos = tokenizer.eos_token_id
        if tok_eos != EXPECTED_LLAMA_EOS_EOD_TOKEN:
            raise ValueError(
                f"Tokenizer EOS/EOD mismatch: expected {EXPECTED_LLAMA_EOS_EOD_TOKEN} "
                f"(<|end_of_text|>), got {tok_eos}"
            )
        if tok_eos != EOS_TOKEN:
            raise ValueError(f"Internal EOS_TOKEN={EOS_TOKEN} does not match tokenizer eos={tok_eos}")
        print(f"EOS_TOKEN={EOS_TOKEN} ({tokenizer.eos_token!r}); this is Megatron EOD")
        tok_end_of_text = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
        if tok_end_of_text != EXPECTED_LLAMA_EOS_EOD_TOKEN:
            raise ValueError(
                f"<|end_of_text|> maps to {tok_end_of_text}, expected {EXPECTED_LLAMA_EOS_EOD_TOKEN}"
            )
        tok_pad = tokenizer.pad_token_id
        if tok_pad != EXPECTED_LLAMA_PAD_TOKEN:
            raise ValueError(
                f"Tokenizer PAD mismatch: expected {EXPECTED_LLAMA_PAD_TOKEN} "
                f"(<|reserved_special_token_3|>), got {tok_pad}"
            )
        if tok_pad == tok_eos:
            raise ValueError(f"Tokenizer pad id equals EOS/EOD id ({tok_eos}); this would confuse padding and EOD")
        if tok_pad != PAD_TOKEN:
            raise ValueError(f"Internal PAD_TOKEN={PAD_TOKEN} does not match tokenizer pad={tok_pad}")
        print(f"PAD_TOKEN={PAD_TOKEN} ({tokenizer.pad_token!r}); verified distinct from EOS/EOD")
        expected_special_ids = {
            "<|reserved_special_token_0|>": FIM_PREFIX,
            "<|reserved_special_token_1|>": FIM_MIDDLE,
            "<|reserved_special_token_2|>": FIM_SUFFIX,
        }
        for token_text, expected_id in expected_special_ids.items():
            actual_id = tokenizer.convert_tokens_to_ids(token_text)
            if actual_id != expected_id:
                raise ValueError(
                    f"Tokenizer special token mismatch for {token_text}: expected {expected_id}, got {actual_id}"
                )
        if use_text:
            pass
        else:
            tokenizer = None

    # Pick input source and count total docs for the progress bar.
    if bin_prefix is not None:
        ds = IndexedDataset(bin_prefix, multimodal=False, mmap=True)
        total_docs = len(ds)
        del ds
        print(f"Reading {total_docs:,} sequences from Megatron MMAP: {bin_prefix}")
        source = _iter_bin(bin_prefix)
    else:
        files = sorted(glob.glob(input_glob))
        if not files:
            print(f"No files matched glob: {input_glob}", file=sys.stderr)
            sys.exit(1)
        total_docs = 0
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as handle:
                for _ in handle:
                    total_docs += 1
        print(f"Loading {total_docs:,} documents from {len(files)} files (glob: {input_glob})")
        source = _iter_jsonl(files, use_text, tokenizer, excerpt_id_field, replica_index_field)

    output_path = Path(output_prefix).parent
    output_path.mkdir(parents=True, exist_ok=True)

    builder = IndexedDatasetBuilder(f"{output_prefix}.bin", dtype=np.int32, multimodal=False)
    created_docs = 0
    created_fim = 0
    created_ltr = 0
    skipped_short = 0
    start_time = time.time()

    if max_docs is not None:
        total_docs = min(total_docs, max_docs)

    with tqdm(total=total_docs, desc=f"Preparing ({mode})", unit="docs", mininterval=1.0) as pbar:
        for doc_index, (tokens, excerpt_id, replica_index) in enumerate(source):
            if max_docs is not None and doc_index >= max_docs:
                break
            pbar.update(1)
            if tokens is None:
                continue

            doc = np.asarray(tokens, dtype=np.int32)
            doc = _strip_bos_eos(doc)
            n = doc.shape[0]
            if n < min_doc_length:
                skipped_short += 1
                continue

            rng = None
            if deterministic_splits:
                if excerpt_id is not None:
                    split_key = excerpt_id
                    if replica_aware_splits:
                        if replica_index is None:
                            raise ValueError(
                                "Replica-aware FIM splits require a replica index field "
                                f"({replica_index_field!r}) for excerpt_id={excerpt_id}"
                            )
                        split_key = f"{excerpt_id}:replica:{replica_index}"
                    rng = _stable_rng(split_seed, split_key)
                else:
                    if replica_aware_splits:
                        raise ValueError("Replica-aware FIM splits require JSONL input with excerpt_id metadata")
                    # bin input has no excerpt_id — derive a stable per-doc RNG from the
                    # sequence index so splits are reproducible across runs.
                    rng = _stable_rng(split_seed, f"bin:{doc_index}")

            if mode == "ltr_clean":
                output = _build_clean_ltr(doc, content_length)
                created_ltr += 1
            else:
                if rng is not None:
                    use_fim = rng.random() <= swap_fraction
                else:
                    use_fim = random.random() <= swap_fraction

                if use_fim:
                    output = _build_fim(doc, content_length, rng=rng)
                    created_fim += 1
                elif mode == "legacy":
                    output = _build_legacy_ltr(doc, content_length, rng=rng)
                    created_ltr += 1
                else:
                    output = _build_clean_ltr(doc, content_length)
                    created_ltr += 1

            eos_count = int(np.count_nonzero(output == EOS_TOKEN))
            if eos_count != 1:
                doc_label = excerpt_id if excerpt_id is not None else f"doc_index={doc_index}"
                raise ValueError(
                    f"Prepared document {doc_label} has {eos_count} EOS/EOD tokens; "
                    "expected exactly one so THD cu_seqlens boundaries remain unambiguous."
                )

            builder.add_item(torch.from_numpy(output))
            builder.end_document()
            created_docs += 1

    builder.finalize(f"{output_prefix}.idx")
    elapsed = time.time() - start_time
    print(f"Done. Created dataset in {elapsed / 60:.1f} minutes.")
    print(f"Output: {output_prefix}.bin / {output_prefix}.idx")
    print(
        f"Written docs: {created_docs:,} (FIM: {created_fim:,}, LTR: {created_ltr:,}); "
        f"skipped short (<{min_doc_length} tokens): {skipped_short:,}"
    )


def main():
    parser = argparse.ArgumentParser(description="Construct clean LTR / FIM datasets (LLaMA 3.2 tokenizer)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-glob", type=str, help="JSONL shards (one document per line)")
    src.add_argument("--bin-prefix", type=str, help="Megatron MMAP prefix (reads .bin and .idx)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--min-doc-length", type=int, default=50)
    parser.add_argument("--swap-percent", type=float, default=None)
    parser.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="Maximum source content tokens to keep before adding EOS and any FIM sentinels. "
             "Longer documents are truncated; shorter documents are not padded.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--text",
        action="store_true",
        help="Tokenize a 'text' field from each JSONL line (requires --input-glob). "
             "Ignored when --bin-prefix is set.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="/iopsstor/scratch/cscs/tvonarx/tokenizer/llama3_2_3B_tokenizer",
        help="HF tokenizer path/name. Used for text tokenization and for special-token protocol checks.",
    )
    parser.add_argument(
        "--mode",
        choices=("legacy", "ltr_clean", "fim_standard", "fim_hybrid"),
        default="fim_hybrid",
        help="Dataset output mode (default: fim_hybrid)",
    )
    parser.add_argument(
        "--deterministic-splits",
        action="store_true",
        help="Use stable per-doc RNG for split points and FIM coin. For JSONL input this "
             "uses the excerpt_id field; for --bin-prefix it uses the sequence index.",
    )
    parser.add_argument(
        "--replica-aware-splits",
        action="store_true",
        help=(
            "With --deterministic-splits on JSONL input, key the split RNG by excerpt_id "
            "and replica_index so repeated copies receive different reproducible FIM spans."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed salt used with the per-doc id when --deterministic-splits is enabled",
    )
    parser.add_argument(
        "--excerpt-id-field",
        type=str,
        default="excerpt_id",
        help="JSON field carrying stable excerpt identifier (JSONL input only)",
    )
    parser.add_argument(
        "--replica-index-field",
        type=str,
        default="replica_index",
        help="JSON field carrying per-excerpt replica index for --replica-aware-splits",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Stop after this many source documents (useful for smoke-testing).",
    )
    args = parser.parse_args()

    if args.swap_percent is None:
        if args.mode == "ltr_clean":
            swap_fraction = 0.0
        elif args.mode == "fim_hybrid":
            swap_fraction = 0.5
        else:
            swap_fraction = 1.0
    else:
        swap_fraction = min(max(float(args.swap_percent) / 100.0, 0.0), 1.0)

    construct_dataset(
        output_prefix=args.output,
        input_glob=args.input_glob,
        bin_prefix=args.bin_prefix,
        min_doc_length=args.min_doc_length,
        swap_fraction=swap_fraction,
        content_length=args.seq_length,
        seed=args.seed,
        use_text=args.text,
        tokenizer_name=args.tokenizer,
        mode=args.mode,
        deterministic_splits=args.deterministic_splits,
        split_seed=args.split_seed,
        excerpt_id_field=args.excerpt_id_field,
        replica_aware_splits=args.replica_aware_splits,
        replica_index_field=args.replica_index_field,
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()
