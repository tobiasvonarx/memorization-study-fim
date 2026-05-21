#!/usr/bin/env python3
import argparse
from pathlib import Path

from transformers import AutoTokenizer


def _token_to_str(token):
    return str(token) if token is not None else None


def _select_pad_from_additional_special_tokens(tokenizer):
    # Prefer an existing configured pad token, otherwise derive from additional specials.
    if tokenizer.pad_token is not None:
        return _token_to_str(tokenizer.pad_token), "already-configured"

    candidates = tokenizer.special_tokens_map.get("additional_special_tokens", [])
    candidate_strings = [_token_to_str(token) for token in candidates if _token_to_str(token)]
    if not candidate_strings:
        # Fallback: use a dedicated reserved token as pad (<|reserved_special_token_3|>,
        # id 128011 for LLaMA 3.2 — note this is not 128005; 128005 is reserved_special_token_2,
        # and 128004 is <|finetune_right_pad_id|>). Keeps pad semantics separate from EOS.
        # Not used for training data padding (THD packing avoids padding entirely), but
        # useful for HF inference tooling.
        vocab = tokenizer.get_vocab()
        if "<|reserved_special_token_3|>" in vocab:
            return "<|reserved_special_token_3|>", "fallback-reserved"
        return None, "none-available"

    # Use the first token containing 'pad' (case-insensitive), fallback to the first candidate.
    for token in candidate_strings:
        if "pad" in token.lower():
            return token, "matched-substring"
    return candidate_strings[0], "fallback-first"


def main():
    parser = argparse.ArgumentParser(description="Download and save tokenizer only")
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Hugging Face tokenizer/model id",
    )
    parser.add_argument("--output", type=str, required=True, help="Local output directory")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer {args.tokenizer_id} to {output}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)

    pad_token, reason = _select_pad_from_additional_special_tokens(tokenizer)
    tokenizer.pad_token = pad_token
    print(f"Using pad_token={tokenizer.pad_token!r} (reason={reason})")
    print(f"pad_token_id={tokenizer.pad_token_id}, eos_token={tokenizer.eos_token!r}")

    tokenizer.save_pretrained(output)
    print(f"Tokenizer saved to: {output}")


if __name__ == "__main__":
    main()
