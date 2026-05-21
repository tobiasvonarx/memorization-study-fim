"""LLaMA 3.2 pretraining with THD-format packed sequences.

Thin wrapper around Megatron-LM's pretrain_gpt.py that adds PackedSeqParams
construction from EOD token positions. This is the intended Megatron extension
pattern (cf. examples/multimodal/train.py).

The key addition: build cu_seqlens from EOD positions in the token stream so
FlashAttention uses per-document attention boundaries (no cross-document
attention leakage), without needing to materialize [seq_len, seq_len] masks.

Requires:
  - micro_batch_size=1 (THD format packs multiple docs into a single sequence)
  - --no-create-attention-mask-in-dataloader
  - --reset-position-ids (GPTDataset resets position_ids at EOD boundaries)
  - --eod-mask-loss (mask loss on EOD tokens)
  - TransformerEngine >= 1.3 for THD support
"""

import os
import sys
from functools import partial

import torch

MEGATRON_DIR = os.environ.get(
    "MEGATRON_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "megatron-lm"),
)
sys.path.insert(0, MEGATRON_DIR)

from pretrain_gpt import (  # noqa: E402
    get_batch,
    is_dataset_built_on_rank,
    loss_func,
    model_provider,
    train_valid_test_datasets_provider,
)
from gpt_builders import gpt_builder  # noqa: E402

from megatron.core.enums import ModelType  # noqa: E402
from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402
from megatron.training import get_args, get_timers, get_tokenizer, pretrain  # noqa: E402


EXPECTED_LLAMA_EOS_EOD_TOKEN = 128001
EXPECTED_LLAMA_PAD_TOKEN = 128011
_TOKENIZER_BOUNDARY_IDS_VERIFIED = False


def _verify_tokenizer_boundary_ids(tokenizer):
    """Fail fast if Megatron's EOD alias is not LLaMA <|end_of_text|>."""
    global _TOKENIZER_BOUNDARY_IDS_VERIFIED
    if _TOKENIZER_BOUNDARY_IDS_VERIFIED:
        return

    eod = tokenizer.eod
    if eod != EXPECTED_LLAMA_EOS_EOD_TOKEN:
        raise RuntimeError(
            f"Megatron tokenizer.eod={eod}; expected {EXPECTED_LLAMA_EOS_EOD_TOKEN} "
            "(LLaMA <|end_of_text|>). Refusing to build THD boundaries with the wrong token."
        )

    eos_id = getattr(tokenizer, "eos_id", None)
    if eos_id is not None and eos_id != eod:
        raise RuntimeError(f"Megatron tokenizer eos_id={eos_id} but eod={eod}; expected them to match.")

    pad_id = getattr(tokenizer, "pad_id", None)
    if pad_id is not None:
        if pad_id == eod:
            raise RuntimeError(f"Megatron tokenizer pad_id equals EOD ({eod}); padding would mimic document ends.")
        if pad_id != EXPECTED_LLAMA_PAD_TOKEN:
            raise RuntimeError(
                f"Megatron tokenizer pad_id={pad_id}; expected {EXPECTED_LLAMA_PAD_TOKEN} "
                "(<|reserved_special_token_3|>)."
            )

    _TOKENIZER_BOUNDARY_IDS_VERIFIED = True


def _build_packed_seq_params(tokens, eod_token):
    """Build PackedSeqParams from EOD positions in the token stream.

    GPTDataset concatenates documents with EOD tokens between them.
    We find EOD positions and build cu_seqlens for THD-format FlashAttention.

    Args:
        tokens: [1, seq_len] tensor (micro_batch_size must be 1)
        eod_token: EOD token id

    Returns:
        PackedSeqParams with cu_seqlens and max_seqlen for FlashAttention
    """
    assert tokens.shape[0] == 1, "THD packing requires micro_batch_size=1"
    seq = tokens[0]  # [seq_len]
    seq_len = seq.shape[0]

    eod_positions = (seq == eod_token).nonzero(as_tuple=True)[0]

    if eod_positions.numel() > 0:
        # cu_seqlens boundaries: [0, after_eod_0, after_eod_1, ..., seq_len]
        # Each EOD ends a document, so the next document starts at eod_pos + 1
        boundaries = eod_positions + 1
        # If the last EOD is not at seq_len-1, the sequence was split mid-document
        if boundaries[-1] != seq_len:
            boundaries = torch.cat([boundaries, seq.new_tensor([seq_len])])
        cu_seqlens = torch.cat([seq.new_tensor([0]), boundaries]).to(torch.int32)
    else:
        # No EOD in sequence — treat entire sequence as one document
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=seq.device)

    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = int(seqlens.max().item())

    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


def forward_step(data_iterator, model, return_schedule_plan: bool = False):
    """Forward step with packed sequence parameters for THD FlashAttention."""
    args = get_args()
    timers = get_timers()

    timers("batch-generator", log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, _ = get_batch(data_iterator)
    timers("batch-generator").stop()

    tokenizer = get_tokenizer()
    _verify_tokenizer_boundary_ids(tokenizer)
    packed_seq_params = _build_packed_seq_params(tokens, tokenizer.eod)

    output_tensor = model(
        tokens,
        position_ids,
        attention_mask=None,  # Masking handled by cu_seqlens inside FlashAttention
        labels=labels,
        loss_mask=loss_mask,
        packed_seq_params=packed_seq_params,
    )
    return output_tensor, partial(loss_func, loss_mask, model=model)


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
