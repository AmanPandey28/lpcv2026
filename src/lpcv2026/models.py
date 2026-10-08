"""Scoped graph rewrites for the fixed-batch MobileCLIP2 inference export.

These are inference/export adapters, not a replacement training implementation.
The PyTorch reference in evaluation.py deliberately does not use these patches.
"""

from contextlib import contextmanager
import math

import torch
import torch.nn.functional as F


def explicit_sdpa(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """Lower inference SDPA to MatMul -> scale/mask -> Softmax -> MatMul."""
    if dropout_p != 0.0 or enable_gqa:
        raise ValueError(
            "This export adapter supports zero-dropout, non-GQA inference only"
        )
    if is_causal and attn_mask is not None:
        raise ValueError("Specify either causal attention or an explicit mask")
    factor = scale if scale is not None else 1.0 / math.sqrt(query.size(-1))
    scores = (query @ key.transpose(-2, -1)) * factor
    if is_causal:
        # Matches PyTorch's upper-left aligned causal mask, including L != S.
        allowed = torch.ones(
            query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
        ).tril()
        scores = scores.masked_fill(~allowed, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    probabilities = torch.softmax(scores, dim=-1)
    # Native SDPA returns zero for fully masked rows, rather than propagating NaN.
    if attn_mask is not None:
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
    return probabilities @ value


def batch_one_text_pool(x, text, pool_type, eos_token_id=None):
    """Avoid the batch-index advanced indexing path for argmax/EOS pooling."""
    if pool_type != "argmax":
        raise ValueError("MobileCLIP2-S2 export expects argmax text pooling")
    if x.shape[0] != 1 or text.shape[0] != 1:
        raise ValueError("This pooling rewrite is valid only for static batch one")
    return x[0:1, text.argmax(dim=-1)[0], :]


@contextmanager
def export_rewrites():
    """Temporarily disable fused MHA and patch SDPA/pooling; always restore."""
    import open_clip.transformer as transformer

    original_sdpa = F.scaled_dot_product_attention
    original_pool = transformer.text_global_pool
    original_fastpath = torch.backends.mha.get_fastpath_enabled()

    def pool(x, text, pool_type, eos_token_id=None):
        if pool_type == "argmax":
            return batch_one_text_pool(x, text, pool_type, eos_token_id)
        return original_pool(x, text, pool_type, eos_token_id)

    try:
        torch.backends.mha.set_fastpath_enabled(False)
        F.scaled_dot_product_attention = explicit_sdpa
        transformer.text_global_pool = pool
        yield
    finally:
        F.scaled_dot_product_attention = original_sdpa
        transformer.text_global_pool = original_pool
        torch.backends.mha.set_fastpath_enabled(original_fastpath)
