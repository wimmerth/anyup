try:
    from .natten_attention import NATTENCrossAttentionBlock
except ImportError:
    NATTENCrossAttentionBlock = None

from .chunked_attention import CrossAttentionBlock
from typing import Optional
from torch import nn
import warnings


def setup_cross_attention_block(use_natten: bool,
                                qk_dim: int,
                                num_heads: int,
                                window_ratio: float = 0.1,
                                q_chunk_size: Optional[int] = None,
                                **kwargs) -> nn.Module:
    if use_natten:
        if NATTENCrossAttentionBlock is None:
            warnings.warn(
                "NATTENCrossAttentionBlock is not available."
                "Please ensure that the natten module is installed correctly."
                "Falling back to standard CrossAttentionBlock."
            )
            return CrossAttentionBlock(
                qk_dim=qk_dim,
                num_heads=num_heads,
                window_ratio=window_ratio,
                q_chunk_size=q_chunk_size,
                **kwargs
            )
        print("Using the optimized NATTEN Cross-Attention Block. Does not match the standard cross-attention exactly.")
        return NATTENCrossAttentionBlock(
            qk_dim=qk_dim,
            num_heads=num_heads,
            window_ratio=window_ratio,
            q_chunk_size=q_chunk_size,
            **kwargs
        )
    else:
        return CrossAttentionBlock(
            qk_dim=qk_dim,
            num_heads=num_heads,
            window_ratio=window_ratio,
            q_chunk_size=q_chunk_size,
            **kwargs
        )
