from torch import nn
import torch.nn.functional as F
import torch

from .layers import ResBlock
from .layers import LearnedFeatureUnification
from .layers import setup_cross_attention_block
from .layers import RoPE
from .layers.attention import CrossAttentionBlock
from .utils.img import create_coordinate


class AnyUp(nn.Module):
    def __init__(
            self,
            input_dim=3,
            qk_dim=128,
            kernel_size=1,
            kernel_size_lfu=5,
            window_ratio=0.1,
            num_heads=4,
            init_gaussian_derivatives=False,
            use_natten=False,
            lfu_dim=None,
            **kwargs,
    ):
        super().__init__()
        self.qk_dim = qk_dim
        self.lfu_dim = lfu_dim if lfu_dim is not None else qk_dim
        self.window_ratio = window_ratio
        self._rb_args = dict(kernel_size=1, num_groups=8, pad_mode="reflect", norm_fn=nn.GroupNorm,
                             activation_fn=nn.SiLU)

        # Encoders
        self.image_encoder = self._make_encoder(input_dim, kernel_size)
        self.key_encoder = self._make_encoder(qk_dim, 1)
        self.query_encoder = self._make_encoder(qk_dim, 1)
        self.key_features_encoder = self._make_encoder(None, 1, first_layer_k=kernel_size_lfu,
                                                       init_gaussian_derivatives=init_gaussian_derivatives)

        # Cross-attention
        self.cross_decode = setup_cross_attention_block(
            use_natten=use_natten,
            qk_dim=qk_dim,
            num_heads=num_heads,
            window_ratio=window_ratio
        )
        self.aggregation = self._make_encoder(2 * qk_dim, 3)

        # RoPE for (H*W, C)
        self.rope = RoPE(qk_dim)
        self.rope._device_weight_init()

    def _make_encoder(self, in_ch, k, layers=2, first_layer_k=0, init_gaussian_derivatives=False):
        pre = (
            nn.Conv2d(in_ch, self.qk_dim, k, padding=k // 2, padding_mode="reflect", bias=False)
            if first_layer_k == 0 else
            LearnedFeatureUnification(self.lfu_dim, first_layer_k, init_gaussian_derivatives=init_gaussian_derivatives)
        )
        blocks = [ResBlock(self.qk_dim if first_layer_k == 0 or i !=0 else self.lfu_dim, self.qk_dim, **self._rb_args)
                  for i in range(layers)]
        return nn.Sequential(pre, *blocks)

    def upsample(self, enc_img, feats, out_size, vis_attn=False, q_chunk_size=None):
        b, c, h, w = feats.shape

        # Q
        q = F.adaptive_avg_pool2d(self.query_encoder(enc_img), output_size=out_size)

        # K
        k = F.adaptive_avg_pool2d(self.key_encoder(enc_img), output_size=(h, w))
        k = torch.cat([k, self.key_features_encoder(F.normalize(feats, dim=1))], dim=1)
        k = self.aggregation(k)

        # V
        v = feats

        if not isinstance(self.cross_decode, CrossAttentionBlock) and vis_attn:
            import warnings
            warnings.warn("Visualization of attention maps is not supported for NATTEN-based cross-attention.")
            vis_attn = False

        return self.cross_decode(q, k, v, vis_attn=vis_attn, q_chunk_size=q_chunk_size)

    def forward(self, image, features, output_size=None, vis_attn=False, q_chunk_size=None):
        output_size = output_size if output_size is not None else image.shape[-2:]
        enc = self.image_encoder(image)
        h = enc.shape[-2]
        coords = create_coordinate(h, enc.shape[-1], device=enc.device, dtype=enc.dtype)
        enc = enc.permute(0, 2, 3, 1).view(enc.shape[0], -1, enc.shape[1])
        enc = self.rope(enc, coords)
        enc = enc.view(enc.shape[0], h, -1, enc.shape[-1]).permute(0, 3, 1, 2)
        return self.upsample(enc, features, output_size, vis_attn=vis_attn, q_chunk_size=q_chunk_size)
