from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from natten import na2d
from .chunked_attention import CrossAttentionBlock


class NATTENCrossAttention(nn.Module):
    def __init__(self,
                 qk_dim,
                 num_heads,
                 use_mha_params_from: Optional[nn.MultiheadAttention] = None,
                 **kwargs):
        super().__init__()
        assert qk_dim % num_heads == 0
        self.num_heads = num_heads

        self.q_proj = nn.Conv2d(qk_dim, qk_dim, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(qk_dim, qk_dim, kernel_size=1, bias=True)

        if use_mha_params_from is not None:
            q_w, k_w, _ = use_mha_params_from.in_proj_weight.chunk(3)
            q_b, k_b, _ = use_mha_params_from.in_proj_bias.chunk(3)
            self.q_proj.weight.data.copy_(q_w.view_as(self.q_proj.weight))
            self.q_proj.bias.data.copy_(q_b)
            self.k_proj.weight.data.copy_(k_w.view_as(self.k_proj.weight))
            self.k_proj.bias.data.copy_(k_b)

    @staticmethod
    def _to_natten(x, n_heads):
        # (b, c, h, w) -> (b, h, w, n, d)
        b, c, h, w = x.shape
        d = c // n_heads
        x = x.view(b, n_heads, d, h, w).permute(0, 3, 4, 1, 2).contiguous()
        return x

    @staticmethod
    def _channels_dividable_by_8(v):
        pad_c = (8 - (v.shape[-1] % 8)) % 8
        v = F.pad(v, (0, pad_c), mode='constant', value=0) if pad_c != 0 else v
        return v, pad_c

    @staticmethod
    def _nearest_resize(x, size):
        return F.interpolate(x, size=size, mode="nearest-exact")

    def forward(self, q, k, v, kernel_size, dilation, **kwargs):
        dtype = q.dtype
        b, cq, hq_chunk, wq_chunk = q.shape

        q = self.q_proj(q)
        k = self.k_proj(k)

        # Upsample k/v to the size of the q chunk.
        k_hr = self._nearest_resize(k, size=(hq_chunk, wq_chunk)).to(dtype)
        v_hr = self._nearest_resize(v, size=(hq_chunk, wq_chunk)).to(dtype)

        q = self._to_natten(q, self.num_heads)
        k_hr = self._to_natten(k_hr, self.num_heads)

        v_hr = v_hr.permute(0, 2, 3, 1).unsqueeze(-2).expand(-1, -1, -1, self.num_heads, -1).contiguous()
        # necessary for NATTEN cutlass kernel, all others require query/key/value to be the same channel size
        v_hr, pad_c = self._channels_dividable_by_8(v_hr)

        out = na2d(
            q, k_hr, v_hr,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=1
        )

        if pad_c != 0:
            out = out[..., :-pad_c]

        out = out.permute(0, 3, 4, 1, 2).mean(dim=1)
        return out


class NATTENCrossAttentionBlock(nn.Module):
    def __init__(self, qk_dim, num_heads, window_ratio: float = 0.1,
                 q_chunk_size: Optional[int] = None,
                 use_params_from: Optional[CrossAttentionBlock] = None,
                 **kwargs):
        super().__init__()
        self.cross_attn = NATTENCrossAttention(
            qk_dim, num_heads,
            use_mha_params_from=use_params_from.cross_attn.attention if use_params_from is not None else None
        )
        self.norm_q = nn.RMSNorm(qk_dim)
        self.norm_k = nn.RMSNorm(qk_dim)

        self.q_chunk_size = q_chunk_size
        self.window_ratio = window_ratio
        self.conv2d = nn.Conv2d(qk_dim, qk_dim, kernel_size=3, stride=1, padding=1, bias=False)

        if use_params_from is not None:
            self.norm_q.weight.data.copy_(use_params_from.cross_attn.norm_q.weight.data)
            self.norm_k.weight.data.copy_(use_params_from.cross_attn.norm_k.weight.data)
            self.conv2d.weight.data.copy_(use_params_from.conv2d.weight.data)

    @staticmethod
    def _odd(n: int) -> int:
        return n if (n % 2 == 1) else (n + 1)

    def forward(self, q, k, v, q_chunk_size: Optional[int] = None, window_ratio: Optional[float] = None,
                **kwargs):
        if q_chunk_size is not None:
            import warnings
            warnings.warn(
                "In the NATTEN implementation, the q_chunk_size parameter corresponds to the number of rows in the "
                "low-res feature map processed at a time. E.g., for a 16x upsampling with q_chunk_size=8, a window_size"
                " of 0.1, and a KV height of 128 patches (corresponding Q height: 2048px), the number of loaded rows "
                "per chunk will be at most: 16 * (8 + round( 0.1 * 2 * 128)) ~ 560 rows of the high-res QKV.")

        q = self.conv2d(q)

        b, _, hq, wq = q.shape
        _, _, hk, wk = k.shape

        # --- Normalization ---
        q_flat = q.permute(0, 2, 3, 1).view(b, hq * wq, -1)
        k_flat = k.permute(0, 2, 3, 1).view(b, hk * wk, -1)
        v_flat = v.permute(0, 2, 3, 1).view(b, hk * wk, -1)

        q_norm = self.norm_q(q_flat)
        k_norm = self.norm_k(k_flat)

        q = q_norm.view(b, hq, wq, -1).permute(0, 3, 1, 2)
        k = k_norm.view(b, hk, wk, -1).permute(0, 3, 1, 2)
        v = v_flat.view(b, hk, wk, -1).permute(0, 3, 1, 2)

        # --- NATTEN Parameter Calculation ---
        chunk_size_h = q_chunk_size if q_chunk_size is not None else self.q_chunk_size
        win_ratio = self.window_ratio if window_ratio is None else window_ratio

        sh = hq // hk
        sw = wq // wk

        sh = sh.item() if isinstance(sh, torch.Tensor) else sh
        sw = sw.item() if isinstance(sw, torch.Tensor) else sw

        if 0 < win_ratio < 0.5:
            kh_l = 2 * win_ratio * hk
            kw_l = 2 * win_ratio * wk
            kh_l = kh_l.item() if isinstance(kh_l, torch.Tensor) else kh_l
            kw_l = kw_l.item() if isinstance(kw_l, torch.Tensor) else kw_l
            kh_l = self._odd(max(3, round(kh_l)))
            kw_l = self._odd(max(3, round(kw_l)))
        else:
            kh_l = hk
            kw_l = wk

        kernel_size = (kh_l, kw_l)
        dilation = (sh, sw)

        # --- Fast path: no chunking ---
        if chunk_size_h is None or chunk_size_h >= hk:
            return self.cross_attn(q, k, v,
                                   kernel_size=kernel_size,
                                   dilation=dilation)

        outputs = []

        r = kh_l // 2
        start_k = 0

        while start_k < hk:
            in_start = max(0, start_k - r)
            in_end = min(hk, start_k + chunk_size_h + r)
            valid_start = start_k
            valid_end = min(start_k + chunk_size_h, hk)

            if in_end - in_start < kh_l:
                missing = kh_l - (in_end - in_start)

                grow_right = min(missing, hk - in_end)
                in_end += grow_right
                missing -= grow_right

                if missing > 0:
                    in_start = max(0, in_start - missing)

                if hk >= kh_l and in_end - in_start < kh_l:
                    in_start = max(0, min(in_start, hk - kh_l))
                    in_end = in_start + kh_l

            next_start_k = start_k + chunk_size_h

            if in_start == 0:
                valid_end = in_end - r
                next_start_k = valid_end

            if in_end == hk:
                valid_end = hk
                next_start_k = hk

            start_k = next_start_k

            in_start_q = in_start * sh
            valid_start_q = valid_start * sh
            in_end_q = in_end * sh
            valid_end_q = valid_end * sh

            q_chunk = q[:, :, in_start_q:in_end_q]
            k_chunk = k[:, :, in_start:in_end]
            v_chunk = v[:, :, in_start:in_end]

            out_chunk = self.cross_attn(
                q_chunk, k_chunk, v_chunk,
                kernel_size=kernel_size,
                dilation=dilation
            )

            s = max(0, valid_start_q - in_start_q)
            e = min(out_chunk.shape[2], valid_end_q - in_start_q)

            valid_out = out_chunk[:, :, s:e, :]
            outputs.append(valid_out)

        features = torch.cat(outputs, dim=2)
        return features
