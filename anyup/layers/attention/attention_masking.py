import torch
from typing import Tuple
from functools import lru_cache


def window2d(
        low_res: int | Tuple[int, int],
        high_res: int | Tuple[int, int],
        ratio: float,
        *,
        device: str = "cpu"
) -> torch.Tensor:
    # unpack
    if isinstance(high_res, int):
        H = W = high_res
    else:
        H, W = high_res
    if isinstance(low_res, int):
        Lh = Lw = low_res
    else:
        Lh, Lw = low_res

    # pixel-centers in [0,1)
    r_pos = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H  # (H,)
    c_pos = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W  # (W,)
    pos_r, pos_c = torch.meshgrid(r_pos, c_pos, indexing="ij")  # (H,W)

    # clamp before scaling
    r_lo = (pos_r - ratio).clamp(0.0, 1.0)
    r_hi = (pos_r + ratio).clamp(0.0, 1.0)
    c_lo = (pos_c - ratio).clamp(0.0, 1.0)
    c_hi = (pos_c + ratio).clamp(0.0, 1.0)

    # quantise symmetrically
    r0 = (r_lo * Lh).floor().long()  # inclusive start
    r1 = (r_hi * Lh).ceil().long()  # exclusive end
    c0 = (c_lo * Lw).floor().long()
    c1 = (c_hi * Lw).ceil().long()

    return torch.stack([r0, r1, c0, c1], dim=2)


@lru_cache
def compute_attention_mask(high_res_h, high_res_w, low_res_h, low_res_w, window_size_ratio, device="cpu"):
    h, w = high_res_h, high_res_w
    h_, w_ = low_res_h, low_res_w

    windows = window2d(
        low_res=(h_, w_),
        high_res=(h, w),
        ratio=window_size_ratio,
        device=device
    )

    q = h * w  # number of high-res query locations

    # flatten window bounds: (q, 1)
    r0 = windows[..., 0].reshape(q, 1)
    r1 = windows[..., 1].reshape(q, 1)  # exclusive
    c0 = windows[..., 2].reshape(q, 1)
    c1 = windows[..., 3].reshape(q, 1)  # exclusive

    # row / column indices on low-res grid
    rows = torch.arange(h_, device=device)  # (h_,)
    cols = torch.arange(w_, device=device)  # (w_,)

    row_ok = (rows >= r0) & (rows < r1)  # (q, h_)
    col_ok = (cols >= c0) & (cols < c1)  # (q, w_)

    # broadcast to (q, h_, w_) and flatten last two dims
    attention_mask = (row_ok.unsqueeze(2) & col_ok.unsqueeze(1)) \
        .reshape(q, h_ * w_)

    return ~attention_mask
