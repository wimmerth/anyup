import torch
from torch import nn
import torch.nn.functional as F
from ..utils.gaussian_derivative_initialization import compute_basis_size, gauss_deriv


class LearnedFeatureUnification(nn.Module):
    def __init__(self, out_channels: int, kernel_size: int = 3, init_gaussian_derivatives: bool = False):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if init_gaussian_derivatives:
            # find smallest order that gives at least out_channels basis functions
            order = 0
            while compute_basis_size(order, False) < out_channels:
                order += 1
            print(f"FeatureUnification: initializing with Gaussian derivative basis of order {order}")
            self.basis = nn.Parameter(
                gauss_deriv(
                    order, device='cpu', dtype=torch.float32, kernel_size=kernel_size, scale_magnitude=False
                )[:out_channels, None]
            )
        else:
            self.basis = nn.Parameter(
                torch.randn(out_channels, 1, kernel_size, kernel_size)
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b, c, h, w = features.shape
        x = self._depthwise_conv(features, self.basis, self.kernel_size).view(b, self.out_channels, c, h, w)
        attn = F.softmax(x, dim=1)
        return attn.mean(dim=2)

    @staticmethod
    def _depthwise_conv(feats, basis, k):
        b, c, h, w = feats.shape
        p = k // 2
        x = F.pad(feats, (p, p, p, p), value=0)
        x = F.conv2d(x, basis.repeat(c, 1, 1, 1), groups=c)
        mask = torch.ones(1, 1, h, w, dtype=x.dtype, device=x.device)
        denom = F.conv2d(F.pad(mask, (p, p, p, p), value=0), torch.ones(1, 1, k, k, dtype=x.dtype, device=x.device))
        return x / denom  # (B, out_channels*C, H, W)
