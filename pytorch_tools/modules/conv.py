import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


# @torch.jit.script
def _merge_convs(
    w_3x3: Tensor, b_3x3: Tensor, w_1x1: Tensor, b_1x1: Tensor, add_residual: bool = True
) -> Tuple[Tensor, Tensor]:
    # out = (X @ w_3x3 + b_3x3) @ w_1x1 + b_1x1 =
    # X @ (w_3x3 @ w_1x1) + (w_1x1 @ b_3x3 + b_1x1)

    # Flip because this is actually correlation, and permute to adapt to BHCW
    # TODO: maybe this could be replaced by matmul to simplify?
    new_w = torch.conv2d(w_3x3.permute(1, 0, 2, 3), w_1x1.flip(-1, -2)).permute(1, 0, 2, 3)
    new_b = (w_1x1.squeeze() @ b_3x3.view(-1, 1)).flatten() + b_1x1

    if add_residual:
        new_w[torch.arange(new_w.size(0)), torch.arange(new_w.size(0)), 1, 1] += 1
    return new_w, new_b


class SESRBlockExpanded(nn.Module):
    """Main block used in SESR. This class implements **expanded** forward

    Args:
        in_channels: Number of input filters.
        expansion: Factor used to increase number of channels.

    Reference:
        Collapsible Linear Blocks for Super-Efficient Super Resolution
        https://arxiv.org/abs/2103.09404
    """

    def __init__(self, in_channels: int = 16, expansion: int = 16) -> None:
        super().__init__()
        self.c3 = conv3x3(in_channels, in_channels * expansion, bias=True)
        self.c1 = conv1x1(in_channels * expansion, in_channels, bias=True)

    def forward(self, x):
        return self.c1(self.c3(x)) + x


class SESRBlockCollapsed(SESRBlockExpanded):
    """SESR block which collapses convolution during forward pass. Saves a ton of memory."""

    def forward(self, x):
        new_weight, new_bias = _merge_convs(
            self.c3.weight, self.c3.bias, self.c1.weight, self.c1.bias, add_residual=True
        )
        return F.conv2d(x, new_weight, new_bias, padding=1)
