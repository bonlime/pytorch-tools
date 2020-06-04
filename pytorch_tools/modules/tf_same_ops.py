"""Implementations of Conv2d and MaxPool which match Tensorflow `same` padding"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def pad_same(x, k, s, d, value=0):
    # type: (Tensor, int, int, int, float)->Tensor
    # x - input tensor, s - stride, k - kernel_size, d - dilation
    ih, iw = x.size()[-2:]
    pad_h = max((math.ceil(ih / s) - 1) * s + (k - 1) * d + 1 - ih, 0)
    pad_w = max((math.ceil(iw / s) - 1) * s + (k - 1) * d + 1 - iw, 0)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x


# current implementation is only for symmetric case. But there are no non symmetric cases
def conv2d_same(x, weight, bias=None, stride=(1, 1), dilation=(1, 1), groups=1):
    # type: (Tensor, Tensor, Optional[torch.Tensor], Tuple[int, int], Tuple[int, int], int)->Tensor
    x = pad_same(x, weight.shape[-1], stride[0], dilation[0])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


def maxpool2d_same(x, kernel_size, stride):
    # type: (Tensor, Tuple[int, int], Tuple[int, int])->Tensor
    x = pad_same(x, kernel_size[0], stride[0], 1, value=-float("inf"))
    return F.max_pool2d(x, kernel_size, stride, (0, 0))


class Conv2dSamePadding(nn.Conv2d):
    """Assymetric padding matching TensorFlow `same`"""

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.dilation, self.groups)


# as of 1.5 there is no _pair in MaxPool. Remove when this is fixed
class MaxPool2dSamePadding(nn.MaxPool2d):
    """Assymetric padding matching TensorFlow `same`"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = _pair(self.kernel_size)
        self.stride = _pair(self.stride)

    def forward(self, x):
        return maxpool2d_same(x, self.kernel_size, self.stride)


def conv_to_same_conv(module):
    """Turn All Conv2d into SameConv2d to match TF padding"""
    module_output = module
    # skip 1x1 convs
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] != 1:
        module_output = Conv2dSamePadding(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=0,  # explicitly set to 0
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
        )
        with torch.no_grad():
            module_output.weight.copy_(module.weight)
            module_output.weight.requires_grad = module.weight.requires_grad
            if module.bias is not None:
                module_output.bias.copy_(module.bias)
                module_output.bias.requires_grad = module.bias.requires_grad

    for name, child in module.named_children():
        module_output.add_module(name, conv_to_same_conv(child))
    del module
    return module_output


def maxpool_to_same_maxpool(module):
    """Turn All MaxPool2d into SameMaxPool2d to match TF padding"""
    module_output = module
    if isinstance(module, nn.MaxPool2d):
        module_output = MaxPool2dSamePadding(
            kernel_size=module.kernel_size, stride=module.stride, padding=0,  # explicitly set to 0
        )
    for name, child in module.named_children():
        module_output.add_module(name, maxpool_to_same_maxpool(child))
    del module
    return module_output
