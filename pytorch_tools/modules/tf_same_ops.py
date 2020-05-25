"""Implementations of Conv2d and MaxPool which match Tensorflow `same` padding"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dSamePadding(nn.Conv2d):
    """Assymetric padding matching TensorFlow `same`"""

    def forward(self, x):
        h, w = x.shape[-2:]
        pad_w = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        pad_h = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return super().forward(x)


class MaxPool2dSamePadding(nn.MaxPool2d):
    """Assymetric padding matching TensorFlow `same`"""

    def forward(self, x):
        h, w = x.shape[-2:]
        pad_w = (math.ceil(w / self.stride) - 1) * self.stride - w + self.kernel_size
        pad_h = (math.ceil(h / self.stride) - 1) * self.stride - h + self.kernel_size
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return super().forward(x)


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
