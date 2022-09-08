from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from loguru import logger


BN_TYPES = torch.nn.modules.batchnorm._BatchNorm
CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
TCONV_TYPES = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
LINEAR_TYPES = nn.Linear


def _bn_weight_bias(m: nn.Module) -> Tuple[Tensor, Tensor]:
    """convert BN module to identical conv1x1 weights and bias"""

    # BN = gamma * (x - mean) / sqrt(var + eps) + beta =
    # x * gamma / sqrt(var + eps) + (beta - gamma * mean / sqrt(var + eps))
    gamma = torch.ones_like(m.running_mean) if m.weight is None else m.weight
    beta = torch.zeros_like(m.running_mean) if m.bias is None else m.bias

    weight = gamma / (m.running_var + m.eps).sqrt()
    bias = beta - weight * m.running_mean

    return weight, bias


def _conv_bn_weights(
    *,
    conv_weight: Tensor,
    conv_bias: Tensor,
    weight_bn: Tensor,
    bias_bn: Tensor,
    is_conv: bool,
) -> Tuple[Tensor, Tensor]:

    extra_dims = [1] * (conv_weight.ndim - 2)
    weight_bn_view = weight_bn.view(-1, 1, *extra_dims) if is_conv else weight_bn.view(1, -1, *extra_dims)

    # out = (X @ conv_weight + b) * weight_bn + bias_bn =
    # X @ (conv_weight * weight_bn) + (b * weight_bn + bias_bn)
    new_conv_weight = conv_weight * weight_bn_view
    new_conv_bias = weight_bn * conv_bias + bias_bn

    return new_conv_weight, new_conv_bias


def _bn_conv_weights(
    *,
    conv_weight: Tensor,
    conv_bias: Tensor,
    weight_bn: Tensor,
    bias_bn: Tensor,
    is_conv: bool,
) -> Tuple[Tensor, Tensor]:

    assert is_conv, "Fusing BN -> ConvTransposed is not supported"

    extra_dims = [1] * (conv_weight.ndim - 2)

    # this lines is different from `conv -> BN` case!
    weight_bn_view = weight_bn.view(1, -1, *extra_dims)
    bias_bn_view = bias_bn.view(1, -1, *extra_dims)

    new_conv_weight = conv_weight * weight_bn_view
    # this is tricky but works
    new_conv_bias = (bias_bn_view * conv_weight).view(conv_weight.size(0), -1).sum(-1) + conv_bias

    return new_conv_weight, new_conv_bias


def _fuse_conv_bn(conv: nn.Module, bn: nn.Module, is_conv_bn: bool = True) -> None:
    is_conv = isinstance(conv, CONV_TYPES)

    conv_weight = conv.weight
    conv_bias = conv.bias
    if conv_bias is None:
        conv_bias = torch.zeros(
            conv_weight.size(0) if is_conv else conv_weight.size(1),
            dtype=conv_weight.dtype,
            device=conv_weight.device,
        )

    weight_bn, bias_bn = _bn_weight_bias(bn)

    weight_fn = _conv_bn_weights if is_conv_bn else _bn_conv_weights
    new_conv_weight, new_conv_bias = weight_fn(
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        weight_bn=weight_bn,
        bias_bn=bias_bn,
        is_conv=is_conv,
    )

    conv.weight.data.copy_(new_conv_weight)
    conv.bias = nn.Parameter(new_conv_bias)


def _fuse_linear_bn(linear: nn.Module, bn: nn.Module) -> None:
    linear_weight = linear.weight
    linear_bias = linear.bias

    if linear_bias is None:
        linear_bias = torch.zeros(
            linear_weight.shape[0],
            dtype=linear_weight.dtype,
            device=linear_weight.device,
        )

    weight_bn, bias_bn = _bn_weight_bias(bn)

    new_linear_weight = linear_weight * weight_bn[:, None]
    new_linear_bias = linear_bias * weight_bn + bias_bn

    linear.weight.data.copy_(new_linear_weight)
    linear.bias = torch.nn.Parameter(new_linear_bias)


def _fuse_bn_recursively(module: nn.Module) -> None:
    prev_name, prev_module = None, None
    for this_name, this_module in module.named_children():
        if len(this_module._modules) > 0:
            _fuse_bn_recursively(this_module)

        prev_is_conv = isinstance(prev_module, CONV_TYPES)
        prev_is_tconv = isinstance(prev_module, TCONV_TYPES)
        prev_is_linear = isinstance(prev_module, LINEAR_TYPES)
        prev_is_bn = isinstance(prev_module, BN_TYPES)

        this_is_conv = isinstance(this_module, CONV_TYPES)
        this_is_linear = isinstance(this_module, LINEAR_TYPES)
        this_is_bn = isinstance(this_module, BN_TYPES)

        if this_is_bn and (prev_is_conv or prev_is_tconv):
            _fuse_conv_bn(prev_module, this_module, is_conv_bn=True)
            setattr(module, this_name, nn.Identity())
        elif prev_is_bn and this_is_conv:
            _fuse_conv_bn(this_module, prev_module, is_conv_bn=False)
            setattr(module, prev_name, nn.Identity())
        elif this_is_bn and prev_is_linear:
            _fuse_linear_bn(prev_module, this_module)
            setattr(module, this_name, nn.Identity())
        elif prev_is_bn and this_is_linear:
            logger.warning("Unsupported BN+LN operation.")

        this_module = getattr(module, this_name)  # it may have changed to nn.Identity
        prev_name, prev_module = this_name, this_module


def fuse_model(model: nn.Module) -> nn.Module:
    # want to preserve grads status. using first tensor as indicator
    # requires_grad = next(model.parameters()).requires_grad
    _fuse_bn_recursively(model)
    # model = model.requires_grad_(requires_grad)
    return model
