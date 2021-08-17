import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm

from .activations import ACT
from .activations import ACT_FUNC_DICT


class ABCN(nn.Module):
    """Activated Batch Channel Normalization (BCN)
    This gathers a Batch Channel Normalization and an activation function in a single module
    BCN is basically a Batch Norm followed by Group Norm. In case of small batch size it's reccomended to use
    `estimated_stats` flag to force BN to use running statistics instead of batch

    see Ref. for paper

    Args:
        num_features (int): Number of feature channels in the input and output.
        num_groups (int): Number of groups to separate the channels into for Group Norm
        eps (float): Small constant to prevent numerical issues.
        momentum (float): Momentum factor applied to compute running statistics.
        activation (str): Name of the activation functions, one of: `relu`, `leaky_relu`, `elu` or `identity`.
        activation_param (float): Negative slope for the `leaky_relu` activation.
        estimated_stats (bool): Flag to use running stats for normalization instead of batch stats. Useful for
            micro-batch training. Slows down training by ~5%

    Reference:
        https://arxiv.org/abs/1911.09738 - Rethinking Normalization and Elimination Singularity in Neural Networks
        https://arxiv.org/abs/1903.10520 - Micro-Batch Training with Batch-Channel Normalization and Weight Standardization

    """

    def __init__(
        self,
        num_features,
        num_groups=32,
        eps=1e-5,
        momentum=0.1,
        activation="leaky_relu",
        activation_param=0.01,
        estimated_stats=False,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps
        self.momentum = momentum
        self.activation = ACT(activation)
        self.activation_param = activation_param
        self.estimated_stats = estimated_stats

        # init params
        self.register_parameter("weight", nn.Parameter(torch.ones(num_features)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_features)))
        self.register_parameter("weight_gn", nn.Parameter(torch.ones(num_features)))
        self.register_parameter("bias_gn", nn.Parameter(torch.zeros(num_features)))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        nn.init.constant_(self.weight, 1)
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.weight_gn, 1)
        nn.init.constant_(self.bias_gn, 0)

    def forward(self, x):
        # in F.batch_norm `training` regulates whether to use batch stats of buffer stats
        # if `training` is True and buffers are given, they always would be updated!
        use_batch_stats = self.training and not self.estimated_stats
        x = F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            use_batch_stats,
            self.momentum,
            self.eps,
        )
        if self.training and self.estimated_stats:
            with torch.no_grad():  # not sure if needed but just in case
                # PyTorch BN uses biased var by default
                var, mean = torch.var_mean(x, dim=(0, 2, 3), unbiased=False)
                self.running_mean = self.running_mean.mul(1 - self.momentum).add(mean, alpha=self.momentum)
                self.running_var = self.running_var.mul(1 - self.momentum).add(var, alpha=self.momentum)
        x = F.group_norm(x, self.num_groups, self.weight_gn, self.bias_gn, self.eps)
        func = ACT_FUNC_DICT[self.activation]
        if self.activation == ACT.LEAKY_RELU:
            return func(x, inplace=True, negative_slope=self.activation_param)
        elif self.activation == ACT.ELU:
            return func(x, inplace=True, alpha=self.activation_param)
        else:
            return func(x, inplace=True)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Post-Pytorch 1.0 models using standard BatchNorm have a "num_batches_tracked" parameter that we need to ignore
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, error_msgs, unexpected_keys
        )

    def extra_repr(self):
        rep = "{num_features}, num_gn_groups={num_groups}, eps={eps}, momentum={momentum}, activation={activation}"
        if self.activation in [ACT.LEAKY_RELU, ACT.ELU]:
            rep += "[{activation_param}]"
        if self.frozen:
            rep += ", frozen=True"
        if self.estimated_stats:
            rep += ", estimated_stats=True"
        return rep.format(**self.__dict__)
