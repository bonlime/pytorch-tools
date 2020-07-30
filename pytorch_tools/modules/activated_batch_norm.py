import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm

from .activations import ACT
from .activations import ACT_FUNC_DICT


class ABN(nn.Module):
    """Activated Batch Normalization
    This gathers a BatchNorm and an activation function in a single module
    Parameters
    ----------
    num_features : int
        Number of feature channels in the input and output.
    eps : float
        Small constant to prevent numerical issues.
    momentum : float
        Momentum factor applied to compute running statistics.
    affine : bool
        If `True` apply learned scale and shift transformation after normalization.
    activation : str
        Name of the activation functions, one of: `relu`, `leaky_relu`, `elu` or `identity`.
    activation_param : float
        Negative slope for the `leaky_relu` activation.
    frozen (bool): if True turns `weigth` and `bias` into untrainable buffers.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        activation="leaky_relu",
        activation_param=0.01,
        frozen=False,
    ):
        super(ABN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = ACT(activation)
        self.activation_param = activation_param
        self.frozen = frozen

        if frozen:
            self.register_buffer("weight", torch.ones(num_features))
            self.register_buffer("bias", torch.zeros(num_features))
        else:
            if self.affine:
                self.weight = nn.Parameter(torch.ones(num_features))
                self.bias = nn.Parameter(torch.zeros(num_features))
            else:
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )
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

        super(ABN, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, error_msgs, unexpected_keys
        )

    def extra_repr(self):
        rep = "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, activation={activation}"
        if self.activation in [ACT.LEAKY_RELU, ACT.ELU]:
            rep += "[{activation_param}]"
        if self.frozen:
            rep += ", frozen=True"
        return rep.format(**self.__dict__)


class SyncABN(ABN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert int(os.environ.get("WORLD_SIZE", 1)) > 1, "SyncABN is only supported in multi-GPU mode"

    def forward(self, x):
        if not self.training:
            # don't reduce for validation
            return super().forward(x)
        # assume there is only one (default) process group
        process_group = torch.distributed.group.WORLD
        world_size = torch.distributed.get_world_size(process_group)
        x = sync_batch_norm.apply(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.momentum,
            process_group,
            world_size,
        )
        func = ACT_FUNC_DICT[self.activation]
        if self.activation == ACT.LEAKY_RELU:
            return func(x, inplace=True, negative_slope=self.activation_param)
        elif self.activation == ACT.ELU:
            return func(x, inplace=True, alpha=self.activation_param)
        else:
            return func(x, inplace=True)
