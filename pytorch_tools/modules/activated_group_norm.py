import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from .activations import ACT
from .activations import ACT_FUNC_DICT


class AGN(nn.Module):
    """Activated Group Normalization
    This gathers a GroupNorm and an activation function in a single module
    Parameters
    ----------
    num_features : int
        Number of feature channels in the input and output.
    num_groups: int
        Number of groups to separate the channels into
    eps : float
        Small constant to prevent numerical issues.
    affine : bool
        If `True` apply learned scale and shift transformation after normalization.
    activation : str
        Name of the activation functions, one of: `relu`, `leaky_relu`, `elu` or `identity`.
    activation_param : float
        Negative slope for the `leaky_relu` activation.
    """

    def __init__(
        self, num_features, num_groups=32, eps=1e-5, affine=True, activation="relu", activation_param=0.01,
    ):
        super(AGN, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.affine = affine
        self.eps = eps
        self.activation = ACT(activation)
        self.activation_param = activation_param
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        func = ACT_FUNC_DICT[self.activation]
        if self.activation == ACT.LEAKY_RELU:
            return func(x, inplace=True, negative_slope=self.activation_param)
        elif self.activation == ACT.ELU:
            return func(x, inplace=True, alpha=self.activation_param)
        else:
            return func(x, inplace=True)

    def extra_repr(self):
        rep = "{num_features}, eps={eps}, affine={affine}, activation={activation}"
        if self.activation in [ACT.LEAKY_RELU, ACT.ELU]:
            rep += "[{activation_param}]"
        return rep.format(**self.__dict__)
