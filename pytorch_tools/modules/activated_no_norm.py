import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from .activations import ACT
from .activations import ACT_FUNC_DICT


class NoNormAct(nn.Module):
    """Activated No Normalization
    This is just an activation wrapped in class to allow easy swaping with BN and GN
    Args:
        num_features (int): Not used. here for compatability
        activation (str): Name of the activation functions
        activation_param (float): Negative slope for the `leaky_relu` activation.
    """

    def __init__(self, num_features, activation="relu", activation_param=0.01):
        super().__init__()
        self.num_features = num_features
        self.activation = ACT(activation)
        self.activation_param = activation_param

    def forward(self, x):
        func = ACT_FUNC_DICT[self.activation]
        if self.activation == ACT.LEAKY_RELU:
            return func(x, inplace=True, negative_slope=self.activation_param)
        elif self.activation == ACT.ELU:
            return func(x, inplace=True, alpha=self.activation_param)
        else:
            return func(x, inplace=True)

    def extra_repr(self):
        rep = "activation={activation}"
        if self.activation in [ACT.LEAKY_RELU, ACT.ELU]:
            rep += "[{activation_param}]"
        return rep.format(**self.__dict__)
