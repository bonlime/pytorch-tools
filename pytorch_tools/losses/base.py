from torch.nn.modules.loss import _Loss
import torch
from enum import Enum


class Mode(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


class Reduction(Enum):
    SUM = "sum"
    MEAN = "mean"
    NONE = "none"


class Loss(_Loss):
    """Loss which supports addition and multiplication"""

    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return WeightedLoss(self, value)
        else:
            raise ValueError("Loss should be multiplied by int or float")

    def __rmul__(self, other):
        return self.__mul__(other)


class WeightedLoss(Loss):
    """
    Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.register_buffer("weight", torch.tensor([weight]))

    def forward(self, *inputs):
        return self.loss(*inputs) * self.weight[0]


class SumOfLosses(Loss):
    def __init__(self, l1, l2):
        super().__init__()
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs):
        return self.l1(*inputs) + self.l2(*inputs)
