from torch.nn.modules.loss import _Loss
import torch
from enum import Enum
from typing import Union, Dict


class Mode(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


class Reduction(Enum):
    SUM = "sum"
    MEAN = "mean"
    NONE = "none"
    SAMPLE_SUM = "sample_sum"  # mean by sample dim + sum by batch dim


def _reduce(x: torch.Tensor, reduction: Union[str, Reduction] = "mean") -> torch.Tensor:
    r"""Reduce input in batch dimension if needed.
    Args:
        x: Tensor with shape (N, *).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
    """
    reduction = Reduction(reduction)
    if reduction == Reduction.NONE:
        return x
    elif reduction == Reduction.MEAN:
        return x.mean()
    elif reduction == Reduction.SUM:
        return x.sum()
    else:
        raise ValueError("Uknown reduction. Expected one of {'none', 'mean', 'sum'}")


def get_name(module: torch.nn.Module) -> str:
    if hasattr(module, "name"):
        return module.name
    elif hasattr(module, "loss"):
        return get_name(module.loss)
    else:
        return type(module).__name__


class Loss(_Loss):
    """Loss which supports addition and multiplication"""

    def __init__(self):
        super().__init__()
        self._sub_losses: Dict = dict()

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

    @property
    def sub_losses(self) -> Dict:
        # this allows to obtain some values from combined loss, to show as metrics
        return self._sub_losses or dict()


class WeightedLoss(Loss):
    """
    Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss: Loss, weight=1.0):
        super().__init__()
        self.name = get_name(loss)
        self.loss = loss
        self.register_buffer("weight", torch.tensor([weight]))

    def forward(self, *inputs):
        return self.loss(*inputs) * self.weight[0]

    @property
    def sub_losses(self) -> Dict:
        return self.loss.sub_losses


class SumOfLosses(Loss):
    def __init__(self, l1: Loss, l2: Loss):
        super().__init__()
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs):
        return self.l1(*inputs) + self.l2(*inputs)

    @property
    def sub_losses(self) -> Dict:
        return {**self.l1.sub_losses, **self.l2.sub_losses}
