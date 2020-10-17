"""Implementation of Kullback-Leibler Divergence for Bernulli distribution"""

import torch
import torch.nn.functional as F
from .base import Loss
from .base import Reduction


class BinaryKLDivLoss(Loss):
    """
    Original KLDivLoss is not suited for bernulliy distribution and can't be used
    for binary classification with soft labels.
    If y - is target (it may be non-binary) and x - is predicted target

    Then KLD is defined as:
    KLD = -y log (y/x) + (1 - y) * log( (1-y) / (1-x))
    We could calculate it using BCE and adjusting for minimum value:
    BCE - BCE_min = -y*log x - (1-y) log (1-x) - (-y*ln(y)-(1-y)*ln(1-y)) = y ln (y/x) + (1-y) log ((1-y)/(1-x)) = KLD

    Args:
        reduction (str): The reduction type to apply to the output. {'none', 'mean', 'sum'}.
            'none' - no reduction will be applied
            'sum' - the output will be summed
            'mean' - the sum of the output will be divided by the number of elements in the output
        from_logits (bool): If False assumes sigmoid has already been applied to model output
    """

    def __init__(self, reduction="mean", from_logits=True):
        super().__init__()
        self.reduction = Reduction(reduction)
        self.from_logits = from_logits

    def forward(self, y_pred, y_true):
        # squeeze to allow different shapes like BSx1xHxW vs BSxHxW
        if self.from_logits:
            bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
        else:
            bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction="none")
        y_true_clamped = torch.clamp(y_true, 1e-6, 1 - 1e-6)
        bce_min = y_true_clamped * y_true_clamped.log() + (1 - y_true_clamped) * (1 - y_true_clamped).log()
        kld_loss = bce_loss + bce_min  # bce is negative so need to sum
        if self.reduction == Reduction.MEAN:
            kld_loss = kld_loss.mean()
        elif self.reduction == Reduction.SUM:
            kld_loss = kld_loss.sum()
        return kld_loss
