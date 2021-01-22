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
            'sample_sum' - take mean by sample dimension, then sum by batch dimension
        from_logits (bool): If False assumes sigmoid has already been applied to model output
        smoothing (float): if not None, clamps prediction and target to avoid too large values of loss
            should be in [0, 1]
        ignore_label (None or int): If not None, targets may contain values to be ignored.
            Target values equal to ignore_label will be ignored from loss computation.
    """

    def __init__(self, reduction="mean", from_logits=True, smoothing=1e-6, ignore_label=-1):
        super().__init__()
        self.reduction = Reduction(reduction)
        self.from_logits = from_logits
        assert 0 < smoothing < 1, "Smoothing should be in [0, 1]"
        self.prob_clamp = smoothing
        # get value for clamping logits by inverse of sigmoid
        self.logit_clamp = torch.tensor(smoothing / (1 - smoothing)).log().item()
        self.ignore_label = ignore_label

    def forward(self, y_pred, y_true):
        # clamp to avoid nan's and to achieve smoothign effect
        y_true_clamped = torch.clamp(y_true, self.prob_clamp, 1 - self.prob_clamp)
        y_pred = torch.clamp(y_pred, self.logit_clamp, 1 - self.logit_clamp)
        if self.from_logits:
            bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true_clamped, reduction="none")
        else:
            bce_loss = F.binary_cross_entropy(y_pred, y_true_clamped, reduction="none")
        bce_min = y_true * y_true_clamped.log() + (1 - y_true) * (1 - y_true_clamped).log()
        kld_loss = bce_loss + bce_min  # bce is negative so need to sum
        # account for ignore value
        not_ignored = y_true.ne(self.ignore_label)
        kld_loss = kld_loss.where(not_ignored, torch.zeros(1).to(kld_loss))
        if self.reduction == Reduction.MEAN:
            kld_loss = kld_loss.mean()
        elif self.reduction == Reduction.SUM:
            kld_loss = kld_loss.sum()
        elif self.reduction == Reduction.SAMPLE_SUM:
            # this reduction is the same as 'sum' in default nn.SoftMarginLoss
            kld_loss = kld_loss.view(kld_loss.size(0), -1).mean(1).sum()
        return kld_loss
