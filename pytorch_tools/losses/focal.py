import torch
from functools import partial

from .base import Loss
from .base import Mode
from .base import Reduction
from .functional import focal_loss_with_logits


class FocalLoss(Loss):
    """Compute focal loss between target and output logits.

    Args:
        mode (str): Target mode {'binary', 'multiclass', 'multilabel'}
            'multilabel' - expects y_true of shape [N, C, *(any number of dimensions)]
                with values in {0, 1, `ignore_label`}
            'multiclass' - expects y_true of shape [N, *(any number of dimensions)]
                with values in {[0 - num_classes], `ignore_label`}
                NOTE: in current implementation it would one-hot target but then use SIGMOID. be aware
            'binary' - expects y_true of shape [N, 1, *] or [N, *]
        gamma (float): Power factor for dampening weight (focal strength).
        alpha (float): Prior probability of having positive value in target. Passing alpha=-1 would change the behaviour slightly
            in a manner better suited for segmentation. (this is an awful docstring, check code to undertand what it really means)
        reduction (str): The reduction type to apply to the output. {'none', 'mean', 'sum'}.
            'none' - no reduction will be applied
            'sum' - the output will be summed
            'mean' - the sum of the output will be divided by the number of elements in the output
        normalized (bool): Normalize focal loss by sum of focal term in the batch.
            Speeds up convergence at the end slightly. See `Reference` for paper
        combine_thr (float): Threshold for smooth combination of BCE and focal. 0 turns loss into
            pure focal, 1 turns loss into pure BCE. This is also known as `Reduced Focal Loss`.
            Reccomended value is `0.5`. See `Reference` for paper.
        ignore_label (None or int): If not None, targets may contain values to be ignored.
            Target values equal to ignore_label will be ignored from loss computation.

    Shape:
        y_pred (torch.Tensor): Tensor of arbitrary shape
        y_true (torch.Tensor): Tensor of the same shape as y_pred
    
    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
        https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/functional.py
        Normalized focal loss: https://arxiv.org/abs/1909.07829
        Reduced focal loss: https://arxiv.org/abs/1903.01347
    """

    def __init__(
        self,
        mode="binary",
        gamma=2.0,
        alpha=0.25,
        reduction="mean",
        normalized=False,
        combine_thr=0.0,
        ignore_label=-1,
    ):
        super().__init__()
        # use Enum to validate that values are valid
        Mode(mode)
        Reduction(reduction)
        self.mode = mode
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.normalized = normalized
        self.combine_thr = combine_thr
        self.ignore_label = ignore_label

    def forward(self, y_pred, y_true):
        not_ignored = y_true.ne(self.ignore_label)
        if self.mode == "multiclass":
            # to hangle ignore label we set it to 0, then scatter and set it to ignore index back
            y_true = y_true.where(not_ignored, torch.zeros(1).to(y_true))
            y_true_one_hot = torch.zeros_like(y_pred)
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1.0)
            y_true = y_true_one_hot
            # need to avoid mask shape mismatch later
            not_ignored = torch.stack([not_ignored,] * y_pred.size(1), dim=1)

        loss = focal_loss_with_logits(
            y_pred,
            y_true,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction="none",
            normalized=self.normalized,
            combine_thr=self.combine_thr,
        )

        # Filter anchors with -1 label from loss computation
        loss = loss.where(not_ignored, torch.zeros(1).to(loss))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
