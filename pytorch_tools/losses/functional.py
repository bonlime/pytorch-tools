import math
import torch
import torch.nn.functional as F
from .base import Reduction

# code for focal loss was borrowed from Bloodaxe
def focal_loss_with_logits(
    y_pred, y_true, gamma=2.0, alpha=0.25, reduction="mean", normalized=False, combine_thr=0,
    ):
    """see pytorch_tools.losses.focal.FocalLoss for docstring"""
    reduction = Reduction(reduction)
    y_true = y_true.type(y_pred.type())

    logpt = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
    pt = torch.exp(-logpt)

    # compute the smooth combination
    if combine_thr is None or combine_thr == 0:
        focal_term = (1 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / (1 - combine_thr)).pow(gamma)
        focal_term[pt < combine_thr] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss = loss * (alpha * y_true + (1 - alpha) * (1 - y_true))

    if normalized:
        norm_factor = focal_term.sum() + 1e-5
        loss = loss / norm_factor

    if reduction == Reduction.MEAN:
        loss = loss.mean()
    elif reduction == Reduction.SUM:
        loss = loss.sum()
    return loss

def soft_jaccard_score(y_pred, y_true, dims=None, eps=1e-4):
    """
    `Soft` means than when `y_pred` and `y_true` are zero this function will
    return 1, while in many other implementations it will return 0.
    Args:
        y_pred (torch.Tensor): Of shape `NxCx*` where * means any
            number of additional dimensions
        y_true (torch.Tensor): `NxCx*`, same shape as `y_pred`
        dims (Tuple[int], optional): Dims to use for calculating
        eps (float): Laplace smoothing
    """
    if y_pred.size() != y_true.size():
        raise ValueError("Input and target shapes should match")

    if dims is not None:
        intersection = torch.sum(y_pred * y_true, dim=dims)
        cardinality = torch.sum(y_pred + y_true, dim=dims)
    else:
        intersection = torch.sum(y_pred * y_true)
        cardinality = torch.sum(y_pred + y_true)
    union = cardinality - intersection
    jaccard_score = (intersection + eps) / (union + eps)
    return jaccard_score


def soft_dice_score(y_pred, y_true, dims=None, eps=1e-4):
    """
    `Soft` means than when `y_pred` and `y_true` are zero this function will
    return 1, while in many other implementations it will return 0.
    Args:
        y_pred (torch.Tensor): Of shape `NxCx*` where * means any
            number of additional dimensions
        y_true (torch.Tensor): `NxCx*`, same shape as `y_pred`
        dims (Tuple[int], optional): Dims to use for calculating
        eps (float): Laplace smoothing
    """
    if y_pred.size() != y_true.size():
        raise ValueError("Input and target shapes should match")

    if dims is not None:
        intersection = torch.sum(y_pred * y_true, dim=dims)
        cardinality = torch.sum(y_pred + y_true, dim=dims)
    else:
        intersection = torch.sum(y_pred * y_true)
        cardinality = torch.sum(y_pred + y_true)
    dice_score = (2.0 * intersection + eps) / (cardinality + eps)
    return dice_score


def wing_loss(y_pred, y_true, width=5, curvature=0.5, reduction="mean"):
    """
    Implementation of wing loss from https://arxiv.org/pdf/1711.06753.pdf
    Args:
        y_pred (torch.Tensor): Of shape `NxCx*` where * means any
            number of additional dimensions
        y_true (torch.Tensor): `NxCx*`, same shape as `y_pred`
        width (float): ???
        curvature (float): ???
    """
    diff_abs = (y_true - y_pred).abs()
    loss = diff_abs.clone()

    idx_smaller = diff_abs < width
    idx_bigger = diff_abs >= width

    loss[idx_smaller] = width * torch.log(1 + diff_abs[idx_smaller] / curvature)

    C = width - width * math.log(1 + width / curvature)
    loss[idx_bigger] = loss[idx_bigger] - C

    if reduction == "sum":
        loss = loss.sum()

    if reduction == "mean":
        loss = loss.mean()

    return loss


def binary_hinge(y_pred, y_true, margin=1, pos_weight=1.):
    """
    Implements Hinge loss.
    Args:
        y_pred (torch.Tensor): of shape `Nx*` where * means any number
             of additional dimensions
        y_true (torch.Tensor): same shape as y_pred
        margin (float): margin for y_pred after which loss becomes 0.
        pos_weight (float): weighting factor for positive class examples. Useful in case
            of class imbalance.
    """
    y_pred = y_pred.view(y_pred.size(0), -1)
    y_true = y_true.view(y_true.size(0), -1)
    y_true_shifted = 2 * y_true - 1  # [target == 0] = -1
    hinge = torch.nn.functional.relu(margin - y_pred * y_true_shifted)
    hinge *= y_true * pos_weight + (1 - y_true)
    return hinge.mean()  # reduction == mean
