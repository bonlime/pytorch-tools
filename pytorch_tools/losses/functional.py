import math
import torch
import torch.nn.functional as F
from .base import Reduction

# copy from @rwightman with modifications
# https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/loss.py
def focal_loss_with_logits(
    y_pred, y_true, gamma=2.0, alpha=0.25, reduction="mean", normalized=False, combine_thr=0
):
    # type: (Tensor, Tensor, float, float, str, bool, float) -> Tensor
    """see pytorch_tools.losses.focal.FocalLoss for docstring"""
    cross_entropy = F.binary_cross_entropy_with_logits(y_pred, y_true.to(y_pred), reduction="none")
    # Below are comments/derivations for computing modulator (aka focal_term).
    # For brevity, let x = logits,  z = targets, r = gamma, and p_t = sigmod(x)
    # for positive samples and 1 - sigmoid(x) for negative examples.
    #
    # The modulator, defined as (1 - P_t)^r, is a critical part in focal loss
    # computation. For r > 0, it puts more weights on hard examples, and less
    # weights on easier ones. However if it is directly computed as (1 - P_t)^r,
    # its back-propagation is not stable when r < 1. The implementation here
    # resolves the issue.
    #
    # For positive samples (labels being 1),
    #    (1 - p_t)^r
    #  = (1 - sigmoid(x))^r
    #  = (1 - (1 / (1 + exp(-x))))^r
    #  = (exp(-x) / (1 + exp(-x)))^r
    #  = exp(log((exp(-x) / (1 + exp(-x)))^r))
    #  = exp(r * log(exp(-x)) - r * log(1 + exp(-x)))
    #  = exp(- r * x - r * log(1 + exp(-x)))
    #
    # For negative samples (labels being 0),
    #    (1 - p_t)^r
    #  = (sigmoid(x))^r
    #  = (1 / (1 + exp(-x)))^r
    #  = exp(log((1 / (1 + exp(-x)))^r))
    #  = exp(-r * log(1 + exp(-x)))
    #
    # Therefore one unified form for positive (z = 1) and negative (z = 0)
    # samples is:
    #      (1 - p_t)^r = exp(-r * z * x - r * log(1 + exp(-x))).

    # compute the smooth combination
    if combine_thr > 0:
        pt = torch.exp(-cross_entropy)
        focal_term = ((1.0 - pt) / (1 - combine_thr)).pow(gamma)
        focal_term.masked_fill_(pt < combine_thr, 1)
    else:
        neg_logits = y_pred.neg()
        focal_term = torch.exp(gamma * y_true * neg_logits - gamma * torch.log1p(neg_logits.exp()))

    loss = focal_term * cross_entropy

    if alpha != -1:  # in segmentation we don't really want to use alpha
        weighted_loss = torch.where(y_true == 1.0, alpha * loss, (1.0 - alpha) * loss)
    else:
        weighted_loss = loss

    if normalized:
        # helps at the end of training when focal term is small
        norm_factor = focal_term.sum() + 1e-5
        weighted_loss = weighted_loss / norm_factor

    # TODO: maybe remove reduction from function
    if reduction == "mean":
        return weighted_loss.mean()
    elif reduction == "sum":
        return weighted_loss.sum()
    else:
        return weighted_loss


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


def binary_hinge(y_pred, y_true, margin=1, pos_weight=1.0):
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
