"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from itertools import  filterfalse


def _lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

# --------------------------- BINARY LOSSES ---------------------------


def _lovasz_hinge(y_pred, y_true, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      y_pred: [B, H, W] Variable, y_pred at each pixel (between -\infty and +\infty)
      y_true: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(_lovasz_hinge_flat(*_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(y_pred, y_true))
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(y_pred, y_true, ignore))
    return loss


def _lovasz_hinge_flat(y_pred, y_true):
    """
    Binary Lovasz hinge loss
      y_pred: [P] Variable, y_pred at each prediction (between -\infty and +\infty)
      y_true: [P] Tensor, binary ground truth y_true (0 or 1)
      ignore: label to ignore
    """
    if len(y_true) == 0:
        # only void pixels, the gradients should be 0
        return y_pred.sum() * 0.
    signs = 2. * y_true.float() - 1.
    errors = (1. - y_pred * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = y_true[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def _flatten_binary_scores(y_pred, y_true, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove y_true equal to 'ignore'
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    if ignore is None:
        return y_pred, y_true
    valid = (y_true != ignore)
    vy_pred = y_pred[valid]
    vy_true = y_true[valid]
    return vy_pred, vy_true


# --------------------------- MULTICLASS LOSSES ---------------------------


def _lovasz_softmax(y_pred, y_true, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      y_pred: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      y_true: [B, H, W] Tensor, ground truth y_true (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in y_true, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class y_true
    """
    if per_image:
        loss = mean(_lovasz_softmax_flat(*_flatten_preds(pred.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for pred, lab in zip(y_pred, y_true))
    else:
        loss = _lovasz_softmax_flat(*_flatten_preds(y_pred, y_true, ignore), classes=classes)
    return loss


def _lovasz_softmax_flat(y_pred, y_true, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      y_pred: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      y_true: [P] Tensor, ground truth y_true (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in y_true, or a list of classes to average.
    """
    if y_pred.numel() == 0:
        # only void pixels, the gradients should be 0
        return y_pred * 0.
    C = y_pred.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (y_true == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = y_pred[:, 0]
        else:
            class_pred = y_pred[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(_lovasz_grad(fg_sorted))))
    return mean(losses)


def _flatten_preds(y_pred, y_true, ignore=None):
    """
    Flattens predictions in the batch
    """
    if y_pred.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = y_pred.size()
        y_pred = y_pred.view(B, 1, H, W)
    B, C, H, W = y_pred.size()
    y_pred = y_pred.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    y_true = y_true.view(-1)
    if ignore is None:
        return y_pred, y_true
    valid = (y_true != ignore)
    vy_pred = y_pred[valid.nonzero().squeeze()]
    vy_true = y_true[valid]
    return vy_pred, vy_true

# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = filterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

# --------------------------- Convinient classes ---------------------------
from .base import Loss
class BinaryLovaszLoss(Loss):
    """    
    The Binary Lovasz-Softmax loss: A tractable surrogate for the optimization of the 
    intersection-over-union measure in neural networks
    https://arxiv.org/pdf/1705.08790.pdf

      y_pred: [B, 1, H, W] Variable, y_pred at each pixel (between -\infty and +\infty)
      y_true: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image (bool): compute the loss per image instead of per batch
      ignore (int): void class id
    """
    def __init__(self, per_image=False, ignore=None):
        super().__init__()
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, y_pred, y_true):
        assert y_pred.size(1) == 1 # make sure it's binary case
        return _lovasz_hinge(y_pred.squeeze(), y_true, per_image=self.per_image, ignore=self.ignore)


class LovaszLoss(Loss):
    """
    The Lovasz-Softmax loss: A tractable surrogate for the optimization of the 
    intersection-over-union measure in neural networks
    https://arxiv.org/pdf/1705.08790.pdf
    
    y_pred: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      y_true: [B, H, W] Tensor, ground truth y_true (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in y_true, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class y_true
    """
    def __init__(self, per_image=False, ignore=None):
        super().__init__()
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, y_pred, y_true):
        return _lovasz_softmax(y_pred, y_true, per_image=self.per_image, ignore=self.ignore)
