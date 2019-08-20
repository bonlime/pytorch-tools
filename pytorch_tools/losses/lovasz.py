"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division

import torch
from torch.nn.modules.loss import _Loss

from itertools import filterfalse

from .functional import _lovasz_grad, _lovasz_hinge, _lovasz_hinge_flat,\
    _flatten_binary_scores, _lovasz_softmax, _lovasz_softmax_flat, _flatten_probas


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """Nanmean compatible with generators.
    """
    values = iter(values)
    if ignore_nan:
        values = filterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class BinaryLovaszLoss(_Loss):
    """
    The Lovasz-Softmax loss: A tractable surrogate for the optimization of the 
    intersection-over-union measure in neural networks
    https://arxiv.org/pdf/1705.08790.pdf
    """
    def __init__(self, per_image=False, ignore=None):
        super().__init__()
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, logits, target):
        return _lovasz_hinge(logits, target, per_image=self.per_image, ignore=self.ignore)


class LovaszLoss(_Loss):
    """
    The Lovasz-Softmax loss: A tractable surrogate for the optimization of the 
    intersection-over-union measure in neural networks
    https://arxiv.org/pdf/1705.08790.pdf
    """
    def __init__(self, per_image=False, ignore=None):
        super().__init__()
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, logits, target):
        return _lovasz_softmax(logits, target, per_image=self.per_image, ignore=self.ignore)
