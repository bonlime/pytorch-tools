from .base import Loss
from . import functional as F


class BinaryHinge(Loss):
    """
    Implementation of Hinge loss for binary classification tasks.
    Could be used in classification with one-hot encoded target or in
    semantic segmentation
    """
    def __init__(self, margin=1, pos_weight=1):
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
    
    def forward(self, y_pred, y_true):
        return F.binary_hinge(y_pred, y_true, self.margin, self.pos_weight)

