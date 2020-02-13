from .base import Loss
from . import functional as F


class WingLoss(Loss):
    def __init__(self, width=5, curvature=0.5, reduction="mean"):
        super(WingLoss, self).__init__(reduction=reduction)
        self.width = width
        self.curvature = curvature

    def forward(self, y_pred, y_true):
        return F.wing_loss(y_pred, y_true, self.width, self.curvature, self.reduction)
