from torch.nn.modules.loss import _Loss

from . import functional as F


class WingLoss(_Loss):
    def __init__(self, width=5, curvature=0.5, reduction='mean'):
        super(WingLoss, self).__init__(reduction=reduction)
        self.width = width
        self.curvature = curvature

    def forward(self, prediction, target):
        return F.wing_loss(prediction, target, self.width, self.curvature, self.reduction)
