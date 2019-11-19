from functools import partial
from .base import Loss
from .functional import sigmoid_focal_loss, reduced_focal_loss


class BinaryFocalLoss(Loss):
    def __init__(self, alpha=0.5, gamma=2, ignore_index=None, reduction='mean', reduced=False, threshold=0.5):
        """

        :param alpha:
        :param gamma:
        :param ignore_index:
        :param reduced:
        :param threshold:
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        if reduced:
            self.focal_loss = partial(reduced_focal_loss, gamma=gamma, threshold=threshold, reduction=reduction)
        else:
            self.focal_loss = partial(sigmoid_focal_loss, gamma=gamma, alpha=alpha, reduction=reduction)

    def forward(self, y_pred, y_true):
        """Compute focal loss for binary classification problem.
        """
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        if self.ignore_index is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = y_true != self.ignore_index
            y_pred = y_pred[not_ignored]
            y_true = y_true[not_ignored]

        loss = self.focal_loss(y_pred, y_true)
        return loss


class FocalLoss(Loss):
    def __init__(self, alpha=0.5, gamma=2, ignore_index=None):
        """
        Focal loss for multi-class problem.

        :param alpha:
        :param gamma:
        :param ignore_index: If not None, targets with given index are ignored
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true):
        num_classes = y_pred.size(1)
        loss = 0

        # Filter anchors with -1 label from loss computation
        if self.ignore_index is not None:
            not_ignored = y_true != self.ignore_index

        for cls in range(num_classes):
            cls_y_true = (y_true == cls).long()
            cls_y_pred = y_pred[:, cls, ...]

            if self.ignore_index is not None:
                cls_y_true = cls_y_true[not_ignored]
                cls_y_pred = cls_y_pred[not_ignored]

            loss += sigmoid_focal_loss(cls_y_pred, cls_y_true, gamma=self.gamma, alpha=self.alpha)
        return loss
