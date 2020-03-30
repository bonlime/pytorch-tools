import torch
from .base import Loss, Mode
from .functional import soft_dice_score, soft_jaccard_score


class DiceLoss(Loss):
    """
    Implementation of Dice loss for image segmentation task.
    It supports binary, multiclass and multilabel cases

    Args:
        mode (str): Target mode {'binary', 'multiclass', 'multilabel'}
            'multilabel' - expects y_true of shape [N, C, H, W]
            'multiclass', 'binary' - expects y_true of shape [N, H, W]
        log_loss (bool): If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        from_logits (bool): If True assumes input is raw logits
        eps (float): small epsilon for numerical stability
    Shape:
        y_pred: [N, C, H, W]
        y_true: [N, C, H, W] or [N, H, W] depending on mode
    """

    IOU_FUNCTION = soft_dice_score

    def __init__(self, mode="binary", log_loss=False, from_logits=True, eps=1.):
        super(DiceLoss, self).__init__()
        self.mode = Mode(mode)  # raises an error if not valid
        self.log_loss = log_loss
        self.from_logits = from_logits
        self.eps = eps

    def forward(self, y_pred, y_true):
        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            if self.mode == Mode.BINARY or self.mode == Mode.MULTILABEL:
                y_pred = y_pred.sigmoid()
            elif self.mode == Mode.MULTICLASS:
                y_pred = y_pred.softmax(dim=1)

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == Mode.BINARY:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
        elif self.mode == Mode.MULTICLASS:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            y_true = torch.nn.functional.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1)  # H, C, H*W
        elif self.mode == Mode.MULTILABEL:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

        scores = self.__class__.IOU_FUNCTION(y_pred, y_true.type(y_pred.dtype), dims=dims, eps=self.eps)

        if self.log_loss:
            loss = -torch.log(scores)
        else:
            loss = 1 - scores
        
        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        mask = y_true.sum(dims) > 0
        loss *= mask.float()

        return loss.mean()


class JaccardLoss(DiceLoss):
    """
    Implementation of Jaccard loss for image segmentation task.
    It supports binary, multiclass and multilabel cases

    Args:
        mode (str): Target mode {'binary', 'multiclass', 'multilabel'}
            'multilabel' - expects y_true of shape [N, C, H, W]
            'multiclass', 'binary' - expects y_true of shape [N, H, W]
        log_loss (bool): If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        from_logits (bool): If True assumes input is raw logits
        eps (float): small epsilon for numerical stability
    Shape:
        y_pred: [N, C, H, W]
        y_true: [N, C, H, W] or [N, H, W] depending on mode
    """

    # the only difference is which function to use
    IOU_FUNCTION = soft_jaccard_score
