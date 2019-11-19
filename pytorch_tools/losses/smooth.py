import torch
from .base import Loss


class CrossEntropyLoss(Loss):
    """
    CE with optional smoothing and support for multiple positive labels. 
    Can accept one-hot encoded targets
    Supports only one reduction for now
    """
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, y_pred, target):
        # check is explicit on purpose, don't rely on self.one_hot
        if len(target.shape) == 2:
            target_one_hot = target.float()
        else:
            num_classes = y_pred.size(1)
            target_one_hot = torch.zeros(target.size(0), num_classes, 
                                         dtype=torch.float, device=y_pred.device)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
        y_pred = y_pred.float()
        logprobs = torch.nn.functional.log_softmax(y_pred, dim=1)
        # mupltiple labels handling
        nll_loss = -logprobs * target_one_hot
        nll_loss = nll_loss.sum(-1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()