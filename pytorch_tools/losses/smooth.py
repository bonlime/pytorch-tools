import torch
from .base import Loss
    
class CrossEntropyLoss(Loss):
    """CE with optional smoothing and support for multiple positive labels. Supports only one reduction for now
    Args:
        one_hot (bool): set to False if you pass LongTensor as targets
        num_classes (int): number of classes. only used if one_hot is False
    """
    def __init__(self, smoothing = 0.0, one_hot=True, num_classes=None):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        if not one_hot and num_classes == None:
            raise ValueError('To convert data to one_hot you have to specify num_classes')
        self.num_classes = num_classes

    def forward(self, x, target):
        # check is explicit on purpose, don't rely on self.one_hot
        if len(target.shape) == 2:
            target_one_hot = target.float()
        else:
            target_one_hot = torch.zeros(target.size(0), self.num_classes, 
                                            dtype=torch.float, device=x.device)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
        x = x.float()
        logprobs = torch.nn.functional.log_softmax(x, dim = -1)
        # mupltiple labels handling
        nll_loss = -logprobs * target_one_hot
        nll_loss = nll_loss.sum(-1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()