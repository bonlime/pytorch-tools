from .base import Loss
from .base import Reduction

class SmoothL1Loss(Loss):
    """Huber loss aka Smooth L1 Loss
    
    loss = 0.5 * x^2                  if |x| <= d
    loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d

    Args:
        delta (float): point where the Huber loss function changes from a quadratic to linear
        reduction (str): The reduction type to apply to the output. {'none', 'mean', 'sum'}.
            'none' - no reduction will be applied
            'sum' - the output will be summed
            'mean' - the sum of the output will be divided by the number of elements in the output
    """

    def __init__(self, delta=0.1, reduction="none"):
        super().__init__()
        self.delta = delta
        self.reduction = Reduction(reduction)

    def forward(self, pred, target):
        x = (pred - target).abs()
        l1 = self.delta * (x - 0.5 * self.delta)
        l2 = 0.5 * x.pow(2)

        loss = l1.where(x >= self.delta, l2)
        if self.reduction == Reduction.MEAN:
            loss = loss.mean()
        elif self.reduction == Reduction.SUM:
            loss = loss.sum()
        return loss