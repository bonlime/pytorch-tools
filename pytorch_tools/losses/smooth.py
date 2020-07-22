import torch
import torch.nn.functional as F
from .base import Loss
from .base import Mode
from .base import Reduction


class CrossEntropyLoss(Loss):
    """
    CE with optional smoothing and support for multiple positive labels.
    Can accept one-hot encoded y_trues

    Args:
        mode (str): Metric mode {'binary', 'multiclass'}
            'binary' - calculate binary cross entropy
            'multiclass' - calculate categorical cross entropy
        smoothing (float): How much to smooth values toward uniform
        weight (Tensor): A manual rescaling weight given to each class.
            If given, has to be a Tensor of size C. If `mode` is binary
            weight should be weight of positive class
        reduction (str): The reduction type to apply to the output. {'none', 'mean', 'sum'}.
            NOTE: reduction is only supported for `binary` mode! for other modes it's always `mean`
            'none' - no reduction will be applied
            'sum' - the output will be summed
            'mean' - the sum of the output will be divided by the number of elements in the output
        from_logits (bool): If False assumes sigmoid has already been applied to model output
    """

    def __init__(self, mode="multiclass", smoothing=0.0, weight=None, reduction="mean", from_logits=True):
        super().__init__()
        self.mode = Mode(mode)
        self.reduction = Reduction(reduction)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.from_logits = from_logits
        weight = torch.Tensor([1.0]) if weight is None else torch.tensor(weight)
        self.register_buffer("weight", weight)

    def forward(self, y_pred, y_true):
        if self.mode == Mode.BINARY:
            # squeeze to allow different shapes like BSx1xHxW vs BSxHxW
            if self.from_logits:
                loss = F.binary_cross_entropy_with_logits(
                    y_pred.squeeze(), y_true.squeeze(), pos_weight=self.weight, reduction=self.reduction.value
                )
            else:
                loss = F.binary_cross_entropy(  # no pos weight in this case
                    y_pred.squeeze(), y_true.squeeze(), reduction=self.reduction.value
                )
            if self.reduction == Reduction.NONE:
                loss = loss.view(*y_pred.shape)  # restore true shape
            return loss
        if len(y_true.shape) != 1:
            y_true_one_hot = y_true.float()
        else:
            y_true_one_hot = torch.zeros_like(y_pred)
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1.0)
        y_pred = y_pred.float()
        logprobs = F.log_softmax(y_pred, dim=1) if self.from_logits else y_pred.log()
        # loss of each sample is weighted by it's target class
        logprobs = logprobs * self.weight
        sample_weights = self.weight * y_true_one_hot
        # multiple labels handling
        nll_loss = -logprobs * y_true_one_hot
        nll_loss = nll_loss.sum(1)
        smooth_loss = -logprobs.mean(dim=1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.sum().div(sample_weights.sum())
