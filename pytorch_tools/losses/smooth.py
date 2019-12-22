import torch
import torch.nn.functional as F
from .base import Loss, Mode


class CrossEntropyLoss(Loss):
    """
    CE with optional smoothing and support for multiple positive labels. 
    Can accept one-hot encoded y_trues
    Supports only one reduction for now
    """

    def __init__(self, mode="multiclass", smoothing=0.0, weight=None):
        """
        Args:
            mode (str): Metric mode {'binary', 'multiclass'}
                'binary' - calculate binary cross entropy
                'multiclass' - calculate categorical cross entropy
            smoothing (float): How much to smooth values toward uniform 
            weight (Tensor): A manual rescaling weight given to each class.
                If given, has to be a Tensor of size C
        """
        super().__init__()
        self.mode = Mode(mode)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        weight = torch.Tensor([1.]) if weight is None else weight
        self.register_buffer('weight', weight)

    def forward(self, y_pred, y_true):

        if self.mode == Mode.BINARY:
            y_pred, y_true = y_pred.squeeze(), y_true.squeeze()
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=self.weight)
            return loss

        if len(y_true.shape) != 1:
            y_true_one_hot = y_true.float()
        else:
            num_classes = y_pred.size(1)
            y_true_one_hot = torch.zeros(
                y_true.size(0), num_classes, dtype=torch.float, device=y_pred.device
            )
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1.0)
        y_pred = y_pred.float()
        logprobs = F.log_softmax(y_pred, dim=1)
        # loss of each sample is weighted by it's target class
        logprobs = logprobs * self.weight
        sample_weights = self.weight * y_true_one_hot
        # multiple labels handling
        nll_loss = -logprobs * y_true_one_hot
        nll_loss = nll_loss.sum(-1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.sum().div(sample_weights.sum())
