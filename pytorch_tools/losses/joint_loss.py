from torch.nn.modules.loss import _Loss
import torch

class WeightedLoss(_Loss):
    """
    Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = torch.Tensor([weight])

    def forward(self, *input):
        l = self.loss(*input)
        self.weight = self.weight.to(l.device)
        return l * self.weight


class JointLoss(_Loss):
    """
    Class for multiple loss functions training
    input: 
        losses: tuple or list with losses to be combined
        weights: float or list of weights to balance losses
          if list is given as an argument, 
          it's length should be equal to length of `losses` list

    """
    def __init__(self, losses, weights=1.0):
        super().__init__()
        if type(weights) in (float, int):
            weights = [weights] * len(losses)

        self.weighted_losses = []
        for loss, weight in zip(losses, weights):
            self.weighted_losses.append(WeightedLoss(loss, weight))
        print(self.weighted_losses)

    def forward(self, *input):
        output = []
        for loss in self.weighted_losses:
            output.append(loss(*input))
        return sum(output)
