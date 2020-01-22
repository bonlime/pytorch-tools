import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def global_avgmax_pool2d(x):
    x_avg = F.adaptive_avg_pool2d(x, output_size=1)
    x_max = F.adaptive_max_pool2d(x, output_size=1)
    return 0.5 * (x_avg + x_max)


def global_catavgmax_pool2d(x):
    x_avg = F.adaptive_avg_pool2d(x, output_size=1)
    x_max = F.adaptive_max_pool2d(x, output_size=1)
    return torch.cat((x_avg, x_max), 1)


def global_pool2d(x, pool_type):
    """Selectable global pooling function
    """
    if pool_type == "avg":
        x = F.adaptive_avg_pool2d(x, output_size=1)
    elif pool_type == "max":
        x = F.adaptive_max_pool2d(x, output_size=1)
    elif pool_type == "avgmax":
        x = global_avgmax_pool2d(x)
    elif pool_type == "catavgmax":
        x = global_catavgmax_pool2d(x)
    else:
        raise ValueError(f"Invalid pool type: {pool_type}")
    return x


class Flatten(nn.Module):
    """
    This modile is implemented in PyTorch 1.2
    Leave it here for PyToorch 1.1 support
    """

    def forward(self, input):
        return input.view(input.size(0), -1)


class GlobalPool2d(nn.Module):
    """Selectable global pooling layer
    
        Args:
            pool_type (str): One of 'avg', 'max', 'avgmax', 'catavgmax'
    """

    def __init__(self, pool_type):

        super(GlobalPool2d, self).__init__()
        self.pool_type = pool_type
        self.pool = partial(global_pool2d, pool_type=pool_type)

    def forward(self, x):
        return self.pool(x)

    def feat_mult(self):
        return 2 if self.pool_type == "catavgmax" else 1

    def __repr__(self):
        return self.__class__.__name__ + " (" + ", pool_type=" + self.pool_type + ")"


class BlurPool(nn.Module):
    """Idea from https://arxiv.org/abs/1904.11486
        Efficient implementation of Rect-2 using AvgPool"""

    def __init__(self):
        super(BlurPool, self).__init__()
        self.pool = nn.AvgPool2d(3, 2)

    def forward(self, inp):
        return self.pool(inp)
