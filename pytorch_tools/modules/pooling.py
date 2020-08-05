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


# https://github.com/mrT23/TResNet/
class FastGlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        return x.mean(dim=(2, 3), keepdim=not self.flatten)


class BlurPool(nn.Module):
    """
    Idea from https://arxiv.org/abs/1904.11486
    Efficient implementation of Rect-3
    Args:
        channels (int): numbers of input channels. needed to construct gauss kernel
    """

    def __init__(self, channels=0):
        super(BlurPool, self).__init__()
        self.channels = channels
        filt = torch.tensor([1.0, 2.0, 1.0])
        filt = filt[:, None] * filt[None, :]
        filt = filt / torch.sum(filt)
        filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        self.register_buffer("filt", filt)

    def forward(self, inp):
        inp_pad = F.pad(inp, (1, 1, 1, 1), "reflect")
        return F.conv2d(inp_pad, self.filt, stride=2, padding=0, groups=inp.shape[1])


# from https://github.com/mrT23/TResNet/
class SpaceToDepth(nn.Module):
    def __init__(self, block_size=4):
        super().__init__()
        assert block_size in {2, 4}, "Space2Depth only supports blocks size = 4 or 2"
        self.block_size = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        S = self.block_size
        x = x.view(N, C, H // S, S, W // S, S)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * S * S, H // S, W // S)  # (N, C*bs^2, H//bs, W//bs)
        return x
