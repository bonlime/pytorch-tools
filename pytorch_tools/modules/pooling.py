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
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class BlurPool(nn.Module):
    """
    Idea from https://arxiv.org/abs/1904.11486
    Efficient implementation of Rect-3 using AvgPool
    Args:
        channels (int): numbers of channels. needed for gaussian blur
        gauss (bool): flag to use Gaussian Blur instead of Average Blur. Uses more memory
    """

    def __init__(self, channels=0, gauss=False):
        super(BlurPool, self).__init__()
        self.gauss = gauss
        self.channels = channels
        # init both options to be able to switch
        a = torch.tensor([1., 2., 1.])
        filt = (a[:, None] * a[None, :]).clone().detach()
        filt = filt / torch.sum(filt)
        filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        self.register_buffer("filt", filt)
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, inp):
        if self.gauss:
            inp_pad = F.pad(inp, (1, 1, 1, 1), 'reflect')
            return F.conv2d(inp_pad, self.filt, stride=2, padding=0, groups=inp.shape[1])
        else:
            return self.pool(inp)

# from https://github.com/mrT23/TResNet/
class SpaceToDepth(nn.Module):
    def forward(self, x):
        # assuming hard-coded that block_size==4 for acceleration
        N, C, H, W = x.size()
        x = x.view(N, C, H // 4, 4, W // 4, 4)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * 16, H // 4, W // 4)  # (N, C*bs^2, H//bs, W//bs)
        return x