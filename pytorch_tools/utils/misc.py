import os
import math
import time
import torch
import random
import collections
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial


def initialize(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="linear")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def add_docs_for(other_func):
    """Simple decorator to concat docstrings"""

    def dec(func):
        func.__doc__ = func.__doc__ + other_func.__doc__
        return func

    return dec


def count_parameters(model):
    """Count number of parameters of a model

    Returns:
        Tuple[int, int]: Total and trainable number of parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def listify(p):
    if p is None:
        p = []
    elif not isinstance(p, collections.Iterable):
        p = [p]
    return p


def to_numpy(x):
    """Convert whatever to numpy array"""
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list) or isinstance(x, tuple):
        return np.array(x)
    else:
        raise ValueError("Unsupported type")


def to_tensor(x, dtype=None):
    """Convert whatever to torch Tensor"""
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    else:
        raise ValueError("Unsupported input type" + str(type(x)))


class AverageMeter:
    """Computes and stores the average and current value
        Attributes:
            val - last value
            avg - true average
            avg_smooth - smoothed average"""

    def __init__(self, name="Meter", avg_mom=0.9):
        self.avg_mom = avg_mom
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.avg_smooth = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        if self.count == 0:
            self.avg_smooth = val
        else:
            self.avg_smooth = self.avg_smooth * self.avg_mom + val * (1 - self.avg_mom)
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


class TimeMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.start = time.time()

    def batch_start(self):
        self.data_time.update(time.time() - self.start)

    def batch_end(self):
        self.batch_time.update(time.time() - self.start)
        self.start = time.time()


DEFAULT_IMAGENET_SETTINGS = {
    "input_space": "RGB",
    "input_size": [3, 224, 224],
    "input_range": [0, 1],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "num_classes": 1000,
}


def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def env_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def env_rank():
    return int(os.environ.get("RANK", 0))


def reduce_tensor(tensor):
    return sum_tensor(tensor) / env_world_size()


def sum_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def filter_bn_from_wd(model):
    """
    Filter out batch norm parameters and remove them from weight decay. Gives
    higher accuracy for large batch training.
    Idea from: https://arxiv.org/pdf/1807.11205.pdf
    Code from: https://github.com/cybertronai/imagenet18
    Args:
        model (torch.nn.Module): model
    Returns:
        dict with parameters
    """

    def get_bn_params(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            return module.parameters()
        accum = set()
        for child in module.children():
            [accum.add(p) for p in get_bn_params(child)]
        return accum

    bn_params = get_bn_params(model)
    bn_params2 = [p for p in model.parameters() if p in bn_params]
    rem_params = [p for p in model.parameters() if p not in bn_params]
    return [{"params": bn_params2, "weight_decay": 0}, {"params": rem_params}]


def make_divisible(v, divisor=8):
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:  # ensure round down does not go down by more than 10%.
        new_v += divisor
    return new_v

def repeat_channels(conv_weights, new_channels, old_channels=3):
    """Repeat channels to match new number of input channels
    Args:
        conv_weights (torch.Tensor): shape [*, old_channels, *, *]
        new_channels (int): desired number of channels
        old_channels (int): original number of channels
    """
    rep_times = math.ceil(new_channels / old_channels)
    new_weights = conv_weights.repeat(1, rep_times, 1, 1)[:, :new_channels, :, :]
    new_weights *= old_channels / new_channels # to keep the same output amplitude
    return new_weights