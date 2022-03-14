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
from typing import Optional, List, Dict

# 1.71 is default for ReLU. see NFNet paper for details and timm's implementation
def initialize_fn(m: nn.Module, gamma: float = 1.71):
    if isinstance(m, nn.Conv2d):
        # nn.init.kaiming_uniform_ doesn't take into account groups
        # remove when https://github.com/pytorch/pytorch/issues/23854 is resolved
        m.weight.data.normal_(0, gamma / m.weight[0].numel() ** 0.5)  # gamma * 1 / sqrt(fan-in)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    # No check for BN because in PyTorch it is initialized with 1 & 0 by default
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="linear")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def initialize(module: nn.Module, gamma: float = 1.71):
    iterator = module.modules() if hasattr(module, "modules") else module
    for m in iterator:
        initialize_fn(m, gamma=gamma)


def zero_mean_conv_weight(w: torch.Tensor):
    """zero mean conv would prevent mean shift in the network during training"""
    return w - w.mean(dim=(1, 2, 3), keepdim=True)


def normalize_conv_weight(w: torch.Tensor, gamma: float = 1):
    """w: Conv2d weight matrix; gamma: nonlinearity gain. should be 1 for identity, 1.72 for relu. see ... for details
    Idea for implementation is borrowed from `timm` package
    """
    scale = torch.full((w.size(0),), fill_value=gamma * w[0].numel() ** -0.5, device=w.device)
    return F.batch_norm(w.view(1, w.size(0), -1), None, None, weight=scale, training=True, momentum=0).reshape_as(w)


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
    elif isinstance(x, (list, tuple, int, float)):
        return np.array(x)
    else:
        raise ValueError("Unsupported type")


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


def filter_from_weight_decay(model: nn.Module, skip_list: Optional[List[str]] = None):
    decay = []
    no_decay = []
    skip_list = listify(skip_list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        in_stop_list = any(skip_word in name for skip_word in skip_list)
        if len(param.shape) == 1 or name.endswith(".bias") or in_stop_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": decay}, {"params": no_decay, "weight_decay": 0}]


def patch_bn_mom(module, momentum: float = 0.01):
    """changes default bn momentum"""
    # import inside function to avoid problems with circular imports
    import pytorch_tools.modules as pt_modules

    NORM_CLASSES = (
        torch.nn.modules.batchnorm._BatchNorm,
        pt_modules.ABN,
        pt_modules.SyncABN,
        pt_modules.AGN,
        pt_modules.InPlaceABN,
        pt_modules.InPlaceABNSync,
    )
    if isinstance(module, NORM_CLASSES):
        module.momentum = momentum

    for m in module.children():
        patch_bn_mom(m, momentum)


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
    new_weights *= old_channels / new_channels  # to keep the same output amplitude
    return new_weights


# basic CudaLoader more than enough for majority of problems
class ToCudaLoader:
    """Simple wrapper which moves batches to cuda. Usage: loader = ToCudaLoader(loader)"""

    def __init__(self, loader):
        self.loader = loader
        self.batch_size = loader.batch_size

    def __iter__(self):
        return ([i.cuda(non_blocking=True), t.cuda(non_blocking=True)] for i, t in self.loader)

    def __len__(self):
        return len(self.loader)


def update_dict(to_dict: Dict, from_dict: Dict) -> Dict:
    """close to `to_dict.update(from_dict)` but correctly updates internal dicts"""
    for k, v in from_dict.items():
        if hasattr(v, "keys") and k in to_dict.keys():
            to_dict[k].update(v)
        else:
            to_dict[k] = v
    return to_dict
