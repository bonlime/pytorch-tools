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
        self.count = 0

    def update(self, val):
        self.val = val
        if self.count == 0:
            self.avg_smooth = val
        else:
            self.avg_smooth *= self.avg_mom
            self.avg_smooth += val * (1 - self.avg_mom)
        self.count += 1
        self.avg *= (self.count - 1) / self.count
        self.avg += val / self.count

    def __call__(self, val):
        return self.update(val)

    def __repr__(self):
        return f"AverageMeter(name={self.name}, avg={self.avg:.3f}, count={self.count})"
        # return f"{self.name}: {self.avg:.3f}" # maybe use this version for easier printing?


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


def reduce_meter(meter):
    """Args: meter (AverageMeter): meter to reduce"""
    if env_world_size() == 1:
        return meter
    # can't reduce AverageMeter so need to reduce every attribute separately
    reduce_attributes = ["val", "avg", "avg_smooth", "count"]
    for attr in reduce_attributes:
        old_value = to_tensor([getattr(meter, attr)]).float().cuda()
        setattr(meter, attr, reduce_tensor(old_value).cpu().numpy()[0])


def filter_bn_from_wd(model):
    """
    Filter out batch norm parameters (and bias for conv) and remove them from weight decay. Gives
    higher accuracy for large batch training.
    Idea from: https://arxiv.org/pdf/1807.11205.pdf
    Code from: https://github.com/cybertronai/imagenet18
    Args:
        model (torch.nn.Module): model
    Returns:
        dict with parameters
    """
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
    CONV_CLASSES = (
        torch.nn.modules.Linear,  # also add linear here
        torch.nn.modules.conv._ConvNd,
        pt_modules.weight_standartization.WS_Conv2d,
    )

    def _get_params(module):
        # for BN filter both weight and bias
        if isinstance(module, NORM_CLASSES):
            return module.parameters()
        # for conv & linear only filter bias
        elif isinstance(module, CONV_CLASSES) and module.bias is not None:
            return (module.bias,)  # want to return list
        accum = set()
        for child in module.children():
            [accum.add(p) for p in _get_params(child)]
        return accum

    bn_params = _get_params(model)
    bn_params2 = [p for p in model.parameters() if p in bn_params]
    rem_params = [p for p in model.parameters() if p not in bn_params]
    return [{"params": rem_params}, {"params": bn_params2, "weight_decay": 0}]


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

