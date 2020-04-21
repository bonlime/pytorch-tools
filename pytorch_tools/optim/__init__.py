from __future__ import absolute_import
from functools import partial
from apex.optimizers import FusedNovoGrad, FusedAdam, FusedSGD
from .lr_finder import LRFinder
from .adamw import AdamW as AdamW_my
from .radam import RAdam, PlainRAdam
from .sgdw import SGDW
from .schedulers import LinearLR, ExponentialLR
from .lookahead import Lookahead

from torch import optim

# 2e-5 is the lowest epsilon than saves from overflow in fp16
def optimizer_from_name(optim_name):
    optim_name = optim_name.lower()
    if optim_name == "sgd":
        return optim.SGD
    elif optim_name == "sgdw":
        return SGDW
    elif optim_name == "adam":
        return partial(optim.Adam, eps=2e-5)
    elif optim_name == "adamw":
        return partial(AdamW_my, eps=2e-5)
    elif optim_name == "adamw_gc":
        # in this implementation eps in inside sqrt so it can be smaller
        return partial(AdamW_my, center=True, eps=1e-7)
    elif optim_name == "rmsprop":
        return partial(optim.RMSprop, 2e-5)
    elif optim_name == "radam":
        return partial(RAdam, eps=2e-5)
    elif optim_name in ["fused_sgd", "fusedsgd"]:
        return FusedSGD
    elif optim_name in ["fused_adam", "fusedadam"]:
        return partial(FusedAdam, eps=2e-5)
    elif optim_name in ["fused_novograd", "fusednovograd", "novograd"]:
        return partial(FusedNovoGrad, eps=2e-5)
    else:
        raise ValueError(f"Optimizer {optim_name} not found")
