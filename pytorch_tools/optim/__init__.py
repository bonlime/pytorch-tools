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


def optimizer_from_name(optim_name):
    optim_name = optim_name.lower()
    if optim_name == "sgd":
        return optim.SGD
    elif optim_name == "sgdw":
        return SGDW
    elif optim_name == "adam":
        return optim.Adam
    elif optim_name == "adamw":
        return optim.AdamW
    elif optim_name == "adamw_gc":
        return partial(AdamW_my, center=True)
    elif optim_name == "rmsprop":
        return optim.RMSprop
    elif optim_name == "radam":
        return RAdam
    elif optim_name in ["fused_sgd", "fusedsgd"]:
        return FusedSGD
    elif optim_name in ["fused_adam", "fusedadam"]:
        return FusedAdam
    elif optim_name in ["fused_novograd", "fusednovograd", "novograd"]:
        return FusedNovoGrad
    else:
        raise ValueError(f"Optimizer {optim_name} not found")
