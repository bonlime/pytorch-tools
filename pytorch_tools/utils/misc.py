import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from inplace_abn import ABN, InPlaceABN, InPlaceABNSync

def initialize(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def activation_from_name(act_name, act_param=0.01):
    if act_name == 'relu':
        return partial(F.relu, inplace=True)
    elif act_name == 'leaky_relu':
        return partial(F.leaky_relu, negative_slope=act_param, inplace=True)
    elif act_name == "elu":
        return partial(F.elu, alpha=act_param, inplace=True)
    elif act_name == "identity":
        return lambda x: x
    elif act_name == 'sigmoid':
        return nn.Sigmoid()
    elif act_name == 'softmax':
        return nn.Softmax2d()
    else:
        raise ValueError("Activation name {} not supported".format(act_name))

# TODO return proper activation and layer 
def bn_from_name(norm_name):
    norm_name = norm_name.lower()
    if norm_name == 'abn':
        return ABN
    elif norm_name == 'inplaceabn':
        return InPlaceABN
    elif norm_name == 'inplaceabnsync':
        return InPlaceABNSync
    else:
        raise ValueError("Normalization {} not supported".format(norm_name))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_docs_for(other_func):
    """Simple decorator to concat docstrings"""
    def dec(func):
        func.__doc__ = func.__doc__ + other_func.__doc__
        return func
    return dec
