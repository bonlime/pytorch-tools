from __future__ import absolute_import
from .lr_finder import LRFinder
from .radam import RAdam, PlainRAdam, AdamW
from .sgdw import SGDW
from .adamw import AdamW
from .novograd import Novograd
from .schedulers import LinearLR, ExponentialLR

from torch import optim
def optimizer_from_name(optim_name):
    optim_name = optim_name.lower()
    if optim_name == 'sgd':
        return optim.SGD
    elif optim_name == 'sgdw': 
        return SGDW
    elif optim_name == 'adam':
        return optim.Adam
    elif optim_name =='adamw':
        return AdamW
    elif optim_name =='rmsprop':
        return optim.RMSprop
    elif optim_name == 'radam':
        return RAdam
    elif optim_name == 'novograd':
        return Novograd
    else:
        raise ValueError('Optimizer {} not found'.format(optim_name))