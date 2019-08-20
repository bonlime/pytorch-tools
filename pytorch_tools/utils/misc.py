import torch.nn as nn
import time
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
   
def count_parameters(model):
    """Count number of parameters of a model

    Returns:
        Tuple[int, int]: Total and trainable number of parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def listify(p=None, q=None):
    if p is None:
        p = []
    elif not isinstance(p, collections.Iterable):
        p = [p]
    n = q if type(q) == int else 1 if q is None else len(q)
    if len(p) == 1:
        p = p * n
    return p

def to_numpy(x):
    """Convert whatever to numpy array"""
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, list) or isinstance(x, tuple):
        return np.array(x)
    else:
        raise ValueError('Unsupported type')
    return x

class AverageMeter:
    """Computes and stores the average and current value
        Attributes:
            val - last value
            avg - true average
            avg_smooth - smoothed average"""
    def __init__(self, avg_mom=0.9):
        self.val = 0
        self.avg = 0
        self.avg_smooth = 0
        self.sum = 0
        self.count = 0
        self.avg_mom = avg_mom

    def update(self, val):
        self.val = val
        if self.count == 0:
            self.avg_smooth = val
        else:
            self.avg_smooth = self.avg_smooth*self.avg_mom + val*(1-self.avg_mom)
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

class TimeMeter:
  def __init__(self):
    self.batch_time = AverageMeter()
    self.data_time = AverageMeter()
    self.start = time.time()

  def batch_start(self):
    self.data_time.update(time.time() - self.start)

  def batch_end(self):
    self.batch_time.update(time.time() - self.start)
    self.start = time.time()

DEFAULT_IMAGENET_SETTINGS = {
    'input_space': 'RGB',
    'input_size': [3, 224, 224],
    'input_range': [0, 1],
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'num_classes': 1000
}