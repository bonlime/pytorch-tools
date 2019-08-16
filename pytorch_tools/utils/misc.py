
import time
import torch.nn.functional as F
from functools import partial
from inplace_abn import ABN, InPlaceABN, InPlaceABNSync


def activation_from_name(act_name, act_param=0.01):
    if act_name == 'relu':
        return partial(F.relu, inplace=True)
    elif act_name == 'leaky_relu':
        return partial(F.leaky_relu, negative_slope=act_param, inplace=True)
    elif act_name == "elu":
        return partial(F.elu, alpha=act_param, inplace=True)
    elif act_name == "identity":
        return lambda x: x
    else:
        raise ValueError("Activation name {} not supported".format(act_name))

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

def count_parameters(model):
    """Count number of parameters of a model

    Returns:
        Tuple[int, int]: Total and trainable number of parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


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
