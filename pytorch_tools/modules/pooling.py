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
    if pool_type == 'avg':
        x = F.adaptive_avg_pool2d(x, output_size=1)
    elif pool_type == 'max':
        x = F.adaptive_max_pool2d(x, output_size=1)
    elif pool_type == 'avgmax':
        x = global_avgmax_pool2d(x)
    elif pool_type == 'catavgmax':
        x = global_catavgmax_pool2d(x)
    else:
        raise ValueError('Invalid pool type: {}'.format(pool_type))
    return x

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
        return 2 if self.pool_type == 'catavgmax' else 1

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + ', pool_type=' + self.pool_type + ')'

class BlurPool(nn.Module):
    """Idea from https://arxiv.org/abs/1904.11486
       Efficient implementation of Rect-2 with stride 2"""
    def __init__(self):
        super(BlurPool, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)