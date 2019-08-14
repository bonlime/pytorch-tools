import torch.nn.functional as F
from functools import partial


def activation_from_name(act_name, act_param=0.01):
    if act_name == 'relu':
        return partial(F.relu, inplace=True)
    elif act_name == 'leaky_relu':
        return partial(F.leaky_relu, negative_slope=act_param, inplace=True)
    elif act_name == "elu":
        return partial(F.elu, alpha=act_param, inplace=True)
    elif act_name == "identity":
        return lambda x: x


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
