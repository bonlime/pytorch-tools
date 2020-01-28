from .pooling import GlobalPool2d
from .pooling import BlurPool
from .pooling import Flatten

from .residual import BasicBlock
from .residual import Bottleneck
from .residual import SEModule

# from .residual import Transition, DenseLayer

from .activations import ACT_DICT
from .activations import ACT_FUNC_DICT
from .activations import activation_from_name
from .activations import Mish, MishNaive, Swish, SwishNaive

from .activated_batch_norm import ABN
from inplace_abn import InPlaceABN, InPlaceABNSync


def bn_from_name(norm_name):
    norm_name = norm_name.lower()
    if norm_name == "abn":
        return ABN
    elif norm_name == "inplaceabn" or "inplace_abn":
        return InPlaceABN
    elif norm_name == "inplaceabnsync":
        return InPlaceABNSync
    else:
        raise ValueError(f"Normalization {norm_name} not supported")
