from functools import partial

from .pooling import FastGlobalAvgPool2d
from .pooling import SpaceToDepth
from .pooling import GlobalPool2d
from .pooling import BlurPool
from .pooling import Flatten

from .residual import BasicBlock
from .residual import Bottleneck
from .residual import TBasicBlock
from .residual import TBottleneck
from .residual import SEModule

# from .residual import Transition, DenseLayer

from .activations import ACT_DICT
from .activations import ACT_FUNC_DICT
from .activations import activation_from_name
from .activations import Mish, MishNaive, Swish, SwishNaive

from .activated_batch_norm import ABN
from .activated_group_norm import AGN
from inplace_abn import InPlaceABN, InPlaceABNSync

def bn_from_name(norm_name):
    norm_name = norm_name.lower()
    if norm_name == "abn":
        return ABN
    elif norm_name in ("inplaceabn", "inplace_abn"):
        return InPlaceABN
    elif norm_name == "inplaceabnsync":
        return InPlaceABNSync
    elif norm_name in ("frozen_abn", "frozenabn"):
        return partial(ABN, frozen=True)
    elif norm_name in ("agn", "groupnorm", "group_norm"):
        return AGN
    else:
        raise ValueError(f"Normalization {norm_name} not supported")
