from functools import partial
from loguru import logger

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
from .weight_standartization import conv_to_ws_conv

from .activations import ACT_DICT
from .activations import ACT_FUNC_DICT
from .activations import activation_from_name
from .activations import Mish, MishNaive

from .activated_batch_norm import ABN
from .activated_batch_norm import SyncABN
from .activated_group_norm import AGN
from .activated_batch_channel_norm import ABCN
from .activated_no_norm import NoNormAct

from . import conv

try:
    from inplace_abn import InPlaceABN, InPlaceABNSync

    HAS_INPLACE_ABN = True
except (ModuleNotFoundError, ImportError):
    HAS_INPLACE_ABN = False
    InPlaceABN = ABN
    InPlaceABNSync = ABN


def bn_from_name(norm_name):
    norm_name = norm_name.lower()
    if norm_name == "abn":
        return ABN
    elif norm_name in ("syncabn", "sync_abn", "abn_sync"):
        return SyncABN
    elif norm_name in ("inplaceabn", "inplace_abn") and HAS_INPLACE_ABN:
        return InPlaceABN
    elif norm_name == "inplaceabnsync" and HAS_INPLACE_ABN:
        return InPlaceABNSync
    elif norm_name in ("frozen_abn", "frozenabn"):
        return partial(ABN, frozen=True)
    # not sure anyone would ever need this but let's support just in case
    elif norm_name == "frozen_sync_abn":
        return partial(SyncABN, frozen=True)
    elif norm_name == "estimated_abn":
        return partial(ABN, estimated_stats=True)
    elif norm_name in ("agn", "groupnorm", "group_norm"):
        return AGN
    elif norm_name == "abcn":
        return partial(ABCN)  # for large batch
    elif norm_name in ("abcn_micro", "estimated_abcn"):
        return partial(ABCN, estimated_stats=True)  # for micro batch
    elif norm_name in ("none",):
        return NoNormAct
    else:
        raise ValueError(f"Normalization '{norm_name}' not supported or inplaceabn is not installed")
