from enum import Enum
import torch
from torch import nn
from torch.nn import functional as F


class ACT(Enum):
    # Activation names
    CELU = "celu"
    ELU = "elu"
    GLU = "glu"
    IDENTITY = "identity"
    LEAKY_RELU = "leaky_relu"
    MISH = "mish"
    MISH_NAIVE = "mish_naive"
    NONE = "none"
    PRELU = "prelu"
    RELU = "relu"
    RELU6 = "relu6"
    SELU = "selu"
    SWISH = "swish"
    SILU = "silu"
    SWISH_HARD = "swish_hard"  # hard swish


#### MISH ####
# There is equivalent formulation of Mish which could be faster but isn't (in my tests). So not adding it for now
# feel free to open PR if you manage to speed it up
# https://github.com/digantamisra98/Mish/issues/22
# https://github.com/ultralytics/yolov3/issues/1098


@torch.jit.script
def mish_jit_fwd(x):
    return x.mul(torch.tanh(F.softplus(x)))


@torch.jit.script
def mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


class MishFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return mish_jit_fwd(x)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return mish_jit_bwd(x, grad_output)


def mish(x, inplace=False):
    # inplace ignored
    return MishFunction.apply(x)


def mish_naive(x, inplace=False):
    return x.mul(F.softplus(x).tanh())


class Mish(nn.Module):
    """Memory efficient and fast version of Mish. CAN NOT be exported by tracing"""

    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        return MishFunction.apply(x)


class MishNaive(nn.Module):
    """Naive version of Mish CAN be exported by tracing"""

    def __init__(self, inplace=True):
        super().__init__()

    def forward(self, x):
        return mish_naive(x)


#### Helpfull functions ####
def identity(x, *args, **kwargs):
    return x


ACT_DICT = {
    ACT.CELU: nn.CELU,
    ACT.ELU: nn.ELU,
    ACT.GLU: nn.GLU,
    ACT.IDENTITY: nn.Identity,
    ACT.LEAKY_RELU: nn.LeakyReLU,
    ACT.MISH: Mish,
    ACT.MISH_NAIVE: MishNaive,
    ACT.NONE: nn.Identity,
    ACT.PRELU: nn.PReLU,
    ACT.RELU: nn.ReLU,
    ACT.RELU6: nn.ReLU6,
    ACT.SELU: nn.SELU,
    ACT.SWISH: nn.SiLU,
    ACT.SILU: nn.SiLU,
    ACT.SWISH_HARD: nn.Hardswish,
}

ACT_FUNC_DICT = {
    ACT.CELU: F.celu,
    ACT.ELU: F.elu,
    ACT.GLU: F.elu,
    ACT.IDENTITY: identity,
    ACT.LEAKY_RELU: F.leaky_relu,
    ACT.MISH: mish,
    ACT.MISH_NAIVE: mish_naive,
    ACT.NONE: identity,
    ACT.PRELU: F.prelu,
    ACT.RELU: F.relu,
    ACT.RELU6: F.relu6,
    ACT.SELU: F.selu,
    ACT.SWISH: F.silu,
    ACT.SWISH_HARD: F.hardswish,
}


def activation_from_name(activation_name, inplace=True):
    if type(activation_name) == str:
        activation_name = ACT(activation_name.lower())
    return ACT_DICT[activation_name](inplace=inplace)
