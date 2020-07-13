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
    SWISH_NAIVE = "swish_naive"
    SWISH_HARD = "swish_hard"  # hard swish


#### SWISH ####


@torch.jit.script
def swish_jit_fwd(x):
    return x.mul(torch.sigmoid(x))


@torch.jit.script
def swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


class SwishFunction(torch.autograd.Function):
    """ torch.jit.script optimised Swish
    Inspired by conversation btw Jeremy Howard & Adam Pazske
    https://twitter.com/jeremyphoward/status/1188251041835315200
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad_output)


def swish(x, inplace=False):
    # inplace ignored
    return SwishFunction.apply(x)


def swish_naive(x, inplace=False):
    return x * x.sigmoid()


class Swish(nn.Module):
    """Memory efficient and fast version of Swish. CAN NOT be exported by tracing"""

    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        return SwishFunction.apply(x)


class SwishNaive(nn.Module):
    """Naive version of Swish CAN be exported by tracing"""

    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        return swish_naive(x)


# FIXME: remove when master starts to support `inplace` for module. probably after 1.6 or 1.7
class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.hardswish(input, inplace=self.inplace)


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


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


ACT_DICT = {
    ACT.CELU: nn.CELU,
    ACT.ELU: nn.ELU,
    ACT.GLU: nn.GLU,
    ACT.IDENTITY: Identity,
    ACT.LEAKY_RELU: nn.LeakyReLU,
    ACT.MISH: Mish,
    ACT.MISH_NAIVE: MishNaive,
    ACT.NONE: Identity,
    ACT.PRELU: nn.PReLU,
    ACT.RELU: nn.ReLU,
    ACT.RELU6: nn.ReLU6,
    ACT.SELU: nn.SELU,
    ACT.SWISH: Swish,
    ACT.SWISH_NAIVE: SwishNaive,
    ACT.SWISH_HARD: HardSwish,
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
    ACT.SWISH: swish,
    ACT.SWISH_NAIVE: swish_naive,
    ACT.SWISH_HARD: F.hardswish,
}


def activation_from_name(activation_name):
    if type(activation_name) == str:
        activation_name = ACT(activation_name.lower())
    return ACT_DICT[activation_name](inplace=True)


def sanitize_activation_name(activation_name: str) -> str:
    """
    Return reasonable activation name for initialization in `kaiming_uniform_` for hipster activations
    """
    if activation_name in {ACT.MISH, ACT.SWISH, ACT.SWISH_NAIVE}:
        return ACT.LEAKY_RELU

    return activation_name
