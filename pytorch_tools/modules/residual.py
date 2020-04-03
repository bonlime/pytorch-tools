import math
import torch
import torch.nn as nn
from .activated_batch_norm import ABN
from .activations import activation_from_name

# from pytorch_tools.modules import ABN
# from pytorch_tools.modules import activation_from_name
from pytorch_tools.modules import BlurPool
from pytorch_tools.modules import FastGlobalAvgPool2d
from pytorch_tools.utils.misc import make_divisible


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class DepthwiseSeparableConv(nn.Sequential):
    """Depthwise separable conv with BN after depthwise & pointwise."""

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, norm_layer=ABN, norm_act="relu"):
        modules = [
            conv3x3(in_channels, in_channels, stride=stride, groups=in_channels, dilation=dilation),
            conv1x1(in_channels, out_channels),
            norm_layer(out_channels, activation=norm_act),
        ]
        super().__init__(*modules)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dw_kernel_size=3,
        stride=1,
        dilation=1,
        use_se=False,
        expand_ratio=1.0,  # expansion
        keep_prob=1,  # drop connect param
        noskip=False,
        norm_layer=ABN,
        norm_act="relu",
    ):
        super().__init__()
        mid_chs = make_divisible(in_channels * expand_ratio)
        self.has_residual = (in_channels == out_channels and stride == 1) and not noskip
        self.has_expansion = expand_ratio != 1
        if self.has_expansion:
            self.conv_pw = conv1x1(in_channels, mid_chs)
            self.bn1 = norm_layer(mid_chs, activation=norm_act)

        self.conv_dw = nn.Conv2d(
            mid_chs,
            mid_chs,
            dw_kernel_size,
            stride=stride,
            groups=mid_chs,
            dilation=dilation,
            bias=False,
            padding=dilation * (dw_kernel_size - 1) // 2,
        )
        self.bn2 = norm_layer(mid_chs, activation=norm_act)
        # some models like MobileNet use mid_chs here instead of in_channels. But I don't care for now
        self.se = SEModule(mid_chs, in_channels // 4, norm_act) if use_se else nn.Identity()
        self.conv_pw1 = conv1x1(mid_chs, out_channels)
        self.bn3 = norm_layer(out_channels, activation="identity")
        self.drop_connect = DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()

    def forward(self, x):
        residual = x
        if self.has_expansion:
            x = self.conv_pw(x)
            x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pw1(x)
        x = self.bn3(x)

        if self.has_residual:
            x = self.drop_connect(x) + residual
        return x


class DropConnect(nn.Module):
    """Randomply drops samples from input.
    Implements idea close to one from https://arxiv.org/abs/1603.09382"""

    def __init__(self, keep_prob):
        super().__init__()
        self.keep_prob = keep_prob

    def forward(self, x):
        if not self.training:
            return x
        batch_size = x.size(0)
        random_tensor = self.keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        output = x / self.keep_prob * binary_tensor
        return output


class SEModule(nn.Module):
    def __init__(self, channels, reduction_channels, norm_act="relu"):
        super(SEModule, self).__init__()

        self.pool = FastGlobalAvgPool2d()
        # authors of original paper DO use bias
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, stride=1, bias=True)
        self.act1 = activation_from_name(norm_act)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x_se = self.pool(x)
        x_se = self.fc1(x_se)
        x_se = self.act1(x_se)
        x_se = self.fc2(x_se)
        return x * x_se.sigmoid()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        use_se=False,
        dilation=1,
        norm_layer=ABN,
        norm_act="relu",
        antialias=False,
    ):
        super(BasicBlock, self).__init__()
        antialias = antialias and stride == 2
        assert groups == 1, "BasicBlock only supports groups of 1"
        assert base_width == 64, "BasicBlock doest not support changing base width"
        outplanes = planes * self.expansion
        conv1_stride = 1 if antialias else stride
        self.conv1 = conv3x3(inplanes, planes, conv1_stride, groups, dilation)
        self.bn1 = norm_layer(planes, activation=norm_act)
        self.conv2 = conv3x3(planes, outplanes)
        self.bn2 = norm_layer(outplanes, activation="identity")
        self.se_module = SEModule(outplanes, planes // 4) if use_se else None
        self.final_act = activation_from_name(norm_act)
        self.downsample = downsample
        self.blurpool = BlurPool(channels=planes) if antialias else nn.Identity()
        self.antialias = antialias

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        # Conv(s=2)->BN->Relu(s=1) => Conv(s=1)->BN->Relu(s=1)->BlurPool(s=2)
        if self.antialias:
            out = self.blurpool(out)
        out = self.conv2(out)
        # avoid 2 inplace ops by chaining into one long op. Neede for inplaceabn
        if self.se_module is not None:
            out = self.se_module(self.bn2(out)) + residual
        else:
            out = self.bn2(out) + residual
        return self.final_act(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        use_se=False,
        dilation=1,
        norm_layer=ABN,
        norm_act="relu",
        antialias=False,
    ):
        super(Bottleneck, self).__init__()
        antialias = antialias and stride == 2
        width = int(math.floor(planes * (base_width / 64)) * groups)
        outplanes = planes * self.expansion

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, activation=norm_act)
        conv2_stride = 1 if antialias else stride
        self.conv2 = conv3x3(width, width, conv2_stride, groups, dilation)
        self.bn2 = norm_layer(width, activation=norm_act)
        self.conv3 = conv1x1(width, outplanes)
        self.bn3 = norm_layer(outplanes, activation="identity")
        self.se_module = SEModule(outplanes, planes // 4) if use_se else None
        self.final_act = activation_from_name(norm_act)
        self.downsample = downsample
        self.blurpool = BlurPool(channels=width) if antialias else nn.Identity()
        self.antialias = antialias

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)

        # Conv(s=2)->BN->Relu(s=1) => Conv(s=1)->BN->Relu(s=1)->BlurPool(s=2)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.antialias:
            out = self.blurpool(out)

        out = self.conv3(out)
        # avoid 2 inplace ops by chaining into one long op
        if self.se_module is not None:
            out = self.se_module(self.bn3(out)) + residual
        else:
            out = self.bn3(out) + residual
        return self.final_act(out)

# TResnet models use slightly modified versions of BasicBlock and Bottleneck
# need to adjust for it

class TBasicBlock(BasicBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_act = nn.ReLU(inplace=True)
        self.bn1.activation_param = 1e-3  # needed for loading weights
        if not kwargs.get("use_se"):
            return
        planes = kwargs["planes"]
        self.se_module = SEModule(planes, max(planes // 4, 64))

class TBottleneck(Bottleneck):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_act = nn.ReLU(inplace=True)
        self.bn1.activation_param = 1e-3 # needed for loading weights
        self.bn2.activation_param = 1e-3
        if not kwargs["use_se"]:
            return
        planes = kwargs["planes"]
        reduce_planes = max(planes * self.expansion // 8, 64)
        self.se_module = SEModule(planes, reduce_planes)

    # use se after 2nd conv instead of 3rd
    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)

        # Conv(s=2)->BN->Relu(s=1) => Conv(s=1)->BN->Relu(s=1)->BlurPool(s=2)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.antialias:
            out = self.blurpool(out)

        if self.se_module is not None:
            out = self.se_module(out)

        out = self.conv3(out)
        # avoid 2 inplace ops by chaining into one long op
        out = self.bn3(out) + residual
        return self.final_act(out)