import math
import torch
import torch.nn as nn
from functools import partial

from .activated_batch_norm import ABN
from .activations import activation_from_name
from .conv import conv3x3, conv1x1

# from pytorch_tools.modules import ABN
# from pytorch_tools.modules import activation_from_name
from pytorch_tools.modules import BlurPool
from pytorch_tools.modules import FastGlobalAvgPool2d
from pytorch_tools.utils.misc import make_divisible
from pytorch_tools.modules import SpaceToDepth


class SEModule(nn.Module):
    def __init__(self, channels, reduction_channels, norm_act="relu"):
        super(SEModule, self).__init__()

        self.pool = FastGlobalAvgPool2d()
        # authors of original paper DO use bias
        self.fc1 = conv1x1(channels, reduction_channels, bias=True)
        self.act1 = activation_from_name(norm_act)
        self.fc2 = conv1x1(reduction_channels, channels, bias=True)

    def forward(self, x):
        x_se = self.pool(x)
        x_se = self.fc1(x_se)
        x_se = self.act1(x_se)
        x_se = self.fc2(x_se)
        return x * x_se.sigmoid()


class SEVar3(nn.Module):
    """Variant of SE module from ECA paper (see above) which doesn't have dimensionality reduction"""

    def __init__(self, channels, *args):
        super().__init__()

        self.pool = FastGlobalAvgPool2d()
        # authors of original paper DO use bias
        self.fc1 = conv1x1(channels, channels, bias=True)

    def forward(self, x):
        return x * self.fc1(self.pool(x)).sigmoid()


class ECAModule(nn.Module):
    """Efficient Channel Attention
    This implementation is different from the paper. I've removed all hyperparameters and
    use fixed kernel size of 3. If you think it may be better to use different k_size - feel free to open an issue.
    upd. 12.01.2020 increase kernel size

    Ref: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    https://arxiv.org/abs/1910.03151

    """

    def __init__(self, *args, kernel_size=3, **kwargs):
        super().__init__()
        self.pool = FastGlobalAvgPool2d()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        x_s = self.pool(x)
        x_s = self.conv(x_s.view(x.size(0), 1, -1))
        x_s = x_s.view(x.size(0), -1, 1, 1).sigmoid()
        return x * x_s.expand_as(x)


class SSEModule(nn.Module):
    """Spatial Excitation Block (sSE)
    Attention which excites certain locations in spatial domain instead of channel. Works better for segmentation than SE
    Ref: Recalibrating Fully Convolutional Networks with Spatial and Channel ‘Squeeze & Excitation’ Blocks
    https://arxiv.org/abs/1808.08127
    """

    def __init__(self, in_ch, *args):  # parse additional args for compatability
        super().__init__()
        self.conv = conv1x1(in_ch, 1, bias=True)

    def forward(self, x):
        return x * self.conv(x).sigmoid()


class SCSEModule(nn.Module):
    """Idea from Spatial and Channel ‘Squeeze & Excitation’ (scSE)
    ECA is proven to work better than (c)SE so i'm using ECA + sSE instead of original cSE + sSE

    NOTE: This modules also performs additional conv to return the same number of channels as before

    Ref: Recalibrating Fully Convolutional Networks with Spatial and Channel ‘Squeeze & Excitation’ Blocks
    https://arxiv.org/abs/1808.08127

    Ref: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    https://arxiv.org/abs/1910.03151
    """

    def __init__(self, in_ch, *args):  # parse additional args for compatability
        super().__init__()
        self.sse = SSEModule(in_ch)
        self.cse = ECAModule()
        self.reduction_conv = conv1x1(in_ch * 2, in_ch, bias=True)  # use bias because there is no BN after

    def forward(self, x):
        return self.reduction_conv(torch.cat([self.sse(x), self.cse(x)], dim=1))


class MSCAMModule(nn.Module):
    """Idea from Attentional Feature Fusion (MS-CAM)
    Combines global and local attention in a better manner than scSE
    """

    def __init__(self, in_ch, reduced_ch, norm_layer=ABN, norm_act="relu"):  # parse additional args for compatability
        super().__init__()
        self.global_attn = nn.Sequential(
            FastGlobalAvgPool2d(),
            conv1x1(in_ch, reduced_ch),
            norm_layer(reduced_ch, activation=norm_act),
            conv1x1(reduced_ch, in_ch),
            norm_layer(in_ch, activation="identity"),  # no last activation
        )
        # this
        self.local_attn = nn.Sequential(
            conv1x1(in_ch, reduced_ch),
            norm_layer(reduced_ch, activation=norm_act),
            conv1x1(reduced_ch, in_ch),
            norm_layer(in_ch, activation="identity"),  # no last activation
        )

    def forward(self, x):
        xl = self.local_attn(x)
        xg = self.global_attn(x)
        return x * (xl + xg).sigmoid()


class FCAAttn(nn.Module):
    """Inspired by FcaNet: Frequency Channel Attention Networks (https://arxiv.org/pdf/2012.11879.pdf)

    But I view it as positional encoding and multiply be a predefined set of filters
    """

    def __init__(self, channels, reduction_channels, norm_act="relu"):

        super().__init__()

        # self.pool = FastGlobalAvgPool2d()
        # authors of original paper DO use bias
        self.fc = nn.Sequential(
            conv1x1(channels, reduction_channels, bias=True),
            activation_from_name(norm_act),
            conv1x1(reduction_channels, channels, bias=True),
            nn.Sigmoid(),
        )
        # dummy shape. would be overwritten later. not registering as buffer intentionally
        self.pos_encoding = torch.ones(1, 1, 1, 1)

    def _get_pos_encoding(self, inp):
        """Want this to be generated for each input size separately"""
        self.pos_encoding = torch.ones_like(inp)
        c_part = inp.size(1) // 4
        xx = torch.linspace(0, math.pi, inp.size(3)).cos()[None].repeat(inp.size(2), 1)
        yy = torch.linspace(0, math.pi, inp.size(2)).cos()[None].repeat(inp.size(3), 1).T
        xy = torch.linspace(-math.pi, math.pi, inp.size(3)).cos().neg().repeat(inp.size(2), 1)
        self.pos_encoding[:, c_part * 1 : c_part * 2] = xx
        self.pos_encoding[:, c_part * 2 : c_part * 3] = yy
        self.pos_encoding[:, c_part * 3 : c_part * 4] = xy
        return self.pos_encoding

    def forward(self, x):
        if x.shape != self.pos_encoding.shape:
            self._get_pos_encoding(x)
        x_se = (x * self.pos_encoding).sum(dim=(2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * x_se


class FCA_ECA_Attn(nn.Module):
    """Inspired by FcaNet: Frequency Channel Attention Networks (https://arxiv.org/pdf/2012.11879.pdf)

    But I view it as positional encoding and multiply be a predefined set of filters
    This class uses Efficient Channel Attention instead of SE
    """

    def __init__(self, *args, kernel_size=3, **kwargs):
        super().__init__()
        self.pool = FastGlobalAvgPool2d()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        # dummy shape. would be overwritten later. not registering as buffer intentionally
        self.pos_encoding = torch.ones(1, 1, 1, 1)

    def _get_pos_encoding(self, inp):
        """Want this to be generated for each input size separately"""
        self.pos_encoding = torch.ones_like(inp)
        c_part = inp.size(1) // 4
        xx = torch.linspace(0, math.pi, inp.size(3)).cos()[None].repeat(inp.size(2), 1)
        yy = torch.linspace(0, math.pi, inp.size(2)).cos()[None].repeat(inp.size(3), 1).T
        xy = torch.linspace(-math.pi, math.pi, inp.size(3)).cos().neg().repeat(inp.size(2), 1)
        self.pos_encoding[:, c_part * 1 : c_part * 2] = xx
        self.pos_encoding[:, c_part * 2 : c_part * 3] = yy
        self.pos_encoding[:, c_part * 3 : c_part * 4] = xy
        return self.pos_encoding

    def forward(self, x):
        # FCA part
        if x.shape != self.pos_encoding.shape:
            self._get_pos_encoding(x)
        # ECA part
        x_s = self.pool(x * self.pos_encoding)
        x_s = self.conv(x_s.view(x.size(0), 1, -1))
        x_s = x_s.view(x.size(0), -1, 1, 1).sigmoid()
        return x * x_s.expand_as(x)


class MyAttn(nn.Module):
    """Idea from Attentional Feature Fusion (MS-CAM)
    Combines global and local attention in a better manner than scSE
    """

    def __init__(self, in_ch, reduced_ch, norm_layer=ABN, norm_act="relu"):  # parse additional args for compatability
        super().__init__()
        self.global_attn = nn.Sequential(
            FastGlobalAvgPool2d(),
            conv1x1(in_ch, reduced_ch),
            norm_layer(reduced_ch, activation=norm_act),
            conv1x1(reduced_ch, in_ch),
            norm_layer(in_ch, activation="identity"),  # no last activation
        )
        # this
        self.local_attn = nn.Sequential(
            conv1x1(in_ch, reduced_ch),
            norm_layer(reduced_ch, activation=norm_act),
            conv1x1(reduced_ch, in_ch),
            norm_layer(in_ch, activation="identity"),  # no last activation
        )

    def forward(self, x):
        xl = self.local_attn(x)
        xg = self.global_attn(x)
        return x * (xl + xg).sigmoid()


def get_attn(attn_type):
    """Get attention by name
    Args:
        attn_type (Uniont[str, None]): Attention type. Supported:
            `se` - Squeeze and Excitation
            `eca` - Efficient Channel Attention
            `sse` - Spatial Excitation
            `scse` - Spatial and Channel ‘Squeeze & Excitation’
            None - no attention
    """
    ATT_TO_MODULE = {
        "se": SEModule,
        "eca": ECAModule,
        "eca": ECAModule,
        "eca9": partial(ECAModule, kernel_size=9),
        "sse": SSEModule,
        "scse": SCSEModule,
        "se-var3": SEVar3,
        "ms-cam": MSCAMModule,
        "fca": FCAAttn,
        "fca-eca": FCA_ECA_Attn,
        "xca": XCA,
        "esa": ESA,
    }

    if attn_type is None:
        return nn.Identity
    else:
        return ATT_TO_MODULE[attn_type.lower()]


class DepthwiseSeparableConv(nn.Sequential):
    """Depthwise separable conv with BN after depthwise & pointwise."""

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, norm_layer=ABN, norm_act="relu", use_norm=True):
        modules = [
            conv3x3(in_channels, in_channels, stride=stride, groups=in_channels, dilation=dilation),
            # Do we need normalization here? If yes why? If no why?
            # bias is needed for EffDet because in head conv is separated from normalization
            conv1x1(in_channels, out_channels, bias=not use_norm),
            norm_layer(out_channels, activation=norm_act) if use_norm else nn.Identity(),
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
        attn_type=None,
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
        self.se = get_attn(attn_type)(mid_chs, in_channels // 4, norm_act)
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
        random_tensor = self.keep_prob
        random_tensor += torch.rand((x.size(0),) + (1,) * (x.dim() - 1), dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        output = x / self.keep_prob * binary_tensor
        return output

    def extra_repr(self):
        return f"keep_prob={self.keep_prob:.2f}"


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
        attn_type=None,
        dilation=1,
        norm_layer=ABN,
        norm_act="relu",
        antialias=False,
        keep_prob=1,
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
        self.se_module = get_attn(attn_type)(outplanes, planes // 4)
        self.final_act = activation_from_name(norm_act)
        self.downsample = downsample
        self.blurpool = BlurPool(channels=planes) if antialias else nn.Identity()
        self.antialias = antialias
        self.drop_connect = DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()

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
        # avoid 2 inplace ops by chaining into one long op. Needed for inplaceabn
        out = self.drop_connect(self.se_module(self.bn2(out))) + residual
        return self.final_act(out)


# This class is from torchvision with many (many) modifications
# it's not very intuitive. Check this article if you want to understand the code more
# https://medium.com/@erikgaas/resnet-torchvision-bottlenecks-and-layers-not-as-they-seem-145620f93096
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
        attn_type=None,
        dilation=1,
        norm_layer=ABN,
        norm_act="relu",
        antialias=False,
        keep_prob=1,  # for drop connect
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
        self.se_module = get_attn(attn_type)(outplanes, planes // 4)
        self.final_act = activation_from_name(norm_act)
        self.downsample = downsample
        self.blurpool = BlurPool(channels=width) if antialias else nn.Identity()
        self.antialias = antialias
        self.drop_connect = DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()

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
        out = self.drop_connect(self.se_module(self.bn3(out))) + residual
        return self.final_act(out)


# TResnet models use slightly modified versions of BasicBlock and Bottleneck
# need to adjust for it
class TBasicBlock(BasicBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_act = nn.ReLU(inplace=True)
        self.bn1.activation_param = 1e-3  # needed for loading weights
        if not kwargs.get("attn_type") == "se":
            return
        planes = kwargs["planes"]
        self.se_module = SEModule(planes, max(planes // 4, 64))


class TBottleneck(Bottleneck):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_act = nn.ReLU(inplace=True)
        self.bn1.activation_param = 1e-3  # needed for loading weights
        self.bn2.activation_param = 1e-3
        if not kwargs.get("attn_type") == "se":
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

        out = self.se_module(out)

        out = self.conv3(out)
        # avoid 2 inplace ops by chaining into one long op
        out = self.drop_connect(self.bn3(out)) + residual
        return self.final_act(out)


## DarkNet blocks
class DarkBasicBlock(nn.Module):
    """Basic Block for DarkNet family models"""

    def __init__(
        self,
        in_channels,
        out_channels,
        bottle_ratio=0.5,
        attn_type=None,
        norm_layer=ABN,
        norm_act="leaky_relu",
        keep_prob=1,
    ):
        super().__init__()
        mid_channels = int(in_channels * bottle_ratio)
        self.bn1 = norm_layer(mid_channels, activation=norm_act)
        self.conv1 = conv1x1(in_channels, mid_channels)
        self.bn2 = norm_layer(out_channels, activation=norm_act)
        self.conv2 = conv3x3(mid_channels, out_channels, groups=32)
        # In original DarkNet they have activation after second BN but the most recent papers
        # (Mobilenet v2 for example) show that it is better to use linear here
        # out_channels // 4 is for SE attention. other attentions don't use second parameter
        self.attention = get_attn(attn_type)(out_channels, out_channels // 4)
        self.drop_connect = DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()

    def forward(self, x):
        # preAct
        out = self.bn1(x)
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.conv2(out)
        # out = self.bn3(out)
        # out = self.conv3(out)
        out = self.drop_connect(self.attention(out)) + x
        return out


class CSPDarkBasicBlock(nn.Module):
    """Idea from https://github.com/WongKinYiu/CrossStagePartialNetworks
    But implementaion is different. This block divides input into two passes only one part through bottleneck
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        attn_type=None,
        norm_layer=ABN,
        norm_act="leaky_relu",
        keep_prob=1,
    ):
        super().__init__()
        mid_channels = int(in_channels * bottle_ratio)
        self.conv1 = conv1x1(in_channels, mid_channels)
        self.bn1 = norm_layer(mid_channels, activation=norm_act)
        self.conv2 = conv3x3(mid_channels, out_channels)
        # In original DarkNet they have activation after second BN but the most recent papers
        # (Mobilenet v2 for example) show that it is better to use linear here
        self.bn2 = norm_layer(out_channels, activation="identity")
        # out_channels // 4 is for SE attention. other attentions don't use second parameter
        self.attention = get_attn(attn_type)(out_channels, out_channels // 4)
        self.drop_connect = DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()

    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        # avoid 2 inplace ops by chaining into one long op. Needed for inplaceabn
        out = self.drop_connect(self.attention(self.bn2(out))) + x
        return out


class SimpleBottleneck(nn.Module):
    """Simple Bottleneck without downsample support"""

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        stride=1,
        # attn_type=None,
        groups=1,
        groups_width=None,
        no_groups_with_stride=False,
        norm_layer=ABN,
        norm_act="relu",
        keep_prob=1,  # for drop connect
        final_act=False,  # add activation after summation with residual
    ):
        super().__init__()
        groups = mid_chs // groups_width if groups_width else groups
        if no_groups_with_stride and stride == 2:
            groups = 1  # no groups in first block in stage. helps to avoid representational bottleneck
        self.conv1 = conv1x1(in_chs, mid_chs)
        self.bn1 = norm_layer(mid_chs, activation=norm_act)
        self.conv2 = conv3x3(mid_chs, mid_chs, stride=stride, groups=groups)
        self.bn2 = norm_layer(mid_chs, activation=norm_act)
        self.conv3 = conv1x1(mid_chs, out_chs)
        self.bn3 = norm_layer(out_chs, activation="identity")
        self.has_residual = in_chs == out_chs and stride == 1
        self.final_act = activation_from_name(norm_act) if final_act else nn.Identity()
        # self.se_module = get_attn(attn_type)(outplanes, planes // 4)
        # self.drop_connect = DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        # avoid 2 inplace ops by chaining into one long op
        if self.has_residual:
            out = self.bn3(out) + x
        else:
            out = self.bn3(out)
        out = self.final_act(out)  # optional last activation
        return out


class SimpleBasicBlock(nn.Module):
    """Simple Bottleneck without downsample support"""

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        stride=1,
        # attn_type=None,
        groups=1,
        groups_width=None,
        norm_layer=ABN,
        norm_act="relu",
        keep_prob=1,  # for drop connect
        dim_reduction="stride -> expand",  # "expand -> stride", "stride & expand"
        final_act=False,  # add activation after summation with residual
    ):
        super().__init__()
        groups = in_chs // groups_width if groups_width else groups
        if dim_reduction == "expand -> stride":
            self.conv1 = conv3x3(in_chs, mid_chs)
            self.bn1 = norm_layer(mid_chs, activation=norm_act)
            self.conv2 = conv3x3(mid_chs, out_chs, stride=stride)
        elif dim_reduction == "stride -> expand":
            # it's ~20% faster to have stride first. maybe accuracy drop isn't that big
            # TODO: test MixConv type of block here. I expect it to have the same speed and N params
            # while performance should increase
            self.conv1 = conv3x3(in_chs, in_chs, stride=stride)
            self.bn1 = norm_layer(in_chs, activation=norm_act)
            self.conv2 = conv3x3(in_chs, out_chs)
        elif dim_reduction == "stride & expand":
            self.conv1 = conv3x3(in_chs, mid_chs, stride=stride)
            self.bn1 = norm_layer(mid_chs, activation=norm_act)
            self.conv2 = conv3x3(out_chs, out_chs)
        else:
            raise ValueError(f"{dim_reduction} is not valid dim reduction in BasicBlock")

        self.bn3 = norm_layer(out_chs, activation="identity")
        self.has_residual = in_chs == out_chs and stride == 1
        self.final_act = activation_from_name(norm_act) if final_act else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        # avoid 2 inplace ops by chaining into one long op
        if self.has_residual:
            out = self.bn3(out) + x
        else:
            out = self.bn3(out)
        out = self.final_act(out)  # optional last activation
        return out


class SimplePreActBottleneck(nn.Module):
    """Simple Bottleneck with preactivation"""

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        stride=1,
        groups=1,
        groups_width=None,
        norm_layer=ABN,
        norm_act="relu",
        force_residual=False,  # force residual in stride=2 blocks
        # keep_prob=1,  # for drop connect
    ):
        super().__init__()
        groups = mid_chs // groups_width if groups_width else groups
        self.bn1 = norm_layer(in_chs, activation=norm_act)
        self.conv1 = conv1x1(in_chs, mid_chs)
        self.bn2 = norm_layer(mid_chs, activation=norm_act)
        self.conv2 = conv3x3(mid_chs, mid_chs, stride=stride, groups=groups)
        self.bn3 = norm_layer(mid_chs, activation=norm_act)
        # last conv is not followed by bn, but anyway bias here makes it slightly worse (on Imagenet)
        self.conv3 = conv1x1(mid_chs, out_chs)
        self.has_residual = in_chs == out_chs and stride == 1
        self.force_residual = force_residual
        if force_residual:
            self.blurpool = BlurPool(channels=in_chs)
            self.in_chs = in_chs

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.conv3(out)
        if self.has_residual:
            out += x
        elif self.force_residual:  # forces partial residual for stride=2 block
            out[:, : self.in_chs] += self.blurpool(x)
        return out


class MixConv(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        in_chs_4 = in_chs // 4
        self.conv1x1 = nn.Sequential(
            pt.modules.BlurPool(in_chs // 4), pt.modules.residual.conv1x1(in_chs // 4, out_chs // 4)
        )
        self.conv3x3 = nn.Conv2d(in_chs // 4, out_chs // 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(in_chs // 4, out_chs // 4, kernel_size=5, stride=2, padding=2, bias=False)
        self.conv7x7 = nn.Conv2d(in_chs // 4, out_chs // 4, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x_0, x_1, x_2, x_3 = x.chunk(4, dim=1)
        return torch.cat([self.conv1x1(x_0), self.conv3x3(x_1), self.conv5x5(x_2), self.conv7x7(x_3)], dim=1)


class SimplePreActBasicBlock(nn.Module):
    """Simple BasicBlock with preactivatoin & without downsample support"""

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        stride=1,
        groups=1,
        groups_width=None,
        norm_layer=ABN,
        norm_act="relu",
        keep_prob=1,  # for drop connect
        dim_reduction="stride & expand",  # "expand -> stride", "stride & expand"
        force_residual=False,  # always have residual
    ):
        super().__init__()
        self.has_residual = in_chs == out_chs and stride == 1
        self.force_residual = force_residual
        if force_residual:
            self.blurpool = BlurPool(channels=in_chs)
            self.in_chs = in_chs
        groups = in_chs // groups_width if groups_width else groups
        if dim_reduction == "expand -> stride":
            self.bn1 = norm_layer(in_chs, activation=norm_act)
            self.conv1 = conv3x3(in_chs, mid_chs)
            self.bn2 = norm_layer(mid_chs, activation=norm_act)
            self.conv2 = conv3x3(mid_chs, out_chs, stride=stride)
        elif dim_reduction == "s2d":
            if stride == 2:
                # BN before S2D to make sure different pixels from one channel are normalized the same way
                self.bn1 = nn.Sequential(norm_layer(in_chs, activation=norm_act), SpaceToDepth(block_size=2))
                self.conv1 = conv3x3(in_chs * 4, mid_chs)
                self.bn2 = norm_layer(mid_chs, activation=norm_act)
                self.conv2 = conv3x3(mid_chs, out_chs)
            else:  # same as stride & expand
                self.bn1 = norm_layer(in_chs, activation=norm_act)
                self.conv1 = conv3x3(in_chs, mid_chs, stride=stride)
                self.bn2 = norm_layer(mid_chs, activation=norm_act)
                self.conv2 = conv3x3(mid_chs, out_chs)
        # elif dim_reduction == "stride -> expand":
        #     # it's ~20% faster to have stride first. maybe accuracy drop isn't that big
        #     # TODO: test MixConv type of block here. I expect it to have the same speed and N params
        #     # while performance should increase
        #     self.conv1 = conv3x3(in_chs, in_chs, stride=stride)
        #     self.bn1 = norm_layer(in_chs, activation=norm_act)
        #     self.conv2 = conv3x3(in_chs, out_chs)
        elif dim_reduction == "stride & expand":  # only this one is supported for now
            self.bn1 = norm_layer(in_chs, activation=norm_act)
            self.conv1 = conv3x3(in_chs, mid_chs, stride=stride)
            self.bn2 = norm_layer(mid_chs, activation=norm_act)
            self.conv2 = conv3x3(mid_chs, out_chs)
        elif dim_reduction == "mixconv stride & expand":
            self.bn1 = norm_layer(in_chs, activation=norm_act)
            self.conv1 = conv3x3(in_chs, mid_chs, stride=stride)
            self.bn2 = norm_layer(mid_chs, activation=norm_act)
            self.conv2 = conv3x3(mid_chs, out_chs)
        else:
            raise ValueError(f"{dim_reduction} is not valid dim reduction in PreAct BasicBlock")

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(out)
        # avoid 2 inplace ops by chaining into one long op
        if self.has_residual:
            out += x
        elif self.force_residual:  # forces partial residual for stride=2 block
            out[:, : self.in_chs] += self.blurpool(x)
        return out


class SimplePreActRes2BasicBlock(nn.Module):
    """Building block based on BasicBlock with:
    preactivatoin
    without downsample support
    with Res2Net inspired chunking
    """

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        stride=1,
        norm_layer=ABN,
        norm_act="relu",
        keep_prob=1,  # for drop connect
        antialias=False,
    ):
        super().__init__()
        self.has_residual = in_chs == out_chs and stride == 1
        self.stride = stride
        if self.stride == 2:  # only use Res2Net for stride == 1
            self.blocks = nn.Sequential(
                norm_layer(in_chs, activation=norm_act),
                conv3x3(in_chs, mid_chs, stride=1 if antialias else 2),
                BlurPool(channels=mid_chs) if antialias else nn.Identity(),
                norm_layer(mid_chs, activation=norm_act),
                conv3x3(mid_chs, out_chs),
            )
        else:
            self.bn1 = norm_layer(in_chs, activation=norm_act)
            self.block_1 = nn.Sequential(
                conv3x3(in_chs // 4, in_chs // 4),
                norm_layer(in_chs // 4, activation=norm_act),
            )
            self.block_2 = nn.Sequential(
                conv3x3(in_chs // 4, in_chs // 4),
                norm_layer(in_chs // 4, activation=norm_act),
            )
            self.block_3 = nn.Sequential(
                conv3x3(in_chs // 4, in_chs // 4),
                norm_layer(in_chs // 4, activation=norm_act),
            )
            self.last_conv = conv3x3(in_chs, out_chs)  # expand in last conv in block

    def forward(self, x):
        if self.stride == 2:
            return self.blocks(x)
        # split in 4
        x_out0, x_inp1, x_inp2, x_inp3 = self.bn1(x).chunk(4, dim=1)
        x_out1 = self.block_1(x_inp1)
        x_out2 = self.block_2(x_inp2 + x_out1)
        x_out3 = self.block_3(x_inp3 + x_out2)
        out = torch.cat([x_out0, x_out1, x_out2, x_out3], dim=1)
        out = self.last_conv(out)  # always has residual
        if self.has_residual:
            out += x
        return out


class SimpleInvertedResidual(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        attn_type=None,
        keep_prob=1,  # drop connect param
        norm_layer=ABN,
        norm_act="relu",
        final_act=False,  # add activation after summation with residual
    ):
        super().__init__()
        self.has_residual = in_chs == out_chs and stride == 1
        if in_chs != mid_chs:
            self.expansion = nn.Sequential(conv1x1(in_chs, mid_chs), norm_layer(mid_chs, activation=norm_act))
        else:
            self.expansion = nn.Identity()
        self.conv_dw = nn.Conv2d(
            mid_chs,
            mid_chs,
            dw_kernel_size,
            stride=stride,
            groups=mid_chs,
            bias=False,
            padding=dw_kernel_size // 2,
        )
        self.bn2 = norm_layer(mid_chs, activation=norm_act)
        # some models like MobileNet use mid_chs here instead of in_channels. But I don't care for now
        self.se = get_attn(attn_type)(mid_chs, in_chs // 4, norm_act)
        self.conv_pw1 = conv1x1(mid_chs, out_chs)
        self.bn3 = norm_layer(out_chs, activation="identity")
        self.drop_connect = DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()
        self.final_act = activation_from_name(norm_act) if final_act else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.expansion(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pw1(x)
        x = self.bn3(x)

        if self.has_residual:
            x = self.drop_connect(x) + residual
        x = self.final_act(x)  # optional last activation
        return x


class SimplePreActInvertedResidual(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        dw_str2_kernel_size=3,
        stride=1,
        attn_type=None,
        keep_prob=1,  # drop connect param
        norm_layer=ABN,
        norm_act="relu",
        force_residual=False,
        force_expansion=False,  # always have expansion
    ):
        super().__init__()
        self.has_residual = in_chs == out_chs and stride == 1
        self.force_residual = force_residual
        if force_residual:
            self.blurpool = BlurPool(channels=in_chs) if stride == 2 else nn.Identity()
            self.in_chs = in_chs
        if in_chs != mid_chs or force_expansion:
            self.expansion = nn.Sequential(norm_layer(in_chs, activation=norm_act), conv1x1(in_chs, mid_chs))
        else:
            self.expansion = nn.Identity()
        self.bn2 = norm_layer(mid_chs, activation=norm_act)
        dw_kernel_size = dw_str2_kernel_size if stride == 2 else dw_kernel_size
        self.conv_dw = nn.Conv2d(
            mid_chs,
            mid_chs,
            dw_kernel_size,
            stride=stride,
            groups=mid_chs,
            bias=False,
            padding=dw_kernel_size // 2,
        )
        # some models like MobileNet use mid_chs here instead of in_channels. But I don't care for now
        self.se = get_attn(attn_type)(mid_chs, in_chs // 4, norm_act)
        self.bn3 = norm_layer(mid_chs, activation=norm_act)  # Note it's NOT identity for PreAct
        self.conv_pw1 = conv1x1(mid_chs, out_chs)
        self.drop_connect = DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()

    def forward(self, x):
        out = self.expansion(x)
        out = self.bn2(out)
        out = self.conv_dw(out)
        # x = self.se(x)
        out = self.bn3(out)
        out = self.conv_pw1(out)
        if self.has_residual:
            out = self.drop_connect(out) + x
        elif self.force_residual:  # forces partial residual for stride=2 block
            out[:, : self.in_chs] += self.blurpool(x)
        return out


class PreBlock_2(nn.Module):
    """
    pw -> pw -> dw
    Always has at least partial residual
    """

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        dw_str2_kernel_size=3,
        stride=1,
        attn_type=None,
        keep_prob=1,  # drop connect param
        norm_layer=ABN,
        norm_act="relu",
    ):

        super().__init__()
        self.blurpool = BlurPool(channels=in_chs) if stride == 2 else nn.Identity()
        self.in_chs = in_chs

        self.pw1 = nn.Sequential(norm_layer(in_chs, activation=norm_act), conv1x1(in_chs, mid_chs))
        self.pw2 = nn.Sequential(norm_layer(mid_chs, activation=norm_act), conv1x1(mid_chs, out_chs))

        dw_kernel_size = dw_str2_kernel_size if stride == 2 else dw_kernel_size
        conv_dw = nn.Conv2d(
            out_chs,
            out_chs,
            dw_kernel_size,
            stride=stride,
            groups=out_chs,
            bias=False,
            padding=dw_kernel_size // 2,
        )
        self.dw = nn.Sequential(norm_layer(out_chs, activation=norm_act), conv_dw)

        self.se = get_attn(attn_type)(mid_chs, in_chs // 4, norm_act)
        self.drop_connect = DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()

    def forward(self, x):
        out = self.pw1(x)
        out = self.pw2(out)
        out = self.dw(out)
        out[:, : self.in_chs] += self.blurpool(x)
        return out


class SimpleSeparable_2(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        attn_type=None,
        keep_prob=1,  # drop connect param
        norm_layer=ABN,
        norm_act="relu",
    ):
        super().__init__()
        # actially we can have parial residual even when in_chs != out_chs
        self.has_residual = in_chs == out_chs and stride == 1
        self.sep_convs = nn.Sequential(
            DepthwiseSeparableConv(in_chs, out_chs, stride=stride, norm_layer=norm_layer, norm_act=norm_act),
            DepthwiseSeparableConv(out_chs, out_chs, norm_layer=norm_layer, norm_act="identity"),
        )
        self.drop_connect = DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()

    def forward(self, x):
        # x = self.se(x) # maybe attention at the beginning would work better?
        # the idea is: it would allow to accentuate what features to process in this block
        out = self.sep_convs(x)
        if self.has_residual:
            out = self.drop_connect(out) + x
        return out


class SimplePreActSeparable_2(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        attn_type=None,
        keep_prob=1,  # drop connect param
        norm_layer=ABN,
        norm_act="relu",
        dim_reduction=None,
    ):
        super().__init__()
        # actially we can have parial residual even when in_chs != out_chs
        self.has_residual = in_chs == out_chs and stride == 1
        if dim_reduction == "s2d_full" and stride == 2:  # redesign reduction
            self.blocks = nn.Sequential(
                # replace first DW -> PW with s2d -> full conv to lose less information
                # gives very large increase in number of parameters
                norm_layer(in_chs, activation=norm_act),
                SpaceToDepth(block_size=2),
                conv3x3(in_chs * 4, out_chs),
                norm_layer(out_chs, activation=norm_act),
                conv3x3(mid_chs, mid_chs, groups=mid_chs),  # DW 2
                norm_layer(mid_chs, activation=norm_act),
                conv1x1(mid_chs, out_chs),  # PW 2
            )
        elif dim_reduction == "s2d_dw" and stride == 2:  # redesign reduction
            self.blocks = nn.Sequential(
                # BN before S2D to make sure different pixels from one channel are normalized the same way
                # expand with s2d -> DW -> PW
                norm_layer(in_chs, activation=norm_act),
                SpaceToDepth(block_size=2),
                conv3x3(in_chs * 4, in_chs * 4, groups=in_chs * 4),  # DW 1
                norm_layer(in_chs * 4, activation=norm_act),
                conv1x1(in_chs * 4, out_chs),  # PW 1
                norm_layer(mid_chs, activation=norm_act),
                conv3x3(mid_chs, mid_chs, groups=mid_chs),  # DW 2
                norm_layer(mid_chs, activation=norm_act),
                conv1x1(mid_chs, out_chs),  # PW 2
            )
        else:
            self.blocks = nn.Sequential(
                norm_layer(in_chs, activation=norm_act),
                conv3x3(in_chs, in_chs, stride=stride, groups=in_chs),  # DW 1
                norm_layer(in_chs, activation=norm_act),
                conv1x1(in_chs, mid_chs),  # PW 1
                norm_layer(mid_chs, activation=norm_act),
                conv3x3(mid_chs, mid_chs, groups=mid_chs),  # DW 2
                norm_layer(mid_chs, activation=norm_act),
                conv1x1(mid_chs, out_chs),  # PW 2
            )
        self.drop_connect = DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()

    def forward(self, x):
        # x = self.se(x) # maybe attention at the beginning would work better?
        # the idea is: it would allow to accentuate what features to process in this block
        out = self.blocks(x)
        if self.has_residual:
            out = self.drop_connect(out) + x
        return out


class SimpleSeparable_3(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        attn_type=None,
        keep_prob=1,  # drop connect param
        norm_layer=ABN,
        norm_act="relu",
    ):
        super().__init__()
        # actially we can have parial residual even when in_chs != out_chs
        self.has_residual = in_chs == out_chs and stride == 1
        self.sep_convs = nn.Sequential(
            DepthwiseSeparableConv(in_chs, out_chs, stride=stride, norm_layer=norm_layer, norm_act=norm_act),
            DepthwiseSeparableConv(out_chs, out_chs, norm_layer=norm_layer, norm_act=norm_act),
            DepthwiseSeparableConv(out_chs, out_chs, norm_layer=norm_layer, norm_act="identity"),
        )
        self.drop_connect = DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()

    def forward(self, x):
        # x = self.se(x) # maybe attention at the beginning would work better?
        # the idea is: it would allow to accentuate what features to process in this block
        out = self.sep_convs(x)
        if self.has_residual:
            out = self.drop_connect(out) + x
        return out


class SimpleStage(nn.Module):
    """One stage in DarkNet models. It consists of first transition conv (with stride == 2) and
    DarkBasicBlock repeated num_blocks times
    Args:
        in_channels (int): input channels for this stage
        out_channels (int): output channels for this stage
        num_blocks (int): number of residual blocks in stage
        stride (int): stride for first convolution
        bottle_ratio (float): how much channels are reduced inside blocks
        antialias (bool): flag to apply gaussiian smoothing before conv with stride 2

    Ref: TODO: add

    """

    def __init__(
        self,
        in_chs,
        out_chs,
        num_blocks,
        stride=2,
        bottle_ratio=1.0,
        # antialias=False,
        block_fn=DarkBasicBlock,
        attn_type=None,
        norm_layer=ABN,
        norm_act="leaky_relu",
        keep_prob=1,
        csp_block_ratio=None,  # for compatability
        x2_transition=None,  # for compatability
        filter_steps=0,
        **block_kwargs,
    ):
        super().__init__()
        if csp_block_ratio is not None:
            print("Passing csp block ratio to Simple Stage")
        norm_kwarg = dict(norm_layer=norm_layer, norm_act=norm_act, **block_kwargs)  # this is dirty
        mid_chs = max(int(out_chs * bottle_ratio), 64)
        layers = [block_fn(in_chs=in_chs, mid_chs=mid_chs, out_chs=out_chs, stride=stride, **norm_kwarg)]
        block_kwargs = dict(
            in_chs=out_chs, mid_chs=out_chs + filter_steps, out_chs=out_chs + filter_steps, **norm_kwarg
        )
        for _ in range(num_blocks - 1):
            layers.append(block_fn(**block_kwargs))
            block_kwargs["in_chs"] += filter_steps
            block_kwargs["mid_chs"] += filter_steps
            block_kwargs["out_chs"] += filter_steps
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


class CrossStage(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        num_blocks,
        stride=2,
        bottle_ratio=0.5,
        antialias=False,
        block_fn=SimpleBottleneck,
        attn_type=None,
        norm_layer=ABN,
        norm_act="leaky_relu",
        keep_prob=1,
        csp_block_ratio=0.5,  # how many channels go to blocks
        x2_transition=True,
        **block_kwargs,
    ):
        super().__init__()
        extra_kwarg = dict(norm_layer=norm_layer, norm_act=norm_act, **block_kwargs)
        self.first_layer = block_fn(in_chs=in_chs, mid_chs=out_chs, out_chs=out_chs, stride=stride, **extra_kwarg)
        block_chs = int(csp_block_ratio * out_chs)  # todo: maybe change to make divizable or hardcode values
        extra_kwarg.update(in_chs=block_chs, mid_chs=block_chs, out_chs=block_chs)
        self.blocks = nn.Sequential(*[block_fn(**extra_kwarg) for _ in range(num_blocks - 1)])
        # using identity activation in transition conv. the idea is the same as in Linear Bottleneck
        # maybe need to test this design choice later. maybe I can simply remove this transition
        self.x2_transition = (
            nn.Sequential(conv1x1(block_chs, block_chs), norm_layer(block_chs, activation="identity"))
            if x2_transition
            else nn.Identity()
        )
        self.csp_block_ratio = csp_block_ratio

    def forward(self, x):
        x = self.first_layer(x)
        if self.csp_block_ratio == 0.5:
            x1, x2 = torch.chunk(x, chunks=2, dim=1)
        elif self.csp_block_ratio == 0.75:
            x1, x2, x3, x4 = torch.chunk(x, chunks=4, dim=1)
            x2 = torch.cat([x2, x3, x4], dim=1)
        x2 = self.blocks(x2)
        x2 = self.x2_transition(x2)
        out = torch.cat([x1, x2], dim=1)
        # no explicit transition here. first conv in the next stage would perform transition
        return out


class RepVGGBlock(nn.Module):
    """
    This block is inspired by [1] and [2]. The idea is to have `n_heads` parallel branches of convolutions which are then summed + residual
    For performance they are implemented as one conv3x3. I removed normalization by making sure this block is variance preserving
    RepVGGBlock(act=Identity)( N(0, 1)) ~= N(0, 1)

    NOTE: this block requires correct initialization of conv weighs to work. Highly recommended to use with ConvWS
    NOTE: for some reasons SELU activation still causes increase in variance. The possible solution is to use gain = 0.5 in ConvWS instead of 1

    For inference this block could be replaced with `FusedRepVGGBlock` for significant speed-up

    Args:
        in_chs (int):
            number of input channels
        out_chs (int):
            number of output channels. if out_chs > in_chs performs partial residual
        n_heads (int):
            number of parallel branches
        act (nn.Module):
            activation to use. recommended is nn.SELU
        alpha (float):
            parameter for skip init, balancing signal between block and residual path. output would be
            x * (1 - alpha) + sum[conv3x3_i(x) * alpha], for i in range(n_heads)
        trainable_alpha (bool):
            if True make alpha a parameter, else it's un-trainable buffer
            having trainable alpha may be beneficial. Timm repo says it could bring noticeable
            slowdown for training. need to investigate

    Ref:
        [1] RepVGG: Making VGG-style ConvNets Great Again
        [2] High-Performance Large-Scale Image Recognition Without Normalization
    """

    def __init__(self, in_chs, out_chs, n_heads=2, act=nn.SELU, alpha=0.2, trainable_alpha=False):
        super().__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.n_heads = n_heads

        assert 0 <= alpha <= 1

        self.conv = nn.Conv2d(in_chs, out_chs * n_heads, kernel_size=3, padding=1)

        # it's important to carefully initialize alpha so that this block is variance preserving
        branch_alpha = torch.ones(1, out_chs, 1, 1) * (alpha / n_heads) ** 0.5
        # take care of extra features without residual (they appear if out_chs > in_chs)
        branch_alpha[:, in_chs:] = (1 / n_heads) ** 0.5
        residual_alpha = torch.tensor((1 - alpha) ** 0.5)

        if trainable_alpha:
            self.register_parameter("skipinit_res", nn.Parameter(residual_alpha))
            self.register_parameter("skipinit_branch", nn.Parameter(branch_alpha))
        else:
            self.register_buffer("skipinit_res", residual_alpha)
            self.register_buffer("skipinit_branch", branch_alpha)

        self.activation = act()

    def forward(self, x):
        proj = self.conv(x)
        # same as out = reduce(lambda x, y: torch.chunk(out, self.n_heads, 1)) but faster
        proj = proj.view(proj.size(0), self.n_heads, proj.size(1) // self.n_heads, proj.size(2), proj.size(3))
        branch_res = proj.sum(1)
        res = x * self.skipinit_res
        if self.in_chs == self.out_chs:
            branch_res = branch_res * self.skipinit_branch
            branch_res += res
        else:
            branch_res = branch_res * self.skipinit_branch
            branch_res[:, : self.in_chs] += res
        out = self.activation(branch_res)
        return out

    def extra_repr(self):
        return f"num_heads={self.n_heads}"


class FusedRepVGGBlock(nn.Sequential):
    def __init__(self, in_chs, out_chs, act=nn.SELU):
        super().__init__()
        self.add_module("conv", nn.Conv2d(in_chs, out_chs, kernel_size=3, padding=1))
        self.add_module("activation", act())

    def load_state_dict(self, state_dict):
        if state_dict.get("skipinit_res", None) is None:
            super().load_state_dict(state_dict)

        self_w = self.conv.weight
        self_b = self.conv.bias

        kernel_id = torch.zeros_like(self_w)
        kernel_id[: self_w.size(1), :, 1, 1] = torch.eye(self_w.size(1))
        kernel_id *= state_dict["skipinit_res"]

        kernel_conv = state_dict["conv.weight"].view(-1, *self_w.shape).sum(0)
        bias_conv = state_dict["conv.bias"].view(-1, *self.conv.bias.shape).sum(0)

        kernel_conv *= state_dict["skipinit_branch"].view(self_w.size(0), 1, 1, 1)
        bias_conv *= state_dict["skipinit_branch"].flatten()

        self.conv.weight.data = kernel_id + kernel_conv
        self.conv.bias.data = bias_conv


# modules from XCiT paper
# XCA_Token is from timm, XCA is the same (convered by tests) but works direcly on B x C x H x W tensor
class XCA_Token(nn.Module):
    """Cross-Covariance Attention (XCA)
    Operation where the channels are updated using a weighted sum. The weights are obtained from the (softmax
    normalized) Cross-covariance matrix (Q^T \\cdot K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # Result of next line is (qkv, B, num (H)eads,  (C')hannels per head, N)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # Paper section 3.2 l2-Normalization and temperature scaling
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, H, C', N), permute -> (B, N, H, C')
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class XCA(nn.Module):
    """Cross-Covariance Attention (XCA)
    Operation where the channels are updated using a weighted sum. The weights are obtained from the (softmax
    normalized) Cross-covariance matrix (Q^T \\cdot K \\in d_h \\times d_h)
    This could be viewed as dynamic 1x1 convolution
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = conv1x1(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = conv1x1(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        # C` == channels per head, Hd == num heads
        # B x C x H x W -> B x 3*C x H x W -> B x 3 x Hd x C` x H*W -> 3 x B x Hd x C` x H*W
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, -1).transpose(0, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy

        # Paper section 3.2 l2-Normalization and temperature scaling
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # -> B x Hd x C` x C`
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # B x Hd x C` x C` @ B x Hd x C` x H*W -> B x C x H x W
        x_out = (attn @ v).reshape(B, C, H, W)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        # in original paper there is no residual here
        return x + x_out

    def load_state_dict(self, state_dict):
        # to allow loading from Linear layer
        new_sd = {}
        for k, v in state_dict.items():
            if v.dim() == 2:
                new_sd[k] = v[..., None, None]
            else:
                new_sd[k] = v
        super().load_state_dict(new_sd)


class ESA(nn.Module):
    """
    Efficient self-attention. Performs self-attention on channels after GAP to significantly reduce ammount of required compute
    Close to SE-Var3 from ECA paper and XCA above

    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, use_proj=True):
        super().__init__()
        self.num_heads = num_heads
        self.pool = FastGlobalAvgPool2d()
        self.qkv = conv1x1(dim, dim * 3, bias=qkv_bias)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.proj = conv1x1(dim, dim, bias=True) if use_proj else nn.Identity()

    def forward(self, x):
        B, C = x.shape[:2]
        # C` == channels per head, Hd == num heads
        # B x C x H x W -> B x 3*C x 1 x 1 -> B x 3 x Hd x C` x 1 -> 3 x B x Hd x C` x 1
        qkv = self.qkv(self.pool(x)).reshape(B, 3, self.num_heads, C // self.num_heads, -1).transpose(0, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy

        # Paper section 3.2 l2-Normalization and temperature scaling. BUT! scaling is for channels because we don't have enough tokens
        q = torch.nn.functional.normalize(q, dim=-2)
        k = torch.nn.functional.normalize(k, dim=-2)
        # -> B x Hd x C` x C`
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # B x Hd x C` x C` @ B x Hd x C` x 1 -> B x C x 1
        x_out = (attn @ v).reshape(B, C, 1, 1)
        x_out = self.proj(x_out)
        return x + x_out.expand_as(x)
