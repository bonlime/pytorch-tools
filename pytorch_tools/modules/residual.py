import torch
import torch.nn as nn
from inplace_abn import ABN
from pytorch_tools.modules import BlurPool, GlobalPool2d
from pytorch_tools.utils.misc import activation_from_name


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SEModule(nn.Module):
    def __init__(self, channels, reduction_channels):
        super(SEModule, self).__init__()
        self.pool = GlobalPool2d('avg')
        # authors of original paper DO use bias
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x_se = self.pool(x)
        x_se = self.fc1(x_se)
        x_se = self.relu(x_se)
        x_se = self.fc2(x_se)
        return x * x_se.sigmoid()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, 
                 stride=1, downsample=None,
                 groups=1, base_width=64, 
                 use_se=False,
                 dilation=1,
                 norm_layer=ABN,
                 norm_act='relu',
                 antialias=False):
        super(BasicBlock, self).__init__()
        antialias = antialias and stride == 2
        assert groups == 1, 'BasicBlock only supports groups of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'
        outplanes = planes * self.expansion
        conv1_stride = 1 if antialias else stride
        self.conv1 = conv3x3(inplanes, planes, conv1_stride, dilation)
        self.bn1 = norm_layer(planes, activation=norm_act)
        self.conv2 = conv3x3(planes, outplanes)
        self.bn2 = norm_layer(outplanes, activation='identity')
        self.se_module = SEModule(outplanes, planes // 4) if use_se else None
        self.final_act = activation_from_name(norm_act)
        self.downsample = downsample
        self.blurpool = BlurPool()
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
        # avoid 2 inplace ops by chaining into one long op
        if self.se_module is not None:
            out = self.se_module(self.bn2(out)) + residual
        else:
            out = self.bn2(out) + residual
        return self.final_act(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, 
                 stride=1, downsample=None,
                 groups=1, base_width=64, 
                 use_se=False,
                 dilation=1,
                 norm_layer=ABN,
                 norm_act='relu',
                 antialias=False):
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
        self.bn3 = norm_layer(outplanes, activation='identity')
        self.se_module = SEModule(outplanes, planes // 4) if use_se else None
        self.final_act = activation_from_name(norm_act)
        self.downsample = downsample
        self.blurpool = BlurPool()
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


class Transition(nn.Module):
    r"""
    Transition Block as described in [DenseNet](https://arxiv.org/abs/1608.06993)
    
    - ReLU
    - 1x1 Convolution (with optional compression of the number of channels)
    - 2x2 Average Pooling
    """
    def __init__(self, in_planes, out_planes,
                 drop_rate=0.0,
                 norm_layer=nn.BatchNorm2d,
                 norm_act=nn.ReLU(inplace=True),
                 global_pool=nn.AvgPool2d(kernel_size=2, stride=2, padding=0)):

        super(Transition, self).__init__()
        self.norm = norm_layer(in_planes)
        self.relu = norm_act
        self.conv = conv1x1(in_planes, out_planes)
        self.pool = global_pool
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, inplace=True)
        out = self.pool(out)


class DenseLayer(nn.Module):
    expansion = 4

    def __init__(self, in_planes, growth_rate, 
                 norm_layer=ABN,
                 norm_act='relu',
                 drop_rate=0.0):
        super(DenseLayer, self).__init__()
        
        width = growth_rate * expansion

        self.norm1 = norm_layer(in_planes, activation=norm_act)
        self.conv1 = conv1x1(inplanes, width)

        self.norm2 = norm_layer(width, activation=norm_act)
        self.conv2 = conv3x3(width, growth_rate)
        self.drop_rate = drop_rate

    def forward(self, x):

        bottleneck_out = self.norm1(self.conv1(x))
        out = self.conv2(self.norm2(bottleneck_output))
        
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
    
        return torch.cat([x, out], 1)

