import torch
import torch.nn as nn
from inplace_abn import ABN
from .residual import conv3x3, conv1x1

relu = nn.ReLU(inplace=True)

class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_bn=True,
                 norm_layer=ABN, norm_act='relu'):
        super(UnetDecoderBlock, self).__init__()

        conv1 = conv3x3(in_channels, out_channels)
        conv2 = conv3x3(out_channels, out_channels)
        abn1 = norm_layer(out_channels, norm_act) if use_bn else relu
        abn2 = norm_layer(out_channels, norm_act) if use_bn else relu
        self.block = nn.Sequential(conv1, abn1, conv2, abn2)

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x

class TransposeX2(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 use_bn=True, 
                 norm_layer=ABN, norm_act='relu'):
        super().__init__()
        conv1 = nn.ConvTranspose2d(in_channels, out_channels, 
                                    kernel_size=4, stride=2, padding=1)
        abn = norm_layer(out_channels, norm_act) if use_bn else relu
        self.block = nn.Sequential(conv1, abn)

    def forward(self, x):
        return self.block(x)

class LinknetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 use_bn=True,
                 norm_layer=ABN, norm_act='relu'):
        super().__init__()
        middle_channels = in_channels // 4
        conv1 = conv1x1(in_channels, middle_channels)
        transpose = TransposeX2(middle_channels, middle_channels,
                                use_bn, norm_layer, norm_act)
        conv2 = conv1x1(middle_channels, out_channels)
        abn1 = norm_layer(middle_channels, norm_act) if use_bn else nn.ReLU
        abn2 = norm_layer(out_channels, norm_act) if use_bn else nn.ReLU
        self.block = nn.Sequential(conv1, abn1, transpose, conv2, abn2)

    def forward(self, x):
        x, skip = x
        x = self.block(x)
        if skip is not None:
            x = x + skip
        return x
