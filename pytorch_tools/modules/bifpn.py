import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import activation_from_name
from .residual import DepthwiseSeparableConv
from .residual import conv1x1
from . import ABN


class FastNormalizedFusion(nn.Module):
    """Combines 2 or 3 feature maps into one with weights.
    Args:
        input_num (int): 2 for intermediate features, 3 for output features
    """

    def __init__(self, in_nodes, activation="relu"):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(in_nodes, dtype=torch.float32))
        self.eps = 1e-4
        self.act = activation_from_name(activation)

    def forward(self, *features):
        # Assure that weights are positive (see paper)
        weights = F.relu(self.weights)
        # Normalize weights
        weights /= weights.sum() + self.eps
        fused_features = sum([p * w for p, w in zip(features, weights)])
        return self.act(fused_features)


# need to create weights to allow loading anyway. So inherit from FastNormalizedFusion for simplicity
class SumFusion(FastNormalizedFusion):
    def forward(self, *features):
        return self.act(sum(features))


class BiFPNLayer(nn.Module):
    """Builds one layer of Bi-directional Feature Pyramid Network
    Args:
        channels (int): Number of channels in each feature map after BiFPN. Defaults to 64.
        
    Input:
        features (List): 5 feature maps from encoder with resolution from 1/128 to 1/8

    Returns:
        p_out: features processed by 1 layer of BiFPN
    """

    def __init__(self, channels=64, norm_layer=ABN, norm_act="relu"):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.down = nn.MaxPool2d(3, stride=2, padding=1)

        # disable attention for large models. This is very dirty way to check that it's B6 & B7. But i don't care
        Fusion = SumFusion if channels > 288 else FastNormalizedFusion

        # There is no activation in SeparableConvs, instead activation is in fusion layer
        self.fuse_p6_up = Fusion(in_nodes=2, activation=norm_act)
        self.fuse_p5_up = Fusion(in_nodes=2, activation=norm_act)
        self.fuse_p4_up = Fusion(in_nodes=2, activation=norm_act)

        self.fuse_p3_out = Fusion(in_nodes=2, activation=norm_act)
        self.fuse_p4_out = Fusion(in_nodes=3, activation=norm_act)
        self.fuse_p5_out = Fusion(in_nodes=3, activation=norm_act)
        self.fuse_p6_out = Fusion(in_nodes=3, activation=norm_act)
        self.fuse_p7_out = Fusion(in_nodes=2, activation=norm_act)

        bn_args = dict(norm_layer=norm_layer, norm_act="identity")
        # Top-down pathway, no block for P7 layer
        self.p6_up = DepthwiseSeparableConv(channels, channels, **bn_args)
        self.p5_up = DepthwiseSeparableConv(channels, channels, **bn_args)
        self.p4_up = DepthwiseSeparableConv(channels, channels, **bn_args)

        # Bottom-up pathway
        self.p3_out = DepthwiseSeparableConv(channels, channels, **bn_args)
        self.p4_out = DepthwiseSeparableConv(channels, channels, **bn_args)
        self.p5_out = DepthwiseSeparableConv(channels, channels, **bn_args)
        self.p6_out = DepthwiseSeparableConv(channels, channels, **bn_args)
        self.p7_out = DepthwiseSeparableConv(channels, channels, **bn_args)

    def forward(self, features):

        # p7, p6, p5, p4, p3
        p7_in, p6_in, p5_in, p4_in, p3_in = features

        # Top-down pathway (from low res to high res)
        p6_up = self.p6_up(self.fuse_p6_up(p6_in, self.up(p7_in)))
        p5_up = self.p5_up(self.fuse_p5_up(p5_in, self.up(p6_up)))
        p4_up = self.p4_up(self.fuse_p4_up(p4_in, self.up(p5_up)))
        p3_out = self.p3_out(self.fuse_p3_out(p3_in, self.up(p4_up)))

        # Bottom-Up Pathway (from high res to low res)
        p4_out = self.p4_out(self.fuse_p4_out(p4_in, p4_up, self.down(p3_out)))
        p5_out = self.p5_out(self.fuse_p5_out(p5_in, p5_up, self.down(p4_out)))
        p6_out = self.p6_out(self.fuse_p6_out(p6_in, p6_up, self.down(p5_out)))
        p7_out = self.p7_out(self.fuse_p7_out(p7_in, self.down(p6_out)))

        return p7_out, p6_out, p5_out, p4_out, p3_out


# additionally downsamples the input
class FirstBiFPNLayer(BiFPNLayer):
    def __init__(self, encoder_channels, channels=64, norm_layer=ABN, norm_act="relu"):
        super().__init__(channels=channels, norm_layer=norm_layer, norm_act=norm_act)

        # TODO: later remove bias from downsample
        self.p5_downsample_1 = nn.Sequential(
            conv1x1(encoder_channels[0], channels, bias=True), norm_layer(channels, activation="identity")
        )
        self.p4_downsample_1 = nn.Sequential(
            conv1x1(encoder_channels[1], channels, bias=True), norm_layer(channels, activation="identity")
        )
        self.p3_downsample_1 = nn.Sequential(
            conv1x1(encoder_channels[2], channels, bias=True), norm_layer(channels, activation="identity")
        )

        # Devil is in the details. In original repo they use 2 different downsamples from encoder channels
        # it makes sense to preseve more information, but most of implementations in the internet
        # use output of the first downsample
        self.p4_downsample_2 = nn.Sequential(
            conv1x1(encoder_channels[1], channels, bias=True), norm_layer(channels, activation="identity")
        )
        self.p5_downsample_2 = nn.Sequential(
            conv1x1(encoder_channels[0], channels, bias=True), norm_layer(channels, activation="identity")
        )
        # only one downsample for p3

    def forward(self, features):

        # p7, p6, p5, p4, p3
        p7_in, p6_in, p5_in, p4_in, p3_in = features

        # downsample input's convs
        p5_in_down1 = self.p5_downsample_1(p5_in)
        p5_in_down2 = self.p5_downsample_2(p5_in)
        p4_in_down1 = self.p4_downsample_1(p4_in)
        p4_in_down2 = self.p4_downsample_2(p4_in)
        p3_in_down1 = self.p3_downsample_1(p3_in)

        # Top-down pathway (from low res to high res)
        p6_up = self.p6_up(self.fuse_p6_up(p6_in, self.up(p7_in)))
        p5_up = self.p5_up(self.fuse_p5_up(p5_in_down1, self.up(p6_up)))
        p4_up = self.p4_up(self.fuse_p4_up(p4_in_down1, self.up(p5_up)))
        p3_out = self.p3_out(self.fuse_p3_out(p3_in_down1, self.up(p4_up)))

        # Bottom-Up Pathway (from high res to low res)
        p4_out = self.p4_out(self.fuse_p4_out(p4_in_down2, p4_up, self.down(p3_out)))
        p5_out = self.p5_out(self.fuse_p5_out(p5_in_down2, p5_up, self.down(p4_out)))
        p6_out = self.p6_out(self.fuse_p6_out(p6_in, p6_up, self.down(p5_out)))
        p7_out = self.p7_out(self.fuse_p7_out(p7_in, self.down(p6_out)))

        return p7_out, p6_out, p5_out, p4_out, p3_out


class BiFPN(nn.Sequential):
    """
    Implementation of Bi-directional Feature Pyramid Network

    Args:
        encoder_channels (List[int]): Number of channels for each feature map from low res to high res.
        pyramid_channels (int): Number of channels in each feature map after BiFPN. Defaults to 64.
        num_layers (int): Number or repeats for BiFPN block. Default is 2 
    
    Input:
        features (List): 5 feature maps from encoder [low_res, ... , high_res]

    https://arxiv.org/pdf/1911.09070.pdf
    """

    def __init__(self, encoder_channels, pyramid_channels=64, num_layers=1, **bn_args):
        # First layer preprocesses raw encoder features
        bifpns = [FirstBiFPNLayer(encoder_channels, pyramid_channels, **bn_args)]
        # Apply BiFPN block `num_layers` times
        for _ in range(num_layers - 1):
            bifpns.append(BiFPNLayer(pyramid_channels, **bn_args))
        super().__init__(*bifpns)
