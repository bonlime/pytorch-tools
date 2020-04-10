import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual import DepthwiseSeparableConv

class FastNormalizedFusion(nn.Module):
    """Combines 2 or 3 feature maps into one with weights.
    Args:
        input_num (int): 2 for intermediate features, 3 for output features
    """
    def __init__(self, in_nodes):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(in_nodes, dtype=torch.float32))
        self.register_buffer("eps", torch.tensor(0.0001))

    def forward(self, *features):
        # Assure that weights are positive (see paper)
        weights = F.relu(self.weights)
        # Normalize weights
        weights /= weights.sum() + self.eps
        fused_features = sum([p * w for p, w in zip(features, weights)])
        return fused_features




# close to one in the paper
class BiFPNLayer(nn.Module):
    r"""Builds one layer of Bi-directional Feature Pyramid Network
    Args:
        channels (int): Number of channels in each feature map after BiFPN. Defaults to 64.
        downsample_by_stride (bool): If True, use convolution layer with stride=2 instead of F.interpolate
        upsample_mode (str): how to upsample low resolution features during top_down pathway. 
            See F.interpolate mode for details.

    Input:
        features (List): 5 feature maps from encoder with resolution from 1/32 to 1/2

    Returns:
        p_out: features processed by 1 layer of BiFPN
    """

    def __init__(self, channels=64, output_stride=32, upsample_mode="nearest", **bn_args):
        super(BiFPNLayer, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.first_up = self.up if output_stride == 32 else nn.Identity()
        last_stride = 2 if output_stride == 32 else 1
        self.down_p2 = DepthwiseSeparableConv(channels, channels, stride=2, **bn_args)
        self.down_p3 = DepthwiseSeparableConv(channels, channels, stride=2, **bn_args)
        self.down_p4 = DepthwiseSeparableConv(channels, channels, stride=last_stride, **bn_args)

        ## TODO (jamil) 11.02.2020 Rewrite this using list comprehensions
        self.fuse_p4_td = FastNormalizedFusion(in_nodes=2)
        self.fuse_p3_td = FastNormalizedFusion(in_nodes=2)
        self.fuse_p2_td = FastNormalizedFusion(in_nodes=2)
        self.fuse_p1_td = FastNormalizedFusion(in_nodes=2)

        # Top-down pathway, no block for P1 layer
        self.p4_td = DepthwiseSeparableConv(channels, channels, **bn_args)
        self.p3_td = DepthwiseSeparableConv(channels, channels, **bn_args)
        self.p2_td = DepthwiseSeparableConv(channels, channels, **bn_args)

        # Bottom-up pathway
        self.fuse_p2_out = FastNormalizedFusion(in_nodes=3)
        self.fuse_p3_out = FastNormalizedFusion(in_nodes=3)
        self.fuse_p4_out = FastNormalizedFusion(in_nodes=3)
        self.fuse_p5_out = FastNormalizedFusion(in_nodes=2)

        self.p5_out = DepthwiseSeparableConv(channels, channels, **bn_args)
        self.p4_out = DepthwiseSeparableConv(channels, channels, **bn_args)
        self.p3_out = DepthwiseSeparableConv(channels, channels, **bn_args)
        
    
    def forward(self, features):
        p5_inp, p4_inp, p3_inp, p2_inp = features
        
        # Top-down pathway
        p4_td = self.p4_td(self.fuse_p4_td(p4_inp, self.first_up(p5_inp)))
        p3_td = self.p3_td(self.fuse_p3_td(p3_inp, self.up(p4_td)))
        p2_out = self.p2_td(self.fuse_p2_td(p2_inp, self.up(p3_td)))

        # Calculate Bottom-Up Pathway
        p3_out = self.p3_out(self.fuse_p3_out(p3_inp, p3_td, self.down_p2(p2_out)))
        p4_out = self.p4_out(self.fuse_p4_out(p4_inp, p4_td, self.down_p3(p3_out)))
        p5_out = self.p5_out(self.fuse_p5_out(p5_inp, self.down_p4(p4_out)))

        return p5_out, p4_out, p3_out, p2_out

# very simplified
class SimpleBiFPNLayer(nn.Module):
    def __init__(self, channels=64, **bn_args):
        super(SimpleBiFPNLayer, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.down_p2 = DepthwiseSeparableConv(channels, channels, stride=2)
        self.down_p3 = DepthwiseSeparableConv(channels, channels, stride=2)
        self.down_p4 = DepthwiseSeparableConv(channels, channels, stride=2)

        self.fuse = sum

    def forward(self, features):
        p5_inp, p4_inp, p3_inp, p2_inp = features
        
        # Top-down pathway
        p4_td = self.fuse(p4_inp, self.up(p5_inp))
        p3_td = self.fuse(p3_inp, self.up(p4_td))
        p2_out = self.fuse(p2_inp, self.up(p3_td))

        # Calculate Bottom-Up Pathway
        p3_out = self.fuse(p3_inp, p3_td, self.down_p2(p2_out))
        p4_out = self.fuse(p4_inp, p4_td, self.down_p3(p3_out))
        p5_out = self.fuse(p5_inp, self.down_p4(p4_out))

        return p5_out, p4_out, p3_out, p2_out


class BiFPN(nn.Module):
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

    def __init__(
        self,
        encoder_channels,
        pyramid_channels=64,
        num_layers=1,
        output_stride=32,
        **bn_args,
    ):
        super(BiFPN, self).__init__()

        self.input_convs = nn.ModuleList([nn.Conv2d(in_ch, pyramid_channels, 1) for in_ch in encoder_channels])

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNLayer(pyramid_channels, output_stride, **bn_args))
        self.bifpn = nn.Sequential(*bifpns)
    
    def forward(self, features):

        # Preprocces raw encoder features 
        p5, p4, p3, p2 = [inp_conv(feature) for inp_conv, feature in zip(self.input_convs, features)]

        # Apply BiFPN block `num_layers` times
        p5_out, p4_out, p3_out, p2_out = self.bifpn([p5, p4, p3, p2])
        return p5_out, p4_out, p3_out, p2_out
