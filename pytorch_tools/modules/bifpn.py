import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from .residual import DepthwiseSeparableConv
from .decoder import Conv3x3NormAct

# DepthwiseSeparableConv = Conv3x3NormAct

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

    def __init__(self,
                channels=64,
                # downsample_by_stride=True, 
                upsample_mode="bilinear"
            ):
        super(BiFPNLayer, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.down = nn.Upsample(scale_factor=0.5, mode=upsample_mode)
        # self.up = partial(F.interpolate, scale_factor=2, mode=upsample_mode) 
        # TODO (jamil) 11.02.2020 Add PixelShuffle method for interpolation
             
        # No need to interpolate last (P5) layer, thats why only 4 modules.
        # self.down_x2 = partial(F.interpolate, scale_factor=2)
        # self.down = [DepthwiseSeparableConv(channels, channels, stride=2) if downsample_by_stride \
        #         else self.down_x2 for _ in range(4)]
        
        # if downsample_by_stride:
        #     self.down = [DepthwiseSeparableConv(channels, channels, stride=2) for _ in range(4)]
        # else:
        #      self.down = [self.down_x2 for _ in range(4)]

        ## TODO (jamil) 11.02.2020 Rewrite this using list comprehensions
        self.fuse_p4_td = FastNormalizedFusion(in_nodes=2)
        self.fuse_p3_td = FastNormalizedFusion(in_nodes=2)
        self.fuse_p2_td = FastNormalizedFusion(in_nodes=2)
        self.fuse_p1_td = FastNormalizedFusion(in_nodes=2)

        # Top-down pathway, no block for P1 layer
        self.p4_td = DepthwiseSeparableConv(channels, channels)
        self.p3_td = DepthwiseSeparableConv(channels, channels)
        self.p2_td = DepthwiseSeparableConv(channels, channels)
        # self.p1_td = DepthwiseSeparableConv(channels, channels)


        # Bottom-up pathway
        self.fuse_p2_out = FastNormalizedFusion(in_nodes=3)
        self.fuse_p3_out = FastNormalizedFusion(in_nodes=3)
        self.fuse_p4_out = FastNormalizedFusion(in_nodes=3)
        self.fuse_p5_out = FastNormalizedFusion(in_nodes=2)

        self.p5_out = DepthwiseSeparableConv(channels, channels)
        self.p4_out = DepthwiseSeparableConv(channels, channels)
        self.p3_out = DepthwiseSeparableConv(channels, channels)
        # self.p2_out = DepthwiseSeparableConv(channels, channels)
        # self.p1_out = DepthwiseSeparableConv(channels, channels)
        
    
    def forward(self, features):
        p5_inp, p4_inp, p3_inp, p2_inp, p1_inp = features
        
        # Top-down pathway
        # p5_td = self.p5_td(p5_inp) ## Preprocess p5 feature
        # p4_td = self.p4_td(self.fuse_p4_td(p4_inp, self.up(p5_td)))
        p4_td = self.p4_td(self.fuse_p4_td(p4_inp, self.up(p5_inp)))
        p3_td = self.p3_td(self.fuse_p3_td(p3_inp, self.up(p4_td)))
        p2_out = self.p2_td(self.fuse_p2_td(p2_inp, self.up(p3_td)))
        # p1_out = self.p1_td(self.fuse_p1_td(p1_inp, self.up(p2_td)))

        # Calculate Bottom-Up Pathway
        # p1_out = self.p1_out(p1_td) ## DepthWise conv without fusion
        # p2_out = self.p2_out(self.fuse_p2_out(p2_inp, p2_td, self.down(p1_out)))
        p3_out = self.p3_out(self.fuse_p3_out(p3_inp, p3_td, self.down(p2_out)))
        p4_out = self.p4_out(self.fuse_p4_out(p4_inp, p4_td, self.down(p3_out)))
        p5_out = self.p5_out(self.fuse_p5_out(p5_inp, self.down(p4_out)))

        return p5_out, p4_out, p3_out, p2_out, p1_inp


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
        num_layers=1):
        super(BiFPN, self).__init__()

        self.input_convs = nn.ModuleList([nn.Conv2d(in_ch, pyramid_channels, 1) for in_ch in encoder_channels])

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNLayer(pyramid_channels))
        self.bifpn = nn.Sequential(*bifpns)
    
    def forward(self, features):

        # Preprocces raw encoder features 
        p5, p4, p3, p2, p1 = [inp_conv(feature) for inp_conv, feature in zip(self.input_convs, features)]

        # Apply BiFPN block `num_layers` times
        p5_out, p4_out, p3_out, p2_out, p1_out = self.bifpn([p5, p4, p3, p2, p1])
        return p5_out, p4_out, p3_out, p2_out, p1_out
