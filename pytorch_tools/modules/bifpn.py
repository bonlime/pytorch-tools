import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from .residual import DepthwiseSeparableConv

class BiFPNFeatureFusion(nn.Module):
    r"""Combines 2 or 3 feature maps into one with weights.
    Args:
        fast_fusion (bool): If False use softmax, else just normalize, see paper for details.
        input_num (int): 2 for intermediate features, 3 for output features
        epsilon (float): small value to avoid numerical instability. Default: 0.0001

    Input:


    Returns:
        p_out: features processed by 1 layer of BiFPN
    """

    def __init__(self, input_num, fast_fusion=True, epsilon=0.0001):
        super(BiFPNFeatureFusion, self).__init__()
        self.input_num = input_num
        self.fast_fusion = fast_fusion

        ## Init weights with ones, as in the paper
        self.weights = nn.Parameter(torch.ones(input_num, dtype=torch.float32))
        self.epsilon = epsilon

    def forward(self, *features):
        if len(features) != self.input_num:
            raise RuntimeError(
                "Expected to have {} input nodes, but have {}.".format(self.input_num, len(features))
            )

        # Assure that weights are positive (see paper)
        weights = F.relu(self.weights)

        # Normalize weights
        if self.fast_fusion:
            weights /= weights.sum() + self.epsilon
        else:
            # TODO (jamil) 11.02.2020: Add softmax fusion 
            raise NotImplementedError(
                    "Softmax fusion not implemented"
                    )
        
        fused_features = sum([p * w for p, w in zip(features, weights)])
        return fused_features



class BiFPNLayer(nn.Module):
    r"""Builds one layer of Bi-directional Feature Pyramid Network
    Args:
        channels (int): Number of channels in each feature map after BiFPN. Defaults to 64.
        epsilon (float): small value to avoid numerical instability. Default: 0.0001
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
                epsilon=0.0001, 
                downsample_by_stride=True, 
                upsample_mode="bilinear"
            ):
        super(BiFPNLayer, self).__init__()
        self.epsilon = epsilon

        self.up_x2 = partial(F.interpolate, scale_factor=2, mode=upsample_mode) 
        # TODO (jamil) 11.02.2020 Add PixelShuffle method for interpolation
             
        # No need to interpolate last (P5) layer, thats why only 4 modules.
        self.down_x2 = partial(F.interpolate, scale_factor=2)
        self.down = [DepthwiseSeparableConv(channels, channels, stride=2) if downsample_by_stride \
                else self.down_x2 for _ in range(4)]
        
        # if downsample_by_stride:
        #     self.down = [DepthwiseSeparableConv(channels, channels, stride=2) for _ in range(4)]
        # else:
        #      self.down = [self.down_x2 for _ in range(4)]

        ## TODO (jamil) 11.02.2020 Rewrite this using list comprehensions
        self.fuse_p4_td = BiFPNFeatureFusion(input_num=2)
        self.fuse_p3_td = BiFPNFeatureFusion(input_num=2)
        self.fuse_p2_td = BiFPNFeatureFusion(input_num=2)
        self.fuse_p1_td = BiFPNFeatureFusion(input_num=2)

        # Top-down pathway, no block for P1 layer
        self.p5_td = DepthwiseSeparableConv(channels, channels, norm_act="relu")
        self.p4_td = DepthwiseSeparableConv(channels, channels, norm_act="relu")
        self.p3_td = DepthwiseSeparableConv(channels, channels, norm_act="relu")
        self.p2_td = DepthwiseSeparableConv(channels, channels, norm_act="relu")


        # Bottom-up pathway
        self.fuse_p5_out = BiFPNFeatureFusion(input_num=2)
        self.fuse_p4_out = BiFPNFeatureFusion(input_num=3)
        self.fuse_p3_out = BiFPNFeatureFusion(input_num=3)
        self.fuse_p2_out = BiFPNFeatureFusion(input_num=3)

        self.p5_out = DepthwiseSeparableConv(channels, channels, norm_act="relu")
        self.p4_out = DepthwiseSeparableConv(channels, channels, norm_act="relu")
        self.p3_out = DepthwiseSeparableConv(channels, channels, norm_act="relu")
        self.p2_out = DepthwiseSeparableConv(channels, channels, norm_act="relu")
        self.p1_out = DepthwiseSeparableConv(channels, channels, norm_act="relu")
        
    
    def forward(self, features):
        p5_inp, p4_inp, p3_inp, p2_inp, p1_inp = features
        
        # Top-down pathway
        p5_td = self.p5_td(p5_inp) ## Preprocess p5 feature
        p4_td = self.p4_td(self.fuse_p4_td(p4_inp, self.up(p5_td)))
        p3_td = self.p3_td(self.fuse_p3_td(p3_inp, self.up(p4_td)))
        p2_td = self.p2_td(self.fuse_p2_td(p2_inp, self.up(p3_td)))
        p1_td = self.p1_td(self.fuse_p1_td(p1_inp, self.up(p2_td)))

        # Calculate Bottom-Up Pathway
        p1_out = self.p1_out(p1_td) ## DepthWise conv without fusion
        p2_out = self.p2_out(self.fuse_p2_out(p2_inp, p2_td, self.up(p1_out)))
        p3_out = self.p3_out(self.fuse_p3_out(p3_inp, p3_td, self.up(p2_out)))
        p4_out = self.p4_out(self.fuse_p4_out(p4_inp, p4_td, self.up(p3_out)))
        p5_out = self.p5_out(self.fuse_p5_out(p5_td, self.down_p4(p4_out)))

        return p5_out, p4_out, p3_out, p2_out, p1_out


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
        num_layers=2):
        super(BiFPN, self).__init__()

        self.input_convs = nn.ModuleList([nn.Conv2d(in_ch, pyramid_channels, 1) for in_ch in encoder_channels])

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNLayer(pyramid_channels, epsilon=0.0001))
        self.bifpn = nn.Sequential(*bifpns)
    
    def forward(self, features):

        # Preprocces raw encoder features 
        p5, p4, p3, p2, p1 = [inp_conv(feature) for inp_conv, feature in zip(self.input_convs, features)]

        print(p5.shape(), p4.shape(), p3.shape())
        # Apply BiFPN block `num_layers` times
        p5_out, p4_out, p3_out, p2_out, p1_out = self.bifpn(p5, p4, p3, p2, p1)
        return p5_out, p4_out, p3_out, p2_out, p1_out
