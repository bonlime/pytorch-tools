import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual import DepthwiseSeparableConv

class BiFPNBlock(nn.Module):
    r""" Basic block for Bi-directional Feature Pyramid Network
    Args:
        pyramid_channels (int): Number of channels in each feature map after BiFPN. Defaults to 64.
        num_layers (int): Number or repeats for BiFPN block. Default is 2 
    
    Input:
        features (List): 5 feature maps from encoder with resolution from 1/128 to 1/8

    Returns:
        p_out: features processed by 1 layer of BiFPN
    """
    def __init__(self, feature_size=64, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        
        # Top-down pathway
        self.p3_td = DepthwiseSeparableConv(feature_size, feature_size)
        self.p4_td = DepthwiseSeparableConv(feature_size, feature_size)
        self.p5_td = DepthwiseSeparableConv(feature_size, feature_size)
        self.p6_td = DepthwiseSeparableConv(feature_size, feature_size)
        
        # Bottom-up pathway
        self.p4_out = DepthwiseSeparableConv(feature_size, feature_size)
        self.p5_out = DepthwiseSeparableConv(feature_size, feature_size)
        self.p6_out = DepthwiseSeparableConv(feature_size, feature_size)
        self.p7_out = DepthwiseSeparableConv(feature_size, feature_size)
        
        # Weights for top-down feature fusion
        self.w1 = nn.Parameter(torch.Tensor(2, 4))
        self.w1_relu = nn.ReLU()

        # Weights for bottom-up feature fusion
        self.w2 = nn.Parameter(torch.Tensor(3, 4))
        self.w2_relu = nn.ReLU()
    
    def forward(self, inputs):
        p7_x, p6_x, p5_x, p4_x, p3_x = inputs
        
        # Calculate Top-Down Pathway
        w1 = self.w1_relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.epsilon
        w2 = self.w2_relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon
        
        p7_td = p7_x
        p6_td = self.p6_td(w1[0, 0] * p6_x + w1[1, 0] * F.interpolate(p7_td, scale_factor=2))        
        p5_td = self.p5_td(w1[0, 1] * p5_x + w1[1, 1] * F.interpolate(p6_td, scale_factor=2))
        p4_td = self.p4_td(w1[0, 2] * p4_x + w1[1, 2] * F.interpolate(p5_td, scale_factor=2))
        p3_td = self.p3_td(w1[0, 3] * p3_x + w1[1, 3] * F.interpolate(p4_td, scale_factor=2))
        
        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(w2[0, 0] * p4_x + w2[1, 0] * p4_td + w2[2, 0] * nn.Upsample(scale_factor=0.5)(p3_out))
        p6_out = self.p5_out(w2[0, 1] * p5_x + w2[1, 1] * p5_td + w2[2, 1] * nn.Upsample(scale_factor=0.5)(p4_out))
        p6_out = self.p6_out(w2[0, 2] * p6_x + w2[1, 2] * p6_td + w2[2, 2] * nn.Upsample(scale_factor=0.5)(p5_out))
        p7_out = self.p7_out(w2[0, 3] * p7_x + w2[1, 3] * p7_td + w2[2, 3] * nn.Upsample(scale_factor=0.5)(p6_out))

        return p7_out, p6_out, p5_out, p4_out, p3_out


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
            bifpns.append(BiFPNBlock(pyramid_channels, epsilon=0.0001))
        self.bifpn = nn.Sequential(*bifpns)
    
    def forward(self, features):

        # Preprocces raw encoder features 
        p7, p6, p5, p4, p3 = [inp_conv(feature) for inp_conv, feature in zip(self.input_convs, features)]

        print(p7.shape(), p6.shape(), p3.shape())
        # Apply BiFPN block `num_layers` times
        return *self.bifpn(p7, p6, p5, p4, p3)
