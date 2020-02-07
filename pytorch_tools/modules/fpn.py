""" Implements Feature Piramid Pooling for Object Detection and Semantic Segmentation """
# code kindly borrowed from https://github.com/qubvel/segmentation_models.pytorch
import torch.nn as nn
import torch.nn.functional as F
from .residual import conv1x1


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = conv1x1(skip_channels, pyramid_channels)

    def forward(self, x):
        # TODO: (emil) 06.02.20 maybe interpolate to skip.size[-2:] to support dilation?
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class FPN(nn.Module):
    """Feature Pyramid Network for enhancing high-resolution feature maps with semantic
    meaning from low resolution maps
    Ref: https://arxiv.org/abs/1612.03144
        
    Args:
        encoder_channels (List[int]): Number of channels for each feature map
        pyramid_channels (int): Number of channels in each feature map after FPN. Defaults to 256.
    
    Input:
        features (List): this module expects list of 5 feature maps of different resolution
    """     
    
    def __init__(
            self,
            encoder_channels,
            pyramid_channels=256,
    ):     
        super().__init__()

        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

    def forward(self, features):
        # only use resolutions from 1/32 to 1/4
        c5, c4, c3, c2, c1 = features
        p5 = self.p5(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])
        return p5, p4, p3, p2, c1