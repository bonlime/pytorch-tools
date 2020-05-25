""" Implements Feature Piramid Pooling for Object Detection and Semantic Segmentation """
# code kindly borrowed from https://github.com/qubvel/segmentation_models.pytorch
import torch.nn as nn
import torch.nn.functional as F
from .residual import conv1x1, conv3x3


class MergeBlock(nn.Module):
    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x += skip
        return x


class FPN(nn.Module):
    """Feature Pyramid Network for enhancing high-resolution feature maps with semantic
    meaning from low resolution maps
    Ref: https://arxiv.org/abs/1612.03144
        
    Args:
        encoder_channels (List[int]): Number of channels for each input feature map
        pyramid_channels (int): Number of channels in each feature map after FPN. Defaults to 256.
        num_layers (int): Number of FPN layers.
    Input:
        features (List): this module expects list of feature maps of different resolution
    """

    def __init__(
        self,
        encoder_channels,
        pyramid_channels=256,
        num_layers=1,
        **bn_args,  # for compatability only. Not used
    ):
        super().__init__()
        assert num_layers == 1, "More that 1 layer is not supported in FPN"

        # we DO use bias in this convs
        self.lateral_convs = nn.ModuleList(
            [conv1x1(in_ch, pyramid_channels, bias=True) for in_ch in encoder_channels]
        )
        self.smooth_convs = nn.ModuleList(
            [conv3x3(pyramid_channels, pyramid_channels, bias=True) for in_ch in encoder_channels]
        )
        self.merge_block = MergeBlock()

    def forward(self, features):
        """features (List[torch.Tensor]): features from coarsest to finest"""
        # project features
        pyramid_features = [l_conv(feature) for l_conv, feature in zip(self.lateral_convs, features)]
        # merge features inplace
        for idx in range(1, len(pyramid_features)):
            pyramid_features[idx] = self.merge_block([pyramid_features[idx - 1], pyramid_features[idx]])
        # smooth them after merging
        pyramid_features = [s_conv(feature) for s_conv, feature in zip(self.smooth_convs, pyramid_features)]
        return pyramid_features
