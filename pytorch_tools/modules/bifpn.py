"""
Implementation of Bidirectional Feature Pyramid Network module (BiFPN)
Reference: EfficientDet: Scalable and Efficient Object Detection - https://arxiv.org/abs/1911.09070
This version supports any number of input feature maps and is suitable for segmentation as well
hacked by @bonlime
"""
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

    def __init__(self, num_features=5, channels=64, norm_layer=ABN, norm_act="relu"):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.down = nn.MaxPool2d(3, stride=2, padding=1)
        self.num_features = num_features

        # disable attention for large models. This is a very dirty way to check that it's B6 & B7. But i don't care
        Fusion = SumFusion if channels > 288 else FastNormalizedFusion

        # There is no activation in SeparableConvs, instead activation is in fusion layer
        # fusions for p6, p5, p4, p3. (no fusion for first feature map)
        self.fuse_up = nn.ModuleList(
            [Fusion(in_nodes=2, activation=norm_act) for _ in range(num_features - 1)]
        )

        # fusions for p4, p5, p6, p7. last is different because there is no bottop up tensor for it
        self.fuse_out = nn.ModuleList(
            [
                *(Fusion(in_nodes=3, activation=norm_act) for _ in range(num_features - 2)),
                Fusion(in_nodes=2, activation=norm_act),
            ]
        )

        bn_args = dict(norm_layer=norm_layer, norm_act="identity")
        # Top-down pathway, no block for first and last features. P3 and P7 by default
        self.p_up_convs = nn.ModuleList(
            [DepthwiseSeparableConv(channels, channels, **bn_args) for _ in range(num_features - 1)]
        )

        # Bottom-up pathway
        self.p_out_convs = nn.ModuleList(
            [DepthwiseSeparableConv(channels, channels, **bn_args) for _ in range(num_features - 1)]
        )

    def forward(self, features):
        # p7_in, p6_in, p5_in, p4_in, p3_in = features
        # Top-down pathway (from low res to high res). High res features depend on upsampled low res
        p_up = [features[0]]  # from p7 to p3
        for idx in range(self.num_features - 1):
            p_up.append(
                self.p_up_convs[idx](
                    self.fuse_up[idx](  # fuse: input and upsampled previous feature map
                        features[idx + 1], self.up(p_up[-1])
                    )
                )
            )

        # Bottom-Up Pathway (from high res to low res). Low res depends on downscaled high res
        p_out = [p_up[-1]]  # p3 is final and ready to be returned. from p3 to p7
        for idx in range(1, self.num_features - 1):
            p_out.append(
                self.p_out_convs[idx - 1](  # fuse: input, output from top-bottom path and downscaled high res
                    self.fuse_out[idx - 1](features[-(idx + 1)], p_up[-(idx + 1)], self.down(p_out[-1]))
                )
            )
        # fuse for p7: input, downscaled high res
        p_out.append(self.p_out_convs[-1](self.fuse_out[-1](features[0], self.down(p_out[-1]))))

        return p_out[::-1]  # want to return in the same order as input


class FirstBiFPNLayer(BiFPNLayer):
    def __init__(self, encoder_channels, channels=64, norm_layer=ABN, norm_act="relu"):
        """
        Args:
            encoder_channels (List[int]): Number of channels for each feature map from low res to high res
        """
        super().__init__(
            num_features=len(encoder_channels), channels=channels, norm_layer=norm_layer, norm_act=norm_act
        )
        # in original repo there is an additional bias in downsample 1x1 convs. because it's followed by
        # norm layer it becomes redundant, so I've removed it
        # pretrained weights for them were ~1e-5 which additionally shows that they are not needed
        self.downsample_1 = nn.ModuleList()
        self.downsample_2 = []
        for enc_in_channel in encoder_channels:
            layer, layer2 = nn.Identity(), nn.Identity()
            if enc_in_channel != channels:
                layer = nn.Sequential(
                    conv1x1(enc_in_channel, channels), norm_layer(channels, activation="identity"),
                )
                layer2 = nn.Sequential(
                    conv1x1(enc_in_channel, channels), norm_layer(channels, activation="identity"),
                )
            self.downsample_1.append(layer)
            self.downsample_2.append(layer2)
        # no second downsample for last feature map (highest res)
        self.downsample_2[-1] = nn.Identity()
        # [::-1] for proper weight order.
        self.downsample_2 = nn.ModuleList(self.downsample_2[::-1])

    def forward(self, features):
        # type: List[Tensor] -> List[Tensor]
        """Args:
            features (List[Tensor]): number of input features should match number of encoder_channels passed during init
        """
        features_in_1 = [down(feat) for down, feat in zip(self.downsample_1, features)]
        features_in_2 = [down(feat) for down, feat in zip(self.downsample_2[::-1], features)]

        # below is copy from BiFPNLayer with features changed to features_in_1 / features_in_2

        # p7_in, p6_in, p5_in, p4_in, p3_in = features
        # Top-down pathway (from low res to high res). High res features depend on upsampled low res
        p_up = [features_in_1[0]]  # from p7 to p3
        for idx in range(self.num_features - 1):
            p_up.append(
                self.p_up_convs[idx](
                    self.fuse_up[idx](  # fuse: input and upsampled previous feature map
                        features_in_1[idx + 1], self.up(p_up[-1])
                    )
                )
            )

        # Bottom-Up Pathway (from high res to low res). Low res depends on downscaled high res
        p_out = [p_up[-1]]  # p3 is final and ready to be returned. from p3 to p7
        for idx in range(1, self.num_features - 1):
            p_out.append(
                self.p_out_convs[idx - 1](  # fuse: input, output from top-bottom path and downscaled high res
                    self.fuse_out[idx - 1](features_in_2[-(idx + 1)], p_up[-(idx + 1)], self.down(p_out[-1]))
                )
            )
        # fuse for p7: input, downscaled high res
        p_out.append(self.p_out_convs[-1](self.fuse_out[-1](features_in_2[0], self.down(p_out[-1]))))

        return p_out[::-1]  # want to return in the same order as input


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
            bifpns.append(BiFPNLayer(len(encoder_channels), pyramid_channels, **bn_args))
        super().__init__(*bifpns)
