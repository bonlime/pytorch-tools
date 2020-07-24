"""
This file defines segmentation model based on Efficient Det networks
In paper they mention the following:

`We modify our EfficientDet model to keep feature level {P2, P3, ..., P7} in BiFPN, but only use
P2 for the final per-pixel classification. For ... B4 backbone ... we set the channel size to 128 
for BiFPN and 256 for classification head. Both BiFPN and classification head are repeated by 3 times.`

I only use features {P2, P3, P4, P5} because very often such large OS is not needed.
"""

import torch
import torch.nn as nn

from pytorch_tools.modules import ABN
from pytorch_tools.modules.bifpn import BiFPN
from pytorch_tools.modules import bn_from_name
from pytorch_tools.modules.residual import conv1x1
from pytorch_tools.modules.residual import DepthwiseSeparableConv
from pytorch_tools.modules.tf_same_ops import conv_to_same_conv
from pytorch_tools.modules.tf_same_ops import maxpool_to_same_maxpool

from pytorch_tools.models.efficientnet import patch_bn_tf
from pytorch_tools.segmentation_models.encoders import get_encoder


class SegmentationBiFPN(nn.Module):
    def __init__(
        self,
        encoder_name="efficientnet_b0",
        encoder_weights="imagenet",
        pyramid_channels=128,
        head_channels=256,
        num_classes=1,
        last_upsample=True,
        encoder_norm_layer="abn",
        encoder_norm_act="swish",
        decoder_norm_layer="abn",
        decoder_norm_act="swish",
        **encoder_params,
    ):
        super().__init__()
        self.encoder = get_encoder(
            encoder_name,
            norm_layer=encoder_norm_layer,
            norm_act=encoder_norm_act,
            encoder_weights=encoder_weights,
            **encoder_params,
        )
        norm_layer = bn_from_name(decoder_norm_layer)
        bn_args = dict(norm_layer=norm_layer, norm_act=decoder_norm_act)

        self.bifpn = BiFPN(
            # pass P2-P5
            encoder_channels=self.encoder.out_shapes[:-1],
            pyramid_channels=pyramid_channels,
            num_layers=3,  # hardcode num_fpn_layers=3
            **bn_args,
        )

        self.cls_head_conv = nn.Sequential(
            DepthwiseSeparableConv(pyramid_channels, head_channels, **bn_args),
            DepthwiseSeparableConv(head_channels, head_channels, **bn_args),
            DepthwiseSeparableConv(head_channels, num_classes, use_norm=False),
        )

        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear") if last_upsample else nn.Identity()

        self.num_classes = num_classes

        patch_bn_tf(self)
        # set last layer bias for better convergence with sigmoid loss
        # -4.59 = -np.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.cls_head_conv[-1][1].bias, -4.59)

    def _initialize_weights(self):
        pass

    def forward(self, x):
        # Extract features from backbone
        # p5, p4, p3, p2, _
        features = self.encoder(x)
        # enchance with BiFPN. don't use p1
        features = self.bifpn(features[:-1])

        p2 = features[-1]
        out = self.cls_head_conv(p2)
        out = self.upsample(out)  # maybe upsample
        return out
