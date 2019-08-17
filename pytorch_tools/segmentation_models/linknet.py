import torch.nn as nn
from ..modules.decoder import LinknetDecoderBlock
from ..modules.residual import conv1x1
from ..utils.misc import initialize

class LinknetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            prefinal_channels=32,
            final_channels=1,
            use_bn=True,
            **bn_params, #norm layer, norm_act
    ):
        super().__init__()

        in_channels = encoder_channels
        self.layer1 = LinknetDecoderBlock(in_channels[0], in_channels[1], use_bn, **bn_params)
        self.layer2 = LinknetDecoderBlock(in_channels[1], in_channels[2], use_bn, **bn_params)
        self.layer3 = LinknetDecoderBlock(in_channels[2], in_channels[3], use_bn, **bn_params)
        self.layer4 = LinknetDecoderBlock(in_channels[3], in_channels[4], use_bn, **bn_params)
        self.layer5 = LinknetDecoderBlock(in_channels[4], prefinal_channels, use_bn, **bn_params)
        self.final_conv = conv1x1(prefinal_channels, final_channels)

        initialize(self)

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)

        return x
