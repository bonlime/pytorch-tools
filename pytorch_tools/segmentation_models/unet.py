import torch.nn as nn
from ..modules.decoder import UnetDecoderBlock
from ..modules.residual import conv1x1
from ..utils.misc import initialize

class UnetCenterBlock(UnetDecoderBlock):
    def forward(self, x):
        self.block(x)


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_bn=True,
            center=False,
            **bn_params, #norm layer, norm_act
    ):
        super().__init__()
        if center:
                channels = encoder_channels[0]
                self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
            else:
                self.center = None

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = UnetDecoderBlock(in_channels[0], out_channels[0], use_bn, **bn_params)
        self.layer2 = UnetDecoderBlock(in_channels[1], out_channels[1], use_bn, **bn_params)
        self.layer3 = UnetDecoderBlock(in_channels[2], out_channels[2], use_bn, **bn_params)
        self.layer4 = UnetDecoderBlock(in_channels[3], out_channels[3], use_bn, **bn_params)
        self.layer5 = UnetDecoderBlock(in_channels[4], out_channels[4], use_bn, **bn_params)
        self.final_conv = conv1x1(out_channels[4], final_channels)
        
        initialize(self)
    
    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)

        return x



