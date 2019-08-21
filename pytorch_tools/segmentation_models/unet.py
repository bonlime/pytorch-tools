import torch.nn as nn
from ..modules.decoder import UnetDecoderBlock
from ..modules.residual import conv1x1
from ..utils.misc import initialize
from ..utils.misc import bn_from_name
from .base import EncoderDecoder
from .encoders import get_encoder

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
            **bn_params): #norm layer, norm_act
             
        super().__init__()
        if center:
                channels = encoder_channels[0]
                self.center = UnetCenterBlock(channels, channels, use_bn=use_bn)
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

class Unet(EncoderDecoder):
    """Unet_ is a fully convolution neural network for image semantic segmentation
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
        center: if ``True`` add ``Conv2dReLU`` block on encoder head (useful for VGG models)
    Returns:
        ``torch.nn.Module``: **Unet**
    .. _Unet:
        https://arxiv.org/pdf/1505.04597
    """

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=1,
            activation='sigmoid',
            center=False,  # usefull for VGG models
            norm_layer='abn',
            **encoder_params):
        encoder = get_encoder(
            encoder_name,
            norm_layer=norm_layer,
            encoder_weights=encoder_weights,
            **encoder_params,
        )
        norm_act = 'relu' if norm_layer.lower() == 'abn' else 'leaky_relu'
        decoder = UnetDecoder(
            encoder_channels=encoder.out_shapes,
            decoder_channels=decoder_channels,
            final_channels=classes,
            use_bn=decoder_use_batchnorm,
            center=center,
            norm_layer=bn_from_name(norm_layer),
            norm_act=norm_act,
        )

        super().__init__(encoder, decoder, activation)
        self.name = 'u-{}'.format(encoder_name)
