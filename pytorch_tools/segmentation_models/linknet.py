import torch.nn as nn
from pytorch_tools.modules.decoder import LinknetDecoderBlock
from pytorch_tools.modules.residual import conv1x1
from pytorch_tools.utils.misc import initialize
from pytorch_tools.modules import bn_from_name
from .base import EncoderDecoder
from .encoders import get_encoder


class LinknetDecoder(nn.Module):
    def __init__(
        self, encoder_channels, prefinal_channels=32, final_channels=1, **bn_params
    ):  # norm layer, norm_act
        super().__init__()

        in_channels = encoder_channels
        self.layer1 = LinknetDecoderBlock(in_channels[0], in_channels[1], **bn_params)
        self.layer2 = LinknetDecoderBlock(in_channels[1], in_channels[2], **bn_params)
        self.layer3 = LinknetDecoderBlock(in_channels[2], in_channels[3], **bn_params)
        self.layer4 = LinknetDecoderBlock(in_channels[3], in_channels[4], **bn_params)
        self.layer5 = LinknetDecoderBlock(in_channels[4], prefinal_channels, **bn_params)
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


class Linknet(EncoderDecoder):
    """Linknet_ is a fully convolution neural network for fast image semantic segmentation
    Note:
        This implementation by default has 4 skip connections (original - 3).
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
        norm_layer: abn or inplaceabn
    Returns:
        ``torch.nn.Module``: **Linknet**
    .. _Linknet:
        https://arxiv.org/pdf/1707.03718.pdf
    """

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=1,
        norm_layer="abn",
        norm_act="relu",
        **encoder_params
    ):
        encoder = get_encoder(
            encoder_name, 
            norm_layer=norm_layer, 
            norm_act=norm_act,
            encoder_weights=encoder_weights, 
            **encoder_params,
        )


        decoder = LinknetDecoder(
            encoder_channels=encoder.out_shapes,
            prefinal_channels=32,
            final_channels=classes,
            norm_layer=bn_from_name(norm_layer),
            norm_act=norm_act,
        )

        super().__init__(encoder, decoder)

        self.name = f"link-{encoder_name}"
