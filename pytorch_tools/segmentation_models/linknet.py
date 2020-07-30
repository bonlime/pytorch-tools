import torch.nn as nn
from pytorch_tools.modules.decoder import LinknetDecoderBlock
from pytorch_tools.modules.residual import conv1x1
from pytorch_tools.utils.misc import initialize
from pytorch_tools.modules import bn_from_name
from .base import EncoderDecoder
from .encoders import get_encoder


class LinknetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        prefinal_channels=32,
        final_channels=1,
        drop_rate=0,
        attn_type=None,
        **bn_params,  # norm layer, norm_act
    ):
        super().__init__()
        extra_params = {**bn_params, "attn_type": attn_type}
        in_channels = encoder_channels
        self.layer1 = LinknetDecoderBlock(in_channels[0], in_channels[1], **extra_params)
        self.layer2 = LinknetDecoderBlock(in_channels[1], in_channels[2], **extra_params)
        self.layer3 = LinknetDecoderBlock(in_channels[2], in_channels[3], **extra_params)
        self.layer4 = LinknetDecoderBlock(in_channels[3], in_channels[4], **extra_params)
        self.layer5 = LinknetDecoderBlock(in_channels[4], prefinal_channels, **extra_params)
        self.dropout = nn.Dropout2d(drop_rate, inplace=True)
        self.final_conv = conv1x1(prefinal_channels, final_channels)

        # it works much better without initializing decoder. maybe need to investigate into this issue
        # initialize(self)

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.dropout(x)
        x = self.final_conv(x)

        return x


class Linknet(EncoderDecoder):
    """Linknet_ is a fully convolution neural network for fast image semantic segmentation
    Note:
        This implementation by default has 4 skip connections (original - 3).
    Args:
        encoder_name (str): name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights (str): one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        num_classes (int): a number of classes for output (output shape - ``(batch, classes, h, w)``).
        drop_rate (float): Probability of spatial dropout on last feature map
        encoder_norm_layer (str): Normalization layer to use. One of 'abn', 'inplaceabn'. The inplace version lowers
            memory footprint. But increases backward time. Defaults to 'abn'.
        encoder_norm_act (str): Activation for normalizion layer. 'inplaceabn' doesn't support `ReLU` activation.
        decoder_norm_layer (str): same as encoder_norm_layer but for decoder
        decoder_norm_act (str): same as encoder_norm_act but for decoder
    Returns:
        ``torch.nn.Module``: **Linknet**
    .. _Linknet:
        https://arxiv.org/pdf/1707.03718.pdf
    """

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        num_classes=1,
        drop_rate=0,
        decoder_attention_type=None,
        encoder_norm_layer="abn",
        encoder_norm_act="relu",
        decoder_norm_layer="abn",
        decoder_norm_act="relu",
        **encoder_params,
    ):
        encoder = get_encoder(
            encoder_name,
            norm_layer=encoder_norm_layer,
            norm_act=encoder_norm_act,
            encoder_weights=encoder_weights,
            **encoder_params,
        )

        decoder = LinknetDecoder(
            encoder_channels=encoder.out_shapes,
            prefinal_channels=32,
            final_channels=num_classes,
            drop_rate=drop_rate,
            attn_type=decoder_attention_type,
            norm_layer=bn_from_name(decoder_norm_layer),
            norm_act=decoder_norm_act,
        )

        super().__init__(encoder, decoder)

        self.name = f"link-{encoder_name}"
