import torch.nn as nn
from pytorch_tools.modules import bn_from_name
from pytorch_tools.modules.residual import conv1x1
from pytorch_tools.modules.residual import conv3x3
from pytorch_tools.modules.decoder import UnetDecoderBlock
from pytorch_tools.utils.misc import initialize
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
        center=False,
        drop_rate=0,
        output_stride=32,
        attn_type=None,
        **bn_params,  # norm layer, norm_act
    ):

        super().__init__()
        if center:
            channels = encoder_channels[0]
            self.center = UnetCenterBlock(channels, channels)
        else:
            self.center = None

        in_chs = self.compute_channels(encoder_channels, decoder_channels)
        kwargs = {**bn_params, "attn_type": attn_type}
        self.layer1 = UnetDecoderBlock(in_chs[0], decoder_channels[0], upsample=output_stride == 32, **kwargs)
        self.layer2 = UnetDecoderBlock(in_chs[1], decoder_channels[1], upsample=output_stride != 8, **kwargs)
        self.layer3 = UnetDecoderBlock(in_chs[2], decoder_channels[2], **kwargs)
        self.layer4 = UnetDecoderBlock(in_chs[3], decoder_channels[3], **kwargs)
        self.layer5 = UnetDecoderBlock(in_chs[4], decoder_channels[4], **kwargs)
        self.dropout = nn.Dropout2d(drop_rate, inplace=False)  # inplace=True raises a backprop error
        self.final_conv = conv1x1(decoder_channels[4], final_channels, bias=True)

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
        x = self.dropout(x)
        x = self.final_conv(x)

        return x


class Unet(EncoderDecoder):
    """Unet_ is a fully convolution neural network for image semantic segmentation
    Args:
        encoder_name (str): name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights (str): one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels (List[int]): list of numbers of ``Conv2D`` layer filters in decoder blocks
        num_classes (int): a number of classes for output (output shape - ``(batch, classes, h, w)``).
        center (bool): if ``True`` add ``Conv2dReLU`` block on encoder head (useful for VGG models)
        drop_rate (float): Probability of spatial dropout on last feature map
        decoder_attention_type (Union[str, None]): Attention to use in decoder layers. Options are:
            `se`, `sse`, `eca`, `scse`. Check code for reference papers and details about each type of attention.
        encoder_norm_layer (str): Normalization layer to use. One of 'abn', 'inplaceabn'. The inplace version lowers
            memory footprint. But increases backward time. Defaults to 'abn'.
        encoder_norm_act (str): Activation for normalizion layer. 'inplaceabn' doesn't support `ReLU` activation.
        decoder_norm_layer (str): same as encoder_norm_layer but for decoder
        decoder_norm_act (str): same as encoder_norm_act but for decoder

    Returns:
        ``torch.nn.Module``: **Unet**
    .. _Unet:
        https://arxiv.org/pdf/1505.04597
    """

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        decoder_channels=(256, 128, 64, 32, 16),
        num_classes=1,
        center=False,  # usefull for VGG models
        output_stride=32,
        drop_rate=0,
        decoder_attention_type=None,
        encoder_norm_layer="abn",
        encoder_norm_act="relu",
        decoder_norm_layer="abn",
        decoder_norm_act="relu",
        **encoder_params,
    ):
        if output_stride != 32:
            encoder_params["output_stride"] = output_stride
        encoder = get_encoder(
            encoder_name,
            norm_layer=encoder_norm_layer,
            norm_act=encoder_norm_act,
            encoder_weights=encoder_weights,
            **encoder_params,
        )
        decoder = UnetDecoder(
            encoder_channels=encoder.out_shapes,
            decoder_channels=decoder_channels,
            final_channels=num_classes,
            center=center,
            drop_rate=drop_rate,
            output_stride=output_stride,
            attn_type=decoder_attention_type,
            norm_layer=bn_from_name(decoder_norm_layer),
            norm_act=decoder_norm_act,
        )

        super().__init__(encoder, decoder)
        self.name = f"u-{encoder_name}"
        # set last layer bias for better convergence with sigmoid loss
        # -4.59 = -np.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.decoder.final_conv.bias, -4.59)
