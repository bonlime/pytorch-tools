import logging
import torch.nn as nn
from pytorch_tools.modules.decoder import DeepLabHead
from pytorch_tools.modules import bn_from_name
from .base import EncoderDecoder
from .encoders import get_encoder


class DeepLabV3(EncoderDecoder):
    """Deeplabv3+ model for image segmentation

    Args:
        encoder_name (str): name of classification model used as feature extractor to build segmentation model.
            Models expects encoder to have output stride 16 or 8. Only Resnet and Effnet family models are supported for now
        encoder_weights (str): one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        num_classes (int): a number of classes for output (output shape - ``(batch, classes, h, w)``).
        last_upsample (bool): Flag to enable upsampling predictions to the original image size. If set to `False` prediction
            would be 4x times smaller than input image. Default True.
        aspp_dilation_rates (List[int]): Dilations in the aspp module for output_stride == 16.
            If output_stride == 8 this numbers would be doubled implicitly.
        output_stride (int): Output stride of encoder network.
        drop_rate (float): Probability of spatial dropout on last feature map
        norm_layer (str): Normalization layer to use. One of 'abn', 'inplaceabn'. The inplace version lowers memory
            footprint. But increases backward time. Defaults to 'abn'.
        norm_act (str): Activation for normalizion layer. 'inplaceabn' doesn't support `ReLU` activation.
    Returns:
        ``torch.nn.Module``: **Deeplab**

    Ref:
        Deeplab 3+: https://arxiv.org/abs/1706.05587
    """

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        num_classes=1,
        last_upsample=True,
        aspp_dilation_rates=[6, 12, 18],
        output_stride=16,
        drop_rate=0,
        norm_layer="abn",
        norm_act="relu",
        **encoder_params,
    ):

        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights,
            output_stride=output_stride,
            norm_layer=norm_layer,
            norm_act=norm_act,
            **encoder_params,
        )

        decoder = DeepLabHead(
            encoder_channels=encoder.out_shapes,
            num_classes=num_classes,
            dilation_rates=aspp_dilation_rates,
            output_stride=output_stride,
            drop_rate=drop_rate,
            norm_layer=bn_from_name(norm_layer),
            norm_act=norm_act,
        )

        super().__init__(encoder, decoder)
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear") if last_upsample else nn.Identity()
        self.name = f"deeplabv3plus-{encoder_name}"

    def forward(self, x):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.upsample(x)
        return x
