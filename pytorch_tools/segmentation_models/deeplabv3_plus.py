# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import logging
from pytorch_tools.modules.decoder import DeepLabHead

# from pytorch_tools.modules.residual import conv3x3, conv1x1
# from pytorch_tools.modules import ABN
# from pytorch_tools.utils.misc import initialize
from pytorch_tools.modules import bn_from_name
from .base import EncoderDecoder
from .encoders import get_encoder


class DeepLabV3(EncoderDecoder):
    """Deeplabv3+ model for image segmentation

    Args:
        encoder_name: name of classification model used as feature extractor to build segmentation model.
            Models expects encoder to have output stride 16 or 8. Only Resnet family models are supported for now
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        num_classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        norm_layer (str): Normalization layer to use. One of 'abn', 'inplaceabn'. The inplace version lowers memory
            footprint. But increases backward time. Defaults to 'abn'.
        norm_act (str): Activation for normalizion layer. 'inplaceabn' doesn't support `ReLU` activation.
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
        norm_layer="abn",
        norm_act="relu",
        **encoder_params,
    ):
        if encoder_params.get("output_stride") is None:
            logging.warning("No output_stride was given. DeepLab expects OS=16 or 8.")

        encoder = get_encoder(
            encoder_name,
            norm_layer=norm_layer,
            norm_act=norm_act,
            encoder_weights=encoder_weights,
            **encoder_params,
        )

        decoder = DeepLabHead(
            encoder_channels=encoder.out_shapes,
            num_classes=num_classes,
            norm_layer=bn_from_name(norm_layer),
            norm_act=norm_act,
        )

        super().__init__(encoder, decoder)

        self.name = f"link-{encoder_name}"
