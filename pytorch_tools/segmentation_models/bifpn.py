import torch.nn as nn

from pytorch_tools.utils.misc import initialize
from pytorch_tools.modules import bn_from_name
from pytorch_tools.modules.bifpn import BiFPN

from .base import EncoderDecoder
from .encoders import get_encoder
 
# class BiFPNModel(SegmentationModel):
#     """BiFPN is a fully convolution neural network for image semantic segmentation
#     Args:
#         encoder_name: name of classification model (without last dense layers) used as feature
#                 extractor to build segmentation model.
#         encoder_depth: number of stages used in decoder, larger depth - more features are generated.
#             e.g. for depth=3 encoder will generate list of features with following spatial shapes
#             [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature will have
#             spatial resolution (H/(2^depth), W/(2^depth)]
#         encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
#         decoder_pyramid_channels: a number of convolution filters in Feature Pyramid of FPN_.
#         decoder_segmentation_channels: a number of convolution filters in segmentation head of FPN_.
#         decoder_merge_policy: determines how to merge outputs inside FPN.
#             One of [``add``, ``cat``]
#         decoder_dropout: spatial dropout rate in range (0, 1).
#         in_channels: number of input channels for model, default is 3.
#         classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
#         activation (str, callable): activation function used in ``.predict(x)`` method for inference.
#             One of [``sigmoid``, ``softmax2d``, callable, None]
#         upsampling: optional, final upsampling factor
#             (default is 4 to preserve input -> output spatial shape identity)
#         aux_params: if specified model will have additional classification auxiliary output
#             build on top of encoder, supported params:
#                 - classes (int): number of classes
#                 - pooling (str): one of 'max', 'avg'. Default is 'avg'.
#                 - dropout (float): dropout factor in [0, 1)
#                 - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)

#     Returns:
#         ``torch.nn.Module``: **FPN**

#     .. _FPN:
#         http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

#     """

    
#     def __init__(
#         self,
#         encoder_name="resnet34",
#         encoder_weights="imagenet",
#         num_classes=1,
#         decoder_channels=64,
#         norm_layer="abn",
#         norm_act="relu",
#         **encoder_params,
#     ):

#         super().__init__()

#         self.encoder = get_encoder(
#             encoder_name,
#             in_channels=in_channels,
#             depth=encoder_depth,
#             weights=encoder_weights,
#         )

#         self.decoder = BiFPN(
#             encoder_channels=self.encoder.out_channels,
#             pyramid_channels=decoder_channels,
#             num_layers=2
#         )


#         self.segmentation_head = SegmentationHead(
#             in_channels=self.decoder.out_channels,
#             out_channels=classes,
#             activation=activation,
#             kernel_size=1,
#             upsampling=upsampling,
#         )

#         if aux_params is not None:
#             self.classification_head = ClassificationHead(
#                 in_channels=self.encoder.out_channels[-1], **aux_params
#             )
#         else:
#             self.classification_head = None

#         self.name = "efficientdet-{}".format(encoder_name)
        # self.initialize()
