import torch
import torch.nn as nn

# import torch.nn.functional as F
from pytorch_tools.modules.fpn import FPN

# from pytorch_tools.modules.bifpn import BiFPN
from pytorch_tools.modules import bn_from_name

# from pytorch_tools.modules.residual import conv1x1
from pytorch_tools.modules.residual import conv3x3

# from pytorch_tools.modules.decoder import SegmentationUpsample
# from pytorch_tools.utils.misc import initialize
from pytorch_tools.segmentation_models.encoders import get_encoder


class RetinaNet(nn.Module):
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        pyramid_channels=256,
        num_classes=80,
        num_head_repeats=4,
        drop_connect_rate=0,
        encoder_norm_layer="abn", # maybe change default to `frozenabn`
        encoder_norm_act="relu",
        decoder_norm_layer="abn",
        decoder_norm_act="relu",
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
        self.pyramid6 = nn.Sequential(
            conv3x3(pyramid_channels, pyramid_channels, 2, bias=True),
            norm_layer(256, activation="identity"),
        )
        self.pyramid7 = nn.Sequential(
            conv3x3(pyramid_channels, pyramid_channels, 2, bias=True),
            norm_layer(256, activation="identity"),
        )
        self.fpn = FPN(self.encoder.out_shapes[:-2], pyramid_channels=pyramid_channels)

        def make_head(out_size):
            layers = []
            for _ in range(num_head_repeats):
                # The convolution layers in the class net are shared among all levels, but
                # each level has its batch normlization to capture the statistical
                # difference among different levels
                layers += [conv3x3(pyramid_channels, pyramid_channels, bias=True)]
            layers += [nn.Conv2d(pyramid_channels, out_size, 3, padding=1)]
            return nn.ModuleList(layers)
        
        def make_head_norm():
            return nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            norm_layer(pyramid_channels, activation=decoder_norm_act)
                            for _ in range(num_head_repeats)
                        ]
                        + [nn.Identity()]  # no bn after last depthwise conv
                    )
                    for _ in range(5)
                ]
            )

        anchors_per_location = 9
        self.cls_head_convs = make_head(num_classes * anchors_per_location)
        self.cls_head_norms = make_head_norm()
        self.box_head_convs = make_head(4 * anchors_per_location)
        self.box_head_norms = make_head_norm()
        self.num_classes = num_classes

    def forward(self, x):
        # don't use p2 and p1
        p5, p4, p3, _, _ = self.encoder(x)
        # enhance features
        p5, p4, p3 = self.fpn([p5, p4, p3])
        # coarser FPN levels
        p6 = self.pyramid6(p5)
        p7 = self.pyramid7(p6) # in TPU repo there is no relu for some reason
        # want features from lowest OS to highest to align with `generate_anchors_boxes` function
        features = [p3, p4, p5, p6, p7]
        # TODO: (18.03.20) TF implementation has additional BN here before class/box outputs
        class_outputs = []
        box_outputs = []
        for f in features:
            cls_anchor = self.cls_head(f).transpose(1, 3).contiguous().view(x.shape[0], -1, self.num_classes)
            box_anchor = self.box_head(f).transpose(1, 3).contiguous().view(x.shape[0], -1, 4)
            class_outputs.append(cls_anchor)
            box_outputs.append(box_anchor)
        class_outputs = torch.cat(class_outputs, 1)
        box_outputs = torch.cat(box_outputs, 1)
        return class_outputs, box_outputs
