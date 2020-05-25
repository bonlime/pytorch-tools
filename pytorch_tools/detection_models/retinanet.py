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
        norm_layer="abn",
        norm_act="relu",
        **encoder_params,
    ):
        super().__init__()
        self.encoder = get_encoder(
            encoder_name,
            norm_layer=norm_layer,
            norm_act=norm_act,
            encoder_weights=encoder_weights,
            **encoder_params,
        )
        norm_layer = bn_from_name(norm_layer)
        self.pyramid6 = conv3x3(256, 256, 2, bias=True)
        self.pyramid7 = conv3x3(256, 256, 2, bias=True)
        self.fpn = FPN(self.encoder.out_shapes[:-2], pyramid_channels=pyramid_channels,)

        def make_head(out_size):
            layers = []
            for _ in range(4):
                # some implementations don't use BN here but I think it's needed
                # TODO: test how it affects results
                # upd. removed norm_layer. maybe change to group_norm later
                layers += [nn.Conv2d(256, 256, 3, padding=1)]  # norm_layer(256, activation=norm_act)
                # layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()]

            layers += [nn.Conv2d(256, out_size, 3, padding=1)]
            return nn.Sequential(*layers)

        self.ratios = [1.0, 2.0, 0.5]
        self.scales = [4 * 2 ** (i / 3) for i in range(3)]
        anchors = len(self.ratios) * len(self.scales)  # 9

        self.cls_head = make_head(num_classes * anchors)
        self.box_head = make_head(4 * anchors)
        self.num_classes = num_classes

    def forward(self, x):
        # don't use p2 and p1
        p5, p4, p3, _, _ = self.encoder(x)
        # enhance features
        p5, p4, p3 = self.fpn([p5, p4, p3])
        # coarser FPN levels
        p6 = self.pyramid6(p5)
        p7 = self.pyramid7(p6.relu())
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
