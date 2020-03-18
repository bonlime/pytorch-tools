# import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.fpn = FPN(
           self.encoder.out_shapes[:-2],
           pyramid_channels=pyramid_channels,
        )

        def make_head(out_size):
            layers = []
            for _ in range(4):
                # some implementations don't use BN here but I think it's needed
                # TODO: test how it affects results
                layers += [nn.Conv2d(256, 256, 3, padding=1), norm_layer(256, activation=norm_act)]
                # layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()]

            layers += [nn.Conv2d(256, out_size, 3, padding=1)]
            return nn.Sequential(*layers)

        self.ratios = [1.0, 2.0, 0.5]
        self.scales = [4 * 2 ** (i / 3) for i in range(3)]
        anchors = len(self.ratios) * len(self.scales) # 9

        self.cls_head = make_head(num_classes * anchors)
        self.box_head = make_head(4 * anchors)

    def forward(self, x):
        # don't use p2 and p1
        p5, p4, p3, _, _ = self.encoder(x)
        # enhance features
        p5, p4, p3 = self.fpn([p5, p4, p3])
        # coarsers FPN levels
        p6 = self.pyramid6(p5)
        p7 = self.pyramid7(F.relu(p6))
        features = [p7, p6, p5, p4, p3]
        # TODO: (18.03.20) TF implementation has additional BN here before class/box outputs
        class_outputs = [self.cls_head(f) for f in features]
        box_outputs = [self.box_head(f) for f in features]
        return class_outputs, box_outputs
        

        