import torch
import torch.nn as nn
from pytorch_tools.modules import ABN
from pytorch_tools.modules.bifpn import BiFPN
from pytorch_tools.modules import bn_from_name
from pytorch_tools.modules.residual import conv1x1
from pytorch_tools.modules.residual import conv3x3
from pytorch_tools.modules.residual import DepthwiseSeparableConv
from pytorch_tools.modules.tf_same_ops import conv_to_same_conv
from pytorch_tools.modules.tf_same_ops import maxpool_to_same_maxpool
from pytorch_tools.segmentation_models.encoders import get_encoder

def patch_bn(module):
    """TF ported weights use slightly different eps in BN. Need to adjust for better performance"""
    if isinstance(module, ABN):
        module.eps = 1e-3
        module.momentum = 1e-2
    for m in module.children():
        patch_bn(m)

class EfficientDet(nn.Module):
    def __init__(
        self, 
        encoder_name="efficientnet_b0",
        encoder_weights="imagenet",
        pyramid_channels=64,
        num_fpn_layers=3,
        num_head_repeats=3,
        num_classes=90,
        drop_connect_rate=0,
        encoder_norm_layer="abn", # TODO: set to frozenabn when ready
        encoder_norm_act="swish",
        decoder_norm_layer="abn",
        decoder_norm_act="swish",
        match_tf_same_padding=False,
        ):
        super().__init__()
        self.encoder = get_encoder(
            encoder_name,
            norm_layer=encoder_norm_layer,
            norm_act=encoder_norm_act,
            encoder_weights=encoder_weights,
        )
        norm_layer = bn_from_name(decoder_norm_layer)
        bn_args = dict(norm_layer=norm_layer, norm_act=decoder_norm_act)
        self.pyramid6 = nn.Sequential(
            conv1x1(self.encoder.out_shapes[0], pyramid_channels, bias=True),
            norm_layer(pyramid_channels, activation="identity"),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.pyramid7 = nn.MaxPool2d(3, stride=2, padding=1) # in EffDet it's a simple maxpool

        self.bifpn = BiFPN(
            self.encoder.out_shapes[:-2],
            pyramid_channels=pyramid_channels,
            num_layers=num_fpn_layers,
            **bn_args,
        )

        def make_head(out_size):
            layers = []
            for _ in range(num_head_repeats):
                # TODO: add drop connect
                layers += [DepthwiseSeparableConv(pyramid_channels, pyramid_channels, use_norm=False)]
            layers += [DepthwiseSeparableConv(pyramid_channels, out_size, use_norm=False)]
            return nn.ModuleList(layers)
        
        def make_head_norm():
            return nn.ModuleList([
                nn.ModuleList(
                    [
                        norm_layer(pyramid_channels, activation=decoder_norm_act)
                        for _ in range(num_head_repeats)
                    ] + [nn.Identity()] # no bn after last depthwise conv
                )
                for _ in range(5)
            ])
        anchors_per_location = 9 # TODO: maybe allow to pass this arg?
        self.cls_head_convs = make_head(num_classes * anchors_per_location)
        self.cls_head_norms = make_head_norm()
        self.box_head_convs = make_head(4 * anchors_per_location)
        self.box_head_norms = make_head_norm()
        self.num_classes = num_classes

        patch_bn(self)
        if match_tf_same_padding:
            conv_to_same_conv(self)
            maxpool_to_same_maxpool(self)
            

    def forward(self, x):
        # don't use p2 and p1
        p5, p4, p3, _, _ = self.encoder(x)
        # coarser FPN levels
        p6 = self.pyramid6(p5)
        p7 = self.pyramid7(p6)
        features = [p7, p6, p5, p4, p3]
        # enhance features
        features = self.bifpn(features)
        # want features from lowest OS to highest to align with `generate_anchors_boxes` function 
        features = list(reversed(features))
        class_outputs = []
        box_outputs = []
        for feat, (cls_bns, box_bns) in zip(features, zip(self.cls_head_norms, self.box_head_norms)):
            cls_feat, box_feat = feat, feat
            for cls_conv, cls_bn in zip(self.cls_head_convs, cls_bns):
                cls_feat = cls_bn(cls_conv(cls_feat))
            for box_conv, box_bn in zip(self.box_head_convs, box_bns):
                box_feat = box_bn(box_conv(box_feat))

            box_feat = box_feat.permute(0, 2, 3, 1)
            box_outputs.append(box_feat.contiguous().view(box_feat.shape[0], -1, 4))

            cls_feat = cls_feat.permute(0, 2, 3, 1)
            class_outputs.append(cls_feat.contiguous().view(cls_feat.shape[0], -1, self.num_classes))

            # TODO: return back to simplier transpose operations
            # class_outputs.append(cls_feat.transpose(1, 3).contiguous().view(x.shape[0], -1, self.num_classes))
            # box_outputs.append(box_feat.transpose(1, 3).contiguous().view(x.shape[0], -1, 4))

        class_outputs = torch.cat(class_outputs , 1)
        box_outputs = torch.cat(box_outputs, 1)
        return class_outputs, box_outputs



def efficientdet_b0(**kwargs):
    return EfficientDet(**kwargs)

def efficientdet_b1(**kwargs):
    return EfficientDet(
        encoder_name="efficientnet_b1", pyramid_channels=88, num_fpn_layers=4, num_head_repeats=3, **kwargs)

def efficientdet_b2(**kwargs):
    return EfficientDet(encoder_name="efficientnet_b2", pyramid_channels=112, num_fpn_layers=5, num_head_repeats=3, **kwargs)

def efficientdet_b3(**kwargs):
    return EfficientDet(encoder_name="efficientnet_b3", pyramid_channels=160, num_fpn_layers=6, num_head_repeats=4, **kwargs)

def efficientdet_b4(**kwargs):
    return EfficientDet(encoder_name="efficientnet_b4", pyramid_channels=224, num_fpn_layers=7, num_head_repeats=4, **kwargs)
    
def efficientdet_b5(**kwargs):
    return EfficientDet(encoder_name="efficientnet_b5", pyramid_channels=288, num_fpn_layers=7, num_head_repeats=4, **kwargs)

def efficientdet_b6(*args, **kwargs):
    return EfficientDet(encoder_name="efficientnet_b6", pyramid_channels=384, num_fpn_layers=8, num_head_repeats=5, **kwargs)

# No B7 because it's the same model as B6 but with larger input