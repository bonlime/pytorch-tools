import torch
import torch.nn as nn

from pytorch_tools.modules.fpn import FPN
from pytorch_tools.modules import bn_from_name
from pytorch_tools.modules.residual import conv3x3
from pytorch_tools.segmentation_models.encoders import get_encoder


class RetinaNet(nn.Module):
    """RetinaNet 
    Main difference from other implementations are:
    * support of any custom encoder from this repo
    * optional normalization layer in box classification head
    * ability to freeze batch norm in encoder with one line
    
    Args:
        encoder_name (str): name of classification model (without last dense layers) used as feature
            extractor to build detection model
        encoder_weights (str): one of ``None`` (random initialization), ``imagenet`` (pre-trained on ImageNet)
        pyramid_channels (int): size of features after FPN. Default 256
        num_classes (int): a number of classes to predict 
            class_outputs shape is (BS, *, NUM_CLASSES) where each row in * corresponds to one bbox
        encoder_norm_layer (str): Normalization layer to use in encoder. If using pretrained
            it should be the same as in pretrained weights
        encoder_norm_act (str): Activation for normalization layer in encoder
        decoder_norm_layer (str): Normalization to use in head convolutions. Default (none) is not to use normalization. 
            Current implementation is optimized for `GroupNorm`, not `BatchNorm` check code for details
        decoder_norm_act (str): Activation for normalization layer in head convolutions
    
    Ref:
        Focal Loss for Dense Object Detection - https://arxiv.org/abs/1708.02002
        Mmdetection - https://github.com/open-mmlab/mmdetection/ (at commit b9daf23)
        TF TPU version - https://github.com/tensorflow/tpu/tree/master/models/official/retinanet
    """
    def __init__(
        self,
        encoder_name="resnet50",
        encoder_weights="imagenet",
        pyramid_channels=256,
        num_classes=80,
        # drop_connect_rate=0, # TODO: add 
        encoder_norm_layer="abn",
        encoder_norm_act="relu",
        decoder_norm_layer="none", # None by default to match detectron & mmdet versions
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
            conv3x3(self.encoder.out_shapes[0], pyramid_channels, 2, bias=True),
            norm_layer(pyramid_channels, activation="identity"),
        )
        self.pyramid7 = nn.Sequential(
            conv3x3(pyramid_channels, pyramid_channels, 2, bias=True),
            norm_layer(pyramid_channels, activation="identity"),
        )
        self.fpn = FPN(self.encoder.out_shapes[:-2], pyramid_channels=pyramid_channels)

        def make_final_convs():
            layers = []
            for _ in range(4):
                layers += [conv3x3(pyramid_channels, pyramid_channels, bias=True)]
                # Norm here is fine for GroupNorm but for BN it should be implemented the other way
                # see EffDet for example. Maybe need to change this implementation to align with EffDet
                layers += [norm_layer(pyramid_channels, activation=decoder_norm_act)]
            return nn.Sequential(*layers)

        anchors_per_location = 9
        self.cls_convs = make_final_convs()
        self.cls_head_conv = conv3x3(pyramid_channels, num_classes * anchors_per_location, bias=True)
        self.box_convs = make_final_convs()
        self.box_head_conv = conv3x3(pyramid_channels, 4 * anchors_per_location, bias=True)
        self.num_classes = num_classes

    # Name from mmdetectin for convenience
    def extract_features(self, x):
        """Extract features from backbone + enchance with FPN"""
        # don't use p2 and p1
        p5, p4, p3, _, _ = self.encoder(x)
        # coarser FPN levels
        p6 = self.pyramid6(p5)
        p7 = self.pyramid7(p6.relu()) # in mmdet there is no relu here. but i think it's needed
        # enhance features
        p5, p4, p3 = self.fpn([p5, p4, p3])
        # want features from lowest OS to highest to align with `generate_anchors_boxes` function
        features = [p3, p4, p5, p6, p7]
        return features

    def forward(self, x):
        features = self.extract_features(x)
        class_outputs = []
        box_outputs = []
        for feat in features:
            cls_feat = self.cls_head_conv(self.cls_convs(feat))
            box_feat = self.box_head_conv(self.box_convs(feat))
            cls_feat = cls_feat.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
            box_feat = box_feat.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
            class_outputs.append(cls_feat)
            box_outputs.append(box_feat)
        class_outputs = torch.cat(class_outputs, 1)
        box_outputs = torch.cat(box_outputs, 1)
        return class_outputs, box_outputs
