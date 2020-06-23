"""
Implementation of EfficientDet object detection network
Reference: EfficientDet: Scalable and Efficient Object Detection - https://arxiv.org/abs/1911.09070
This version supports init from pretrained weights and easy changing of many hyper-parameters
hacked by @bonlime
"""
import logging
from copy import deepcopy
from functools import wraps

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from pytorch_tools.modules import ABN
from pytorch_tools.modules.bifpn import BiFPN
from pytorch_tools.modules import bn_from_name
from pytorch_tools.modules.residual import conv1x1
from pytorch_tools.modules.residual import DepthwiseSeparableConv
from pytorch_tools.modules.tf_same_ops import conv_to_same_conv
from pytorch_tools.modules.tf_same_ops import maxpool_to_same_maxpool

from pytorch_tools.segmentation_models.encoders import get_encoder

import pytorch_tools.utils.box as box_utils
from pytorch_tools.utils.misc import DEFAULT_IMAGENET_SETTINGS
from pytorch_tools.utils.misc import initialize_iterator
from pytorch_tools.models.efficientnet import patch_bn_tf


class EfficientDet(nn.Module):
    """
    Implementation of the EfficientDet Object Detection model

    Main difference from other implementations available are:
    * cleanest code. all model is defined in this file with only small modules
        imported from somewhere else
    * ability to freeze batch norm in encoder with one line
    * fast train speed and low memory consumption. partly due to memory efficient Swish
        partly due to heavy use of inplace operations
    
    Args:
        pretrained (str): one of `coco` or None. if `coco` - load pretrained weights
        encoder_name (str): name of classification model (without last dense layers) used as feature
                extractor to build detection model. It could be any model even `resnet`
        encoder_weights (str): one of ``None`` (random initialization), ``imagenet`` (pre-trained on ImageNet)
        pyramid_channels (int): size of features after BiFPN. Default 256
        num_fpn_layers (int): Number of BiFPN layers
        num_head_repeats (int): Number of convs layers in regression and classification heads
        num_classes (int): a number of classes to predict
                class_outputs shape is (BS, *, NUM_CLASSES) where each row in * corresponds to one bbox
        encoder_norm_layer (str): Normalization layer to use in encoder. If using pretrained
                it should be the same as in pretrained weights. By default batch norm is frozen in encoder
                pass `abn` not use not frozen version
        encoder_norm_act (str): Activation for normalization layer in encoder
        decoder_norm_layer (str): Normalization to use in head convolutions. Default (none) is not to use normalization.
                Current implementation is optimized for `GroupNorm`, not `BatchNorm` check code for details
        decoder_norm_act (str): Activation for normalization layer in head convolutions
        match_tf_same_padding (bool): If True patches Conv and MaxPool to implements tf-like asymmetric padding
            Should only be used to validate pretrained weights. Not needed for training. Gives ~10% slowdown
        anchors_per_location (int): Number of anchors per feature map pixel. In fact it only affects the size of output
        
    Ref:
        EfficientDet: Scalable and Efficient Object Detection - https://arxiv.org/abs/1911.09070
    """

    def __init__(
        self,
        pretrained="coco",  # Not used. here for proper signature
        encoder_name="efficientnet_b0",
        encoder_weights="imagenet",
        pyramid_channels=64,
        num_fpn_layers=3,
        num_head_repeats=3,
        num_classes=90,
        encoder_norm_layer="frozenabn",
        encoder_norm_act="swish",
        decoder_norm_layer="abn",
        decoder_norm_act="swish",
        match_tf_same_padding=False,
        anchors_per_location=9,
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
        bn_args = dict(norm_layer=norm_layer, norm_act=decoder_norm_act)
        self.pyramid6 = nn.Sequential(
            conv1x1(self.encoder.out_shapes[0], pyramid_channels, bias=True),
            norm_layer(pyramid_channels, activation="identity"),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.pyramid7 = nn.MaxPool2d(3, stride=2, padding=1)  # in EffDet it's a simple maxpool

        self.bifpn = BiFPN(
            encoder_channels=(pyramid_channels,) * 2 + self.encoder.out_shapes[:-2],
            pyramid_channels=pyramid_channels,
            num_layers=num_fpn_layers,
            **bn_args,
        )

        def make_head(out_size):
            layers = []
            for _ in range(num_head_repeats):
                layers += [DepthwiseSeparableConv(pyramid_channels, pyramid_channels, use_norm=False)]
            return nn.ModuleList(layers)

        # The convolution layers in the head are shared among all levels, but
        # each level has its batch normalization to capture the statistical
        # difference among different levels.
        def make_head_norm():
            return nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            norm_layer(pyramid_channels, activation=decoder_norm_act)
                            for _ in range(num_head_repeats)
                        ]
                    )
                    for _ in range(5)
                ]
            )

        self.cls_convs = make_head(num_classes * anchors_per_location)
        self.cls_head_conv = DepthwiseSeparableConv(
            pyramid_channels, num_classes * anchors_per_location, use_norm=False
        )
        self.cls_norms = make_head_norm()

        self.box_convs = make_head(4 * anchors_per_location)
        self.box_head_conv = DepthwiseSeparableConv(
            pyramid_channels, 4 * anchors_per_location, use_norm=False
        )
        self.box_norms = make_head_norm()

        self.num_classes = num_classes
        self.num_head_repeats = num_head_repeats

        patch_bn_tf(self)
        self._initialize_weights()
        if match_tf_same_padding:
            conv_to_same_conv(self)
            maxpool_to_same_maxpool(self)

    # Name from mmdetectin for convenience
    def extract_features(self, x):
        """Extract features from backbone + enchance with BiFPN"""
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
        return features

    def forward(self, x):
        features = self.extract_features(x)
        class_outputs = []
        box_outputs = []
        # for feat, (cls_bns, box_bns) in zip(features, zip(self.cls_head_norms, self.box_head_norms)):
        for feat, (cls_bns, box_bns) in zip(features, zip(self.cls_norms, self.box_norms)):
            cls_feat, box_feat = feat, feat
            # according to https://github.com/google/automl/issues/237
            # DropConnect is not used in FPN or head
            for cls_conv, cls_bn in zip(self.cls_convs, cls_bns):
                cls_feat = cls_bn(cls_conv(cls_feat))
            cls_feat = self.cls_head_conv(cls_feat)  # no bn after last cls conv

            for box_conv, box_bn in zip(self.box_convs, box_bns):
                box_feat = box_bn(box_conv(box_feat))
            box_feat = self.box_head_conv(box_feat)  # no bn after last box conv

            box_feat = box_feat.permute(0, 2, 3, 1)
            box_outputs.append(box_feat.contiguous().view(box_feat.shape[0], -1, 4))

            cls_feat = cls_feat.permute(0, 2, 3, 1)
            class_outputs.append(cls_feat.contiguous().view(cls_feat.shape[0], -1, self.num_classes))

        class_outputs = torch.cat(class_outputs, 1)
        box_outputs = torch.cat(box_outputs, 1)
        # my anchors are in [x1, y1, x2,y2] format while pretrained weights are in [y1, x1, y2, x2] format
        # it may be confusing to reorder x and y every time later so I do it once here. it gives
        # compatability with pretrained weights from Google and doesn't affect training from scratch
        box_outputs = box_outputs[..., [1, 0, 3, 2]]
        return class_outputs, box_outputs

    @torch.no_grad()
    def predict(self, x):
        """
        Run forward on given images and decode raw prediction into bboxes
		Returns:
            torch.Tensor with bboxes, scores and classes. bboxes in `lrtb` format
            shape [BS, MAX_DETECTION_PER_IMAGE, 6]
		"""
        class_outputs, box_outputs = self.forward(x)
        anchors = box_utils.generate_anchors_boxes(x.shape[-2:])[0]
        return box_utils.decode(class_outputs, box_outputs, anchors)

    def _initialize_weights(self):
        # init everything except encoder
        no_encoder_m = [m for n, m in self.named_modules() if not "encoder" in n]
        initialize_iterator(no_encoder_m)
        # need to init last bias so that after sigmoid it's 0.01
        cls_bias_init = -torch.log(torch.tensor((1 - 0.01) / 0.01))  # -4.59
        nn.init.constant_(self.cls_head_conv[1].bias, cls_bias_init)


PRETRAIN_SETTINGS = {**DEFAULT_IMAGENET_SETTINGS, "input_size": (512, 512), "crop_pct": 1, "num_classes": 90}

# fmt: off
CFGS = {
	"efficientdet_d0": {
		"default": {
			"params": {
				"encoder_name":"efficientnet_b0",
				"pyramid_channels":64,
				"num_fpn_layers":3,
				"num_head_repeats":3,
			},
			**PRETRAIN_SETTINGS,
		},
		"coco": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.5/effdet-d0-bdea4ff9.pth",},
	},
	"efficientdet_d1": {
		"default": {
			"params": {
				"encoder_name":"efficientnet_b1",
				"pyramid_channels":88,
				"num_fpn_layers":4,
				"num_head_repeats":3,
			},
			**PRETRAIN_SETTINGS,
			"input_size": (640, 640),
		},
		"coco": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.5/effdet-d1-503733a9.pth",},
	},
	"efficientdet_d2": {
		"default": {
			"params": {
				"encoder_name":"efficientnet_b2",
				"pyramid_channels":112,
				"num_fpn_layers":5,
				"num_head_repeats":3,
			},
			**PRETRAIN_SETTINGS,
			"input_size": (768, 768),
		},
		"coco": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.5/effdet-d2-e5ec6289.pth",},
	},
	"efficientdet_d3": {
		"default": {
			"params": {
				"encoder_name":"efficientnet_b3",
				"pyramid_channels":160,
				"num_fpn_layers":6,
				"num_head_repeats":4,
			},
			**PRETRAIN_SETTINGS,
			"input_size": (896, 896),
		},
		"coco": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.5/effdet-d3-f6e6d52c.pth",},
	},
	"efficientdet_d4": {
		"default": {
			"params": {
				"encoder_name":"efficientnet_b4",
				"pyramid_channels":224,
				"num_fpn_layers":7,
				"num_head_repeats":4,
			},
			**PRETRAIN_SETTINGS,
			"input_size": (1024, 1024),
		},
		"coco": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.5/effdet-d4-76ba1095.pth",},
	},
	"efficientdet_d5": {
		"default": {
			"params": {
				"encoder_name":"efficientnet_b5",
				"pyramid_channels":288,
				"num_fpn_layers":7,
				"num_head_repeats":4,
			},
			**PRETRAIN_SETTINGS,
			"input_size": (1280, 1280),
		},
		"coco": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.5/effdet-d5-52e946b7.pth",},
	},
	"efficientdet_d6": {
		"default": {
			"params": {
				"encoder_name":"efficientnet_b6",
				"pyramid_channels":384,
				"num_fpn_layers":8,
				"num_head_repeats":5,
			},
			**PRETRAIN_SETTINGS,
			"input_size": (1280, 1280),
		},
		"coco": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.5/effdet-d6-3b8c5be0.pth",},
	},
}
# fmt: on


def _efficientdet(arch, pretrained=None, **kwargs):
    cfgs = deepcopy(CFGS)
    cfg_settings = cfgs[arch]["default"]
    cfg_params = cfg_settings.pop("params")
    kwargs.update(cfg_params)
    model = EfficientDet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(cfgs[arch][pretrained]["url"])
        kwargs_cls = kwargs.get("num_classes", None)
        if kwargs_cls and kwargs_cls != cfg_settings["num_classes"]:
            logging.warning(
                f"Using model pretrained for {cfg_settings['num_classes']} classes with {kwargs_cls} classes. Last layer is initialized randomly"
            )
            state_dict["cls_head_conv.1.weight"] = model.state_dict()[f"cls_head_conv.1.weight"]
            state_dict["cls_head_conv.1.bias"] = model.state_dict()["cls_head_conv.1.bias"]
        model.load_state_dict(state_dict, strict=True)
    setattr(model, "pretrained_settings", cfg_settings)
    return model


@wraps(EfficientDet)
def efficientdet_d0(pretrained="coco", **kwargs):
    return _efficientdet("efficientdet_d0", pretrained, **kwargs)


@wraps(EfficientDet)
def efficientdet_d1(pretrained="coco", **kwargs):
    return _efficientdet("efficientdet_d1", pretrained, **kwargs)


@wraps(EfficientDet)
def efficientdet_d2(pretrained="coco", **kwargs):
    return _efficientdet("efficientdet_d2", pretrained, **kwargs)


@wraps(EfficientDet)
def efficientdet_d3(pretrained="coco", **kwargs):
    return _efficientdet("efficientdet_d3", pretrained, **kwargs)


@wraps(EfficientDet)
def efficientdet_d4(pretrained="coco", **kwargs):
    return _efficientdet("efficientdet_d4", pretrained, **kwargs)


@wraps(EfficientDet)
def efficientdet_d5(pretrained="coco", **kwargs):
    return _efficientdet("efficientdet_d5", pretrained, **kwargs)


@wraps(EfficientDet)
def efficientdet_d6(pretrained="coco", **kwargs):
    return _efficientdet("efficientdet_d6", pretrained, **kwargs)


# No B7 because it's the same model as B6 but with larger input
