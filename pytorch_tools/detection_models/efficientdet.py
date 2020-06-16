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
from pytorch_tools.modules.residual import conv3x3
from pytorch_tools.modules.residual import DepthwiseSeparableConv
from pytorch_tools.modules.tf_same_ops import conv_to_same_conv
from pytorch_tools.modules.tf_same_ops import maxpool_to_same_maxpool

from pytorch_tools.segmentation_models.encoders import get_encoder

import pytorch_tools.utils.box as box_utils
from pytorch_tools.utils.misc import DEFAULT_IMAGENET_SETTINGS
from pytorch_tools.utils.misc import initialize_iterator


def patch_bn(module):
    """TF ported weights use slightly different eps in BN. Need to adjust for better performance"""
    if isinstance(module, ABN):
        module.eps = 1e-3
        module.momentum = 1e-2
    for m in module.children():
        patch_bn(m)


class EfficientDet(nn.Module):
    """TODO: add docstring"""

    def __init__(
        self,
        pretrained="coco",  # Not used. here for proper signature
        encoder_name="efficientnet_d0",
        encoder_weights="imagenet",
        pyramid_channels=64,
        num_fpn_layers=3,
        num_head_repeats=3,
        num_classes=90,
        drop_connect_rate=0,
        encoder_norm_layer="abn",  # TODO: set to frozenabn when ready
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
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.pyramid7 = nn.MaxPool2d(3, stride=2, padding=1)  # in EffDet it's a simple maxpool

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
                        + [nn.Identity()]  # no bn after last depthwise conv
                    )
                    for _ in range(5)
                ]
            )

        anchors_per_location = 9  # TODO: maybe allow to pass this arg?
        self.cls_head_convs = make_head(num_classes * anchors_per_location)
        self.cls_head_norms = make_head_norm()
        self.box_head_convs = make_head(4 * anchors_per_location)
        self.box_head_norms = make_head_norm()
        self.num_classes = num_classes
        self.num_head_repeats = num_head_repeats

        patch_bn(self)
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
        for feat, (cls_bns, box_bns) in zip(features, zip(self.cls_head_norms, self.box_head_norms)):
            cls_feat, box_feat = feat, feat
            # according to https://github.com/google/automl/issues/237
            # DropConnect is not used in FPN or head
            for cls_conv, cls_bn in zip(self.cls_head_convs, cls_bns):
                cls_feat = cls_bn(cls_conv(cls_feat))
            for box_conv, box_bn in zip(self.box_head_convs, box_bns):
                box_feat = box_bn(box_conv(box_feat))

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
        """Run forward on given images and decode raw prediction into bboxes
        Returns: bboxes, scores, classes
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
        nn.init.constant_(self.cls_head_convs[-1][1].bias, cls_bias_init)


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
    "coco": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.5/efficientdet-d0.pth",},
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
    "coco": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.5/efficientdet-d1.pth",},
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
    "coco": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.5/efficientdet-d2.pth",},
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
    "coco": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.5/efficientdet-d3.pth",},
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
    "coco": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.5/efficientdet-d4.pth",},
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
    "coco": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.5/efficientdet-d5.pth",},
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
    "coco": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.5/efficientdet-d6.pth",},
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
            last_conv_name = f"cls_head_convs.{kwargs['num_head_repeats']}.1"
            state_dict[f"{last_conv_name}.weight"] = model.state_dict()[f"{last_conv_name}.weight"]
            state_dict[f"{last_conv_name}.bias"] = model.state_dict()[f"{last_conv_name}.bias"]
        model.load_state_dict(state_dict)
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
