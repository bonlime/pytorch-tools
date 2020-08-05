"""dev file for my models"""

import logging
from copy import deepcopy
from collections import OrderedDict
from functools import wraps, partial

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

# from pytorch_tools.modules import BasicBlock, Bottleneck
from pytorch_tools.modules import GlobalPool2d, BlurPool
from pytorch_tools.modules.residual import conv1x1, conv3x3
from pytorch_tools.modules.residual import DarkStage, DarkBasicBlock
from pytorch_tools.modules.pooling import FastGlobalAvgPool2d
from pytorch_tools.modules import bn_from_name
from pytorch_tools.modules import SpaceToDepth
from pytorch_tools.modules import conv_to_ws_conv
from pytorch_tools.utils.misc import initialize
from pytorch_tools.utils.misc import add_docs_for
from pytorch_tools.utils.misc import DEFAULT_IMAGENET_SETTINGS
from pytorch_tools.utils.misc import repeat_channels


class DarkNet(nn.Module):
    def __init__(
        self,
        block=None,
        layers=None,
        pretrained=None,  # not used. here for proper signature
        num_classes=1000,
        in_channels=3,
        attn_type=None,
        # base_width=64,
        stem_type="",
        norm_layer="abn",
        norm_act="relu",
        antialias=False,
        # encoder=False,
        drop_rate=0.0,
        drop_connect_rate=0.0,
    ):

        stem_width = 64
        norm_layer = bn_from_name(norm_layer)
        self.num_classes = num_classes
        self.norm_act = norm_act
        self.block_idx = 0  # for drop connect
        self.drop_connect_rate = drop_connect_rate
        super().__init__()

        # instead of default stem I'm using 2 Space2Depth followed by conv. it's faster than default and works better
        self.stem_conv1 = nn.Sequential(
            SpaceToDepth(block_size=2),
            conv3x3(in_channels * 4, stem_width // 4),
            norm_layer(stem_width // 4, activation=norm_act),
        )
        self.stem_conv2 = nn.Sequential(
            SpaceToDepth(block_size=2),
            conv3x3(stem_width, stem_width),
            norm_layer(stem_width, activation=norm_act),
        )

        # blocks
        largs = dict(
            stride=2,
            bottle_ratio=0.5,
            block_fn=block,
            attn_type=attn_type,
            norm_layer=norm_layer,
            norm_act=norm_act,
            antialias=antialias,
        )
        # fmt: off
        self.layer1 = DarkStage(in_channels=stem_width, out_channels=64, num_blocks=1, **largs)
        self.layer2 = DarkStage(in_channels=64, out_channels=128, num_blocks=2, keep_prob=self.keep_prob, **largs)
        self.layer3 = DarkStage(in_channels=128, out_channels=256, num_blocks=8, keep_prob=self.keep_prob, **largs)
        self.layer4 = DarkStage(in_channels=256, out_channels=512, num_blocks=8, keep_prob=self.keep_prob, **largs)
        self.layer5 = DarkStage(in_channels=512, out_channels=1024, num_blocks=4, keep_prob=self.keep_prob, **largs)
        # fmt: on

        self.global_pool = FastGlobalAvgPool2d(flatten=True)
        self.dropout = nn.Dropout(p=drop_rate, inplace=True)
        self.last_linear = nn.Linear(1024, num_classes)
        initialize(self)

    def features(self, x):
        x = self.stem_conv1(x)
        x = self.stem_conv2(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    @property
    def keep_prob(self):
        # in ResNet it increases every block but here it increases every stage
        keep_prob = 1 - self.drop_connect_rate * self.block_idx / 5
        self.block_idx += 1
        return keep_prob


# fmt: off
CFGS = {
    "darknet53": {
        "default": {"params": {"block": DarkBasicBlock, "layers": [1, 2, 8, 8, 4]}, **DEFAULT_IMAGENET_SETTINGS},
    },
}
# fmt: on


def _darknet(arch, pretrained=None, **kwargs):
    cfgs = deepcopy(CFGS)
    cfg_settings = cfgs[arch]["default"]
    cfg_params = cfg_settings.pop("params")
    if pretrained:
        pretrained_settings = cfgs[arch][pretrained]
        pretrained_params = pretrained_settings.pop("params", {})
        cfg_settings.update(pretrained_settings)
        cfg_params.update(pretrained_params)
    common_args = set(cfg_params.keys()).intersection(set(kwargs.keys()))
    if common_args:
        logging.warning(
            f"Args {common_args} are going to be overwritten by default params for {pretrained} weights"
        )
    kwargs.update(cfg_params)
    model = DarkNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(cfgs[arch][pretrained]["url"])
        kwargs_cls = kwargs.get("num_classes", None)
        if kwargs_cls and kwargs_cls != cfg_settings["num_classes"]:
            logging.warning(
                "Using model pretrained for {} classes with {} classes. Last layer is initialized randomly".format(
                    cfg_settings["num_classes"], kwargs_cls
                )
            )
            # if there is last_linear in state_dict, it's going to be overwritten
            state_dict["fc.weight"] = model.state_dict()["last_linear.weight"]
            state_dict["fc.bias"] = model.state_dict()["last_linear.bias"]
        # support pretrained for custom input channels
        if kwargs.get("in_channels", 3) != 3:
            # TODO: add
            pass
        model.load_state_dict(state_dict)
        if cfg_settings.get("weight_standardization"):
            # convert to ws implicitly. maybe need a logging warning here?
            model = conv_to_ws_conv(model)
    setattr(model, "pretrained_settings", cfg_settings)
    return model


@wraps(DarkNet)
def darknet53(**kwargs):
    r"""Constructs a ResNet-18 model."""
    return _darknet("darknet53", **kwargs)
