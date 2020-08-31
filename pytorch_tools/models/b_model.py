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
from pytorch_tools.modules.pooling import FastGlobalAvgPool2d
from pytorch_tools.modules import bn_from_name
from pytorch_tools.modules import SpaceToDepth
from pytorch_tools.modules import conv_to_ws_conv
from pytorch_tools.utils.misc import initialize
from pytorch_tools.utils.misc import add_docs_for
from pytorch_tools.utils.misc import DEFAULT_IMAGENET_SETTINGS
from pytorch_tools.utils.misc import repeat_channels


from pytorch_tools.modules.residual import CrossStage
from pytorch_tools.modules.residual import SimpleStage
from pytorch_tools.modules.residual import SimpleBottleneck
from pytorch_tools.modules.residual import SimplePreActBottleneck


class DarkNet(nn.Module):
    def __init__(
        self,
        stage_fn=None,
        block_fn=None,
        layers=None,  # num layers in each block
        channels=None,  # it's actually output channels. 256, 512, 1024, 2048 for R50
        pretrained=None,  # not used. here for proper signature
        num_classes=1000,
        in_channels=3,
        attn_type=None,
        # base_width=64,
        stem_type="default",
        norm_layer="abn",
        norm_act="leaky_relu",
        antialias=False,
        # encoder=False,
        bottle_ratio=0.25,  # how much to shrink channels in bottleneck layer
        no_first_csp=False,  # make first stage a Simple Stage
        drop_rate=0.0,
        drop_connect_rate=0.0,
        expand_before_head=True, # add addition conv from 512 -> 2048 to avoid representational bottleneck
        **block_kwargs,
    ):

        stem_width = 64
        norm_layer = bn_from_name(norm_layer)
        self.num_classes = num_classes
        self.norm_act = norm_act
        self.block_idx = 0  # for drop connect
        self.drop_connect_rate = drop_connect_rate
        super().__init__()

        if block_fn != SimplePreActBottleneck:
            stem_norm = norm_layer(stem_width, activation=norm_act)
        else:
            stem_norm = nn.Identity()
        if stem_type == "default":
            self.stem_conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=7, stride=2, padding=3, bias=False),
                stem_norm,
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            first_stride = 1
        elif stem_type == "s2d":
            # instead of default stem I'm using Space2Depth followed by conv. no norm because there is one at the beginning
            # of DarkStage. upd. there is norm in not PreAct version
            self.stem_conv1 = nn.Sequential(
                SpaceToDepth(block_size=2),
                conv3x3(in_channels * 4, stem_width),
                stem_norm,
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            first_stride = 2

        # blocks
        largs = dict(
            stride=2,
            bottle_ratio=bottle_ratio,
            block_fn=block_fn,
            attn_type=attn_type,
            norm_layer=norm_layer,
            norm_act=norm_act,
            antialias=antialias,
            **block_kwargs,
        )
        first_stage_fn = SimpleStage if no_first_csp else stage_fn
        # fmt: off
        self.layer1 = first_stage_fn(
            in_chs=stem_width,
            out_chs=channels[0],
            num_blocks=layers[0],
            keep_prob=self.keep_prob,
            **{**largs, "stride": first_stride}, # overwrite default stride
        )
        # **{**largs, "antialias": False} # antialias in first stage is too expensive
        self.layer2 = stage_fn(in_chs=channels[0], out_chs=channels[1], num_blocks=layers[1], keep_prob=self.keep_prob, **largs)
        self.layer3 = stage_fn(in_chs=channels[1], out_chs=channels[2], num_blocks=layers[2], keep_prob=self.keep_prob, **largs)
        self.layer4 = stage_fn(in_chs=channels[2], out_chs=channels[3], num_blocks=layers[3], keep_prob=self.keep_prob, **largs)
        # fmt: on

        # self.global_pool = FastGlobalAvgPool2d(flatten=True)
        # self.dropout = nn.Dropout(p=drop_rate, inplace=True)
        head_layers = []
        if channels[3] < 2048 and expand_before_head:
            if block_fn == SimplePreActBottleneck:  # for PreAct add additional BN here
                head_layers.append(norm_layer(channels[3], activation=norm_act))
            head_layers.extend([conv1x1(channels[3], 2048), norm_layer(2048, activation=norm_act)])
        head_layers.extend([FastGlobalAvgPool2d(flatten=True), nn.Linear(2048 if expand_before_head else channels[3], num_classes)])
        # self.head = nn.Sequential(
        #     conv1x1(channels[3], 2048),
        #     norm_layer(activation=norm_act),
        #     # norm_layer(1024, activation=norm_act),
        #     FastGlobalAvgPool2d(flatten=True),
        #     nn.Linear(2048, num_classes),
        # )
        self.head = nn.Sequential(*head_layers)
        initialize(self)

    # def _make_stem(self, stem_type):

    # self.bn1 = norm_layer(stem_width, activation=norm_act)
    # self.maxpool = nn.Sequential(
    #     SpaceToDepth(block_size=2),
    #     conv1x1(stem_width * 4, stem_width),
    #     norm_layer(stem_width, activation=norm_act),
    # )
    # self.maxpool =

    def features(self, x):
        x = self.stem_conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.features(x)
        # x = self.global_pool(x)
        # x = self.dropout(x)
        x = self.head(x)
        return x

    @property
    def keep_prob(self):
        # in ResNet it increases every block but here it increases every stage
        keep_prob = 1 - self.drop_connect_rate * self.block_idx / 5
        self.block_idx += 1
        return keep_prob


# fmt: off
CFGS = {
    # "darknet53": {
        # "default": {"params": {"block": DarkBasicBlock, "layers": [1, 2, 8, 8, 4]}, **DEFAULT_IMAGENET_SETTINGS},
    # },
    "simpl_resnet50": {
        "default": {
            "params": {
                "stage_fn": SimpleStage,
                "block_fn": SimpleBottleneck,
                "layers": [3, 4, 6, 3],
                "channels": [256, 512, 1024, 2048],
                "bottle_ratio": 0.25,
            },
            **DEFAULT_IMAGENET_SETTINGS
        },
    },
    "simpl_resnet34": {
        "default": {
            "params": {
                "stage_fn": SimpleStage,
                "block_fn": SimpleBottleneck,
                "layers": [3, 4, 6, 3],
                "channels": [64, 128, 256, 512],
                "bottle_ratio": 1,
            },
            **DEFAULT_IMAGENET_SETTINGS
        },
    },
    "simpl_preactresnet34": {
        "default": {
            "params": {
                "stage_fn": SimpleStage,
                "block_fn": SimplePreActBottleneck,
                "layers": [3, 4, 6, 3],
                "channels": [64, 128, 256, 512],
                "bottle_ratio": 1,
            },
            **DEFAULT_IMAGENET_SETTINGS
        },
    },
    "csp_simpl_resnet34": {
        "default": {
            "params": {
                "stage_fn": CrossStage,
                "block_fn": SimpleBottleneck,
                "layers": [3, 4, 6, 3],
                "channels": [64, 128, 256, 512],
                "bottle_ratio": 1,
            },
            **DEFAULT_IMAGENET_SETTINGS
        },
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
def simpl_resnet50(**kwargs):
    return _darknet("simpl_resnet50", **kwargs)


@wraps(DarkNet)
def simpl_resnet34(**kwargs):
    return _darknet("simpl_resnet34", **kwargs)


@wraps(DarkNet)
def simpl_preactresnet34(**kwargs):
    return _darknet("simpl_preactresnet34", **kwargs)


@wraps(DarkNet)
def csp_simpl_resnet34(**kwargs):
    return _darknet("csp_simpl_resnet34", **kwargs)

