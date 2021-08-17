"""dev file for my models"""

import logging
from copy import deepcopy
from collections import OrderedDict
from functools import wraps, partial

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

# from pytorch_tools.modules import BasicBlock, Bottleneck
import pytorch_tools as pt
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
from pytorch_tools.modules.residual import SimpleBasicBlock
from pytorch_tools.modules.residual import SimpleBottleneck
from pytorch_tools.modules.residual import SimpleInvertedResidual
from pytorch_tools.modules.residual import SimplePreActBasicBlock
from pytorch_tools.modules.residual import SimplePreActRes2BasicBlock
from pytorch_tools.modules.residual import SimplePreActBottleneck
from pytorch_tools.modules.residual import SimplePreActInvertedResidual
from pytorch_tools.modules.residual import SimpleSeparable_2
from pytorch_tools.modules.residual import SimplePreActSeparable_2
from pytorch_tools.modules.residual import SimpleSeparable_3
from pytorch_tools.modules.residual import PreBlock_2


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
        expand_before_head=True,  # add addition conv from 512 -> 2048 to avoid representational bottleneck
        mobilenetv3_head=False,  # put GAP first, then expand convs
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
            # antialias=antialias,
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
        # this is a very dirty if but i don't care for now
        if mobilenetv3_head:
            head_layers.append(FastGlobalAvgPool2d(flatten=True))
            if channels[3] < 2048 and expand_before_head:
                head_layers.append(nn.Linear(channels[3], 2048))  # no norm here as in original MobilnetV3 from google
                head_layers.append(pt.modules.activations.activation_from_name(norm_act))
            head_layers.append(nn.Linear(2048 if expand_before_head else channels[3], num_classes))
        else:
            if channels[3] < 2048 and expand_before_head:
                if block_fn == SimplePreActBottleneck:  # for PreAct add additional BN here
                    head_layers.append(norm_layer(channels[3], activation=norm_act))
                head_layers.extend([conv1x1(channels[3], 2048), norm_layer(2048, activation=norm_act)])
            head_layers.extend(
                [FastGlobalAvgPool2d(flatten=True), nn.Linear(2048 if expand_before_head else channels[3], num_classes)]
            )
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


class BNet(nn.Module):  # copied from DarkNet not to break backward compatability
    def __init__(
        self,
        stage_fns=None,  # list of nn.Module
        block_fns=None,  # list of nn.Module
        stage_args=None,  # list of dicts
        layers=None,  # num layers in each block
        channels=None,  # it's actually output channels. 256, 512, 1024, 2048 for R50
        # pretrained=None,  # not used. here for proper signature
        num_classes=1000,
        in_channels=3,
        norm_layer="abn",
        norm_act="leaky_relu",
        head_norm_act="leaky_relu",  # activation in head
        stem_type="default",
        # antialias=False,
        # encoder=False,
        # drop_rate=0.0,
        drop_connect_rate=0.0,
        head_width=2048,
        stem_width=64,
        head_type="default",  # type of head
    ):
        norm_layer = bn_from_name(norm_layer)
        self.num_classes = num_classes
        self.norm_act = norm_act
        self.block_idx = 0  # for drop connect
        self.drop_connect_rate = drop_connect_rate
        super().__init__()

        first_norm = nn.Identity() if block_fns[0].startswith("Pre") else norm_layer(stem_width, activation=norm_act)
        if stem_type == "default":
            self.stem_conv1 = nn.Sequential(conv3x3(in_channels, stem_width, stride=2), first_norm)
        elif stem_type == "s2d":
            # instead of default stem I'm using Space2Depth followed by conv. no norm because there is one at the beginning
            # of DarkStage. upd. there is norm in not PreAct version
            self.stem_conv1 = nn.Sequential(
                SpaceToDepth(block_size=2),
                conv3x3(in_channels * 4, stem_width),
                first_norm,
            )
        else:
            raise ValueError(f"Stem type `{stem_type}` is not supported")

        bn_args = dict(norm_layer=norm_layer, norm_act=norm_act)
        block_name_to_module = {
            "XX": SimpleBasicBlock,
            "Pre_XX": SimplePreActBasicBlock,
            "Pre_XX_Res2": SimplePreActRes2BasicBlock,
            "Btl": SimpleBottleneck,
            "Pre_Btl": SimplePreActBottleneck,
            "IR": SimpleInvertedResidual,
            "Pre_IR": SimplePreActInvertedResidual,
            "Sep2": SimpleSeparable_2,
            "Pre_Sep2": SimplePreActSeparable_2,
            "Sep3": SimpleSeparable_3,
            "Pre_Custom_2": PreBlock_2,
        }
        stage_name_to_module = {"simpl": SimpleStage}
        # set stride=2 for all blocks
        # using **{**bn_args, **stage_args} to allow updating norm layer for particular stage
        self.layer1 = stage_name_to_module[stage_fns[0]](
            block_fn=block_name_to_module[block_fns[0]],
            in_chs=stem_width,
            out_chs=channels[0],
            num_blocks=layers[0],
            stride=2,
            **{**bn_args, **stage_args[0]},
        )
        self.layer2 = stage_name_to_module[stage_fns[1]](
            block_fn=block_name_to_module[block_fns[1]],
            in_chs=channels[0],
            out_chs=channels[1],
            num_blocks=layers[1],
            stride=2,
            **{**bn_args, **stage_args[1]},
        )
        self.layer3 = stage_name_to_module[stage_fns[2]](
            block_fn=block_name_to_module[block_fns[2]],
            in_chs=channels[1],
            out_chs=channels[2],
            num_blocks=layers[2],
            stride=2,
            **{**bn_args, **stage_args[2]},
        )
        extra_stage3_filters = stage_args[2].get("filter_steps", 0) * (layers[2] - 1)
        self.layer4 = stage_name_to_module[stage_fns[3]](
            block_fn=block_name_to_module[block_fns[3]],
            in_chs=channels[2] + extra_stage3_filters,
            out_chs=channels[3],
            num_blocks=layers[3],
            stride=2,
            **{**bn_args, **stage_args[3]},
        )
        extra_stage4_filters = stage_args[3].get("filter_steps", 0) * (layers[3] - 1)
        channels[3] += extra_stage4_filters  # TODO rewrite it cleaner instead of doing inplace
        last_norm = norm_layer(channels[3], activation=norm_act) if block_fns[0].startswith("Pre") else nn.Identity()
        if head_type == "mobilenetv3":
            self.head = nn.Sequential(  # Mbln v3 head. GAP first, then expand convs
                last_norm,
                FastGlobalAvgPool2d(flatten=True),
                nn.Linear(channels[3], head_width),
                pt.modules.activations.activation_from_name(head_norm_act),
            )
            self.last_linear = nn.Linear(head_width, num_classes)
        elif head_type == "mobilenetv3_norm":  # mobilenet with last norm
            self.head = nn.Sequential(  # Mbln v3 head. GAP first, then expand convs
                last_norm,
                FastGlobalAvgPool2d(flatten=True),
                nn.Linear(channels[3], head_width),
                nn.BatchNorm1d(head_width),
                pt.modules.activations.activation_from_name(head_norm_act),
            )
            self.last_linear = nn.Linear(head_width, num_classes)
        elif head_type == "default":
            self.head = nn.Sequential(
                last_norm,
                conv1x1(channels[3], head_width),
                norm_layer(head_width, activation=head_norm_act),
                FastGlobalAvgPool2d(flatten=True),
            )
            self.last_linear = nn.Linear(head_width, num_classes)
        elif head_type == "default_nonorm":  # if used in angular losses don't want norm
            self.head = nn.Sequential(
                last_norm,
                conv1x1(channels[3], head_width, bias=True),  # need bias because not followed by norm
                FastGlobalAvgPool2d(flatten=True),
            )
            self.last_linear = nn.Linear(head_width, num_classes)
        elif head_type == "mlp_bn_fc_bn":
            self.head = nn.Sequential(
                last_norm,
                conv1x1(channels[3], channels[3]),
                FastGlobalAvgPool2d(flatten=True),
                nn.BatchNorm1d(channels[3]),
                pt.modules.activations.activation_from_name(head_norm_act),
                nn.Linear(channels[3], head_width, bias=False),
                nn.BatchNorm1d(head_width, affine=False),
            )
            self.last_linear = nn.Linear(head_width, num_classes)
        elif head_type == "mlp_bn_fc":  # same as above but without last BN
            self.head = nn.Sequential(
                last_norm,
                conv1x1(channels[3], channels[3]),
                FastGlobalAvgPool2d(flatten=True),
                nn.BatchNorm1d(channels[3]),
                pt.modules.activations.activation_from_name(head_norm_act),
                nn.Linear(channels[3], head_width, bias=False),
            )
            self.last_linear = nn.Linear(head_width, num_classes)
        elif head_type == "mlp_2":
            assert isinstance(head_width, (tuple, list)), head_width
            self.head = nn.Sequential(  # like Mbln v3 head. GAP first, then MLP convs
                last_norm,
                FastGlobalAvgPool2d(flatten=True),
                nn.Linear(channels[3], head_width[0]),
                nn.BatchNorm1d(head_width[0]),
                pt.modules.activations.activation_from_name(head_norm_act),
                nn.Linear(head_width[0], head_width[1]),
                nn.BatchNorm1d(head_width[1]),
                pt.modules.activations.activation_from_name(head_norm_act),
            )
            self.last_linear = nn.Linear(head_width[1], num_classes)
        elif head_type == "mlp_3":
            assert isinstance(head_width, (tuple, list)), head_width
            self.head = nn.Sequential(  # like Mbln v3 head. GAP first, then MLP convs
                last_norm,
                FastGlobalAvgPool2d(flatten=True),
                nn.Linear(channels[3], head_width[0]),
                nn.BatchNorm1d(head_width[0]),
                pt.modules.activations.activation_from_name(head_norm_act),
                nn.Linear(head_width[0], head_width[1]),
                nn.BatchNorm1d(head_width[1]),
                pt.modules.activations.activation_from_name(head_norm_act),
                nn.Linear(head_width[1], head_width[2]),
                nn.BatchNorm1d(head_width[2]),
                pt.modules.activations.activation_from_name(head_norm_act),
            )
            self.last_linear = nn.Linear(head_width[2], num_classes)
        else:
            raise ValueError(f"Head type: {head_type} is not supported!")
        initialize(self)

    def features(self, x):
        x = self.stem_conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
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
    "simpl_dark": {
        "default": {
            "params": {
                "stage_fn": SimpleStage,
                "block_fn": SimpleBasicBlock,
                "layers": [3, 4, 6, 3],
                "channels": [64, 128, 256, 512],
                "bottle_ratio": 1,
            },
            **DEFAULT_IMAGENET_SETTINGS
        },
    },
    "csp_simpl_dark": {
        "default": {
            "params": {
                "stage_fn": CrossStage,
                "block_fn": SimpleBasicBlock,
                "layers": [3, 4, 6, 3],
                "channels": [64, 128, 256, 512],
                "bottle_ratio": 1,
            },
            **DEFAULT_IMAGENET_SETTINGS
        },
    },
}

GNET_CFGS = {
    "gnet_normal_my": {
        "default": {
            "params": {
                "stage_fns": [SimpleStage, SimpleStage, SimpleStage, SimpleStage],
                "block_fns": ["XX", "XX", "Btl", "IR"],
                "stage_args": [
                    {},
                    {},
                    {"bottle_ratio": 0.25},
                    {"expand_ratio": 3},
                ],
                "layers": [1, 2, 6, 5],
                "channels": [128, 192, 640, 640],
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


def _gnet_like(arch, **kwargs):
    cfgs = deepcopy(GNET_CFGS)
    cfg_settings = cfgs[arch]["default"]
    cfg_params = cfg_settings.pop("params")
    kwargs.update(cfg_params)
    model = BNet(**kwargs)
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


@wraps(DarkNet)
def csp_simpl_dark(**kwargs):
    return _darknet("csp_simpl_dark", **kwargs)


@wraps(DarkNet)
def simpl_dark(**kwargs):
    return _darknet("simpl_dark", **kwargs)


def gnet_normal_my(**kwargs):
    return _gnet_like("gnet_normal_my", **kwargs)
