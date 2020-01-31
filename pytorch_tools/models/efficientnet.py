# this implementation is based on two: rw , l... , TF, tpu.

import torch
from torch import nn
from torch.nn import functional as F
import math
from copy import deepcopy
from functools import wraps, partial
import re
from torchvision.models.utils import load_state_dict_from_url
import logging

# from .utils import (
#     round_filters,
#     round_repeats,
#     drop_connect,
#     get_same_padding_conv2d,
#     get_model_params,
#     efficientnet_params,
#     load_pretrained_weights,
# Swish,
# MemoryEfficientSwish,
# )
# from pytorch_tools.modules.activations import Swish
# MemoryEfficientSwish = Swish

from pytorch_tools.modules.residual import InvertedResidual, DepthwiseSeparableConv
from pytorch_tools.modules import bn_from_name
from pytorch_tools.modules.residual import conv1x1, conv3x3
from pytorch_tools.utils.misc import add_docs_for
from pytorch_tools.utils.misc import make_divisible
from pytorch_tools.utils.misc import DEFAULT_IMAGENET_SETTINGS

# avoid overwriting doc string
wraps = partial(wraps, assigned=("__module__", "__name__", "__qualname__", "__annotations__"))


class EfficientNet(nn.Module):
    """TODO: Add docs"""

    def __init__(
        self,
        blocks_args=None,
        width_multiplier=None,
        depth_multiplier=None,
        pretrained=None,  # not used. here for proper signature
        num_classes=1000,
        in_channels=3,
        drop_rate=0,
        drop_connect_rate=0,
        stem_size=32,
        norm_layer="abn",
        norm_act="relu",
    ):
        super().__init__()
        norm_layer = bn_from_name(norm_layer)
        self.norm_layer = norm_layer
        self.norm_act = norm_act
        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        # if pretrained: norm_layer = partial( + eps + default momentum) TODO:
        stem_size = make_divisible(stem_size * width_multiplier)
        self.conv_stem = conv3x3(in_channels, stem_size, stride=2)
        self.bn1 = norm_layer(stem_size, activation=norm_act)
        in_channels = stem_size
        self.blocks = nn.ModuleList([])

        for block_idx, block_arg in enumerate(blocks_args):
            block = []
            block_arg["in_channels"] = make_divisible(block_arg["in_channels"] * self.width_multiplier)
            block_arg["out_channels"] = make_divisible(block_arg["out_channels"] * self.width_multiplier)
            block_arg["keep_prob"] = 1 - drop_connect_rate * block_idx / len(
                blocks_args
            )  # linearly scale keep prob
            repeats = block_arg.pop("num_repeat")
            repeats = int(math.ceil(repeats * self.depth_multiplier))
            # only first layer in block is strided
            block.append(InvertedResidual(**block_arg))
            block_arg["stride"] = 1
            block_arg["in_channels"] = block_arg["out_channels"]
            for _ in range(repeats - 1):
                block.append(InvertedResidual(**block_arg))

            self.blocks.append(nn.Sequential(*block))

        # Head
        out_channels = block_arg["out_channels"]
        num_features = make_divisible(1280 * width_multiplier)
        self.conv_head = conv1x1(out_channels, num_features)
        self.bn2 = norm_layer(num_features, activation=norm_act)

        # TODO: encoder
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop_rate, inplace=True)
        self.classifier = nn.Linear(num_features, num_classes)

        self._initialize_weights()

    def features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="linear")


EFFNET_BLOCKARGS = [
    "r1_k3_s11_e1_i32_o16_se0.25",
    "r2_k3_s22_e6_i16_o24_se0.25",
    "r2_k5_s22_e6_i24_o40_se0.25",
    "r3_k3_s22_e6_i40_o80_se0.25",
    "r3_k5_s11_e6_i80_o112_se0.25",
    "r4_k5_s22_e6_i112_o192_se0.25",
    "r1_k3_s11_e6_i192_o320_se0.25",
]


def _decode_block_string(block_string):
    """ Gets a block through a string notation of arguments. """
    assert isinstance(block_string, str)

    ops = block_string.split("_")
    options = {}
    for op in ops:
        splits = re.split(r"(\d.*)", op)
        if len(splits) >= 2:
            key, value = splits[:2]
            options[key] = value

    # Check stride
    assert ("s" in options and len(options["s"]) == 1) or (
        len(options["s"]) == 2 and options["s"][0] == options["s"][1]
    )
    options["s"] = max(map(int, options["s"]))

    return dict(
        in_channels=int(options["i"]),
        out_channels=int(options["o"]),
        dw_kernel_size=int(options["k"]),
        stride=tuple([options["s"], options["s"]]),
        use_se=float(options["se"]) > 0 if "se" in options else False,
        expand_ratio=int(options["e"]),
        noskip="noskip" in block_string,
        num_repeat=int(options["r"]),
    )


def decode_block_args(string_list):
    """
    Decodes a list of string notations to specify blocks inside the network.

    :param string_list: a list of strings, each string is a notation of block
    :return: a list of BlockArgs namedtuples of block args
    """
    assert isinstance(string_list, list)
    blocks_args = []
    for block_string in string_list:
        blocks_args.append(_decode_block_string(block_string))
    return blocks_args


# fmt: off
CFGS = {
    # pretrained are from rwightman
    # TODO: Double check pretrained settings! 
    "efficientnet-b0": {
        "default": {
            "params": {
                "blocks_args": EFFNET_BLOCKARGS, "width_multiplier": 1.0, "depth_multiplier": 1.0}, 
                **DEFAULT_IMAGENET_SETTINGS
            },
        "imagenet": {"url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth"},
    },
    "efficientnet-b1": {
        "default": {
            "params": {
                "blocks_args": EFFNET_BLOCKARGS, "width_multiplier": 1.0, "depth_multiplier": 1.1}, 
                **DEFAULT_IMAGENET_SETTINGS
            },
        "imagenet": {"url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b1-533bc792.pth"},
    },
    "efficientnet-b2": {
        "default": {
            "params": {
                "blocks_args": EFFNET_BLOCKARGS, "width_multiplier": 1.1, "depth_multiplier": 1.2},
                **DEFAULT_IMAGENET_SETTINGS
            },
        "imagenet": {"url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2_ra-bcdf34b7.pth"},
    },
    "efficientnet-b3": {
        "default": {
            "params": {
                "blocks_args": EFFNET_BLOCKARGS, "width_multiplier": 1.2, "depth_multiplier": 1.4}, 
                **DEFAULT_IMAGENET_SETTINGS
            },
        "imagenet": {"url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra-a5e2fbc7.pth"},
    },
}

# fmt: on


def _efficientnet(arch, pretrained=None, **kwargs):
    cfgs = deepcopy(CFGS)
    cfg_settings = cfgs[arch]["default"]
    cfg_params = cfg_settings.pop("params")
    cfg_params["blocks_args"] = decode_block_args(cfg_params["blocks_args"])
    if pretrained:
        pretrained_settings = cfgs[arch][pretrained]
        pretrained_params = pretrained_settings.pop("params", {})
        cfg_settings.update(pretrained_settings)
        cfg_params.update(pretrained_params)
    common_args = set(cfg_params.keys()).intersection(set(kwargs.keys()))

    assert (
        common_args == set()
    ), "Args {} are going to be overwritten by default params for {} weights".format(
        common_args.keys(), pretrained
    )
    kwargs.update(cfg_params)
    model = EfficientNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(cfgs[arch][pretrained]["url"])
        kwargs_cls = kwargs.get("num_classes", None)
        if kwargs_cls and kwargs_cls != cfg_settings["num_classes"]:
            logging.warning(
                "Using model pretrained for {} classes with {} classes. Last layer is initialized randomly".format(
                    cfg_settings["num_classes"], kwargs_cls
                )
            )
        model.load_state_dict(state_dict)
    return model


@wraps(EfficientNet)
@add_docs_for(EfficientNet)
def efficientnet_b0(**kwargs):
    r"""Constructs a Efficientnet B0 model."""
    return _efficientnet("efficientnet-b0", **kwargs)


@wraps(EfficientNet)
@add_docs_for(EfficientNet)
def efficientnet_b1(**kwargs):
    r"""Constructs a Efficientnet B1 model."""
    return _efficientnet("efficientnet-b1", **kwargs)


@wraps(EfficientNet)
@add_docs_for(EfficientNet)
def efficientnet_b2(**kwargs):
    r"""Constructs a Efficientnet B2 model."""
    return _efficientnet("efficientnet-b2", **kwargs)


@wraps(EfficientNet)
@add_docs_for(EfficientNet)
def efficientnet_b3(**kwargs):
    r"""Constructs a Efficientnet B3 model."""
    return _efficientnet("efficientnet-b3", **kwargs)


@wraps(EfficientNet)
@add_docs_for(EfficientNet)
def efficientnet_b4(**kwargs):
    r"""Constructs a Efficientnet B4 model."""
    return _efficientnet("efficientnet-b4", **kwargs)


@wraps(EfficientNet)
@add_docs_for(EfficientNet)
def efficientnet_b5(**kwargs):
    r"""Constructs a Efficientnet B5 model."""
    return _efficientnet("efficientnet-b5", **kwargs)


@wraps(EfficientNet)
@add_docs_for(EfficientNet)
def efficientnet_b6(**kwargs):
    r"""Constructs a Efficientnet B6 model."""
    return _efficientnet("efficientnet-b6", **kwargs)


@wraps(EfficientNet)
@add_docs_for(EfficientNet)
def efficientnet_b7(**kwargs):
    r"""Constructs a Efficientnet B6 model."""
    return _efficientnet("efficientnet-b7", **kwargs)
