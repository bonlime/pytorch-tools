"""PyTorch EfficienNet
This implementation is based on:
https://github.com/lukemelas/EfficientNet-PyTorch
https://github.com/rwightman/gen-efficientnet-pytorch
but at the same time differs from them significantly

Key differences:
All normalizations and activations are changed to ABN
Can be used as encoder out of the box
"""

import re
import math
import logging
from copy import deepcopy
from collections import OrderedDict
from functools import wraps, partial

import torch
from torch import nn
from torchvision.models.utils import load_state_dict_from_url

from pytorch_tools.modules import ABN
from pytorch_tools.modules import bn_from_name
from pytorch_tools.modules.residual import InvertedResidual
from pytorch_tools.modules.residual import conv1x1, conv3x3
from pytorch_tools.modules.tf_same_ops import conv_to_same_conv
from pytorch_tools.modules.tf_same_ops import maxpool_to_same_maxpool
from pytorch_tools.utils.misc import initialize
from pytorch_tools.utils.misc import add_docs_for
from pytorch_tools.utils.misc import make_divisible
from pytorch_tools.utils.misc import DEFAULT_IMAGENET_SETTINGS
from pytorch_tools.utils.misc import repeat_channels

# avoid overwriting doc string
wraps = partial(wraps, assigned=("__module__", "__name__", "__qualname__", "__annotations__"))


class EfficientNet(nn.Module):
    """EfficientNet B0-B7

    Ref: https://arxiv.org/pdf/1905.11946.pdf


    Args:
        blocks_args (List[Dict]):
            Description of each block for the model. Check `decode_block_args` function for more details. Don't need to be
            passed manually
        width_multiplier (float):
            Multiplyer for number of channels in each block. Don't need to be passed manually
        depth_multiplier (float):
            Multiplyer for number of InvertedResiduals in each block. Don't need to be passed manually
        pretrained (str, optional):
            If not None, returns a model pre-trained on 'str' dataset. `imagenet` is available for every model.
            NOTE: weights which are loaded into this model were ported from TF. There is a drop in
            accuracy for Imagenet (~1-2% top1) but they work well for finetuning.
            NOTE 2: models were pretrained on very different resolutions. take it into account during finetuning
        num_classes (int):
            Number of classification classes. Defaults to 1000.
        in_channels (int):
            Number of input (color) channels. Defaults to 3.
        output_stride (List[8, 16, 32]): Applying dilation strategy to pretrained models. Typically used in
            Semantic Segmentation. Defaults to 32.
        encoder (bool):
            Flag to overwrite forward pass to return 5 tensors with different resolutions. Defaults to False.
        drop_rate (float):
            Dropout probability before classifier, for training. Defaults to 0.0.
        drop_connect_rate (float):
            Drop rate for StochasticDepth.
        norm_layer (str):
            Normalization layer to use. One of 'abn', 'inplaceabn'. The inplace version lowers memory footprint.
            But increases backward time and doesn't support `swish` activation. Defaults to 'abn'.
        norm_act (str):
            Activation for normalizion layer. It's reccomended to use `leacky_relu` with `inplaceabn`. Defaults to `swish`
        match_tf_same_padding (bool): If True patches Conv and MaxPool to implements tf-like asymmetric padding
            Should only be used to validate pretrained weights. Not needed for training. Gives ~10% slowdown
    """

    def __init__(
        self,
        blocks_args=None,
        width_multiplier=None,
        depth_multiplier=None,
        pretrained=None,  # not used. here for proper signature
        num_classes=1000,
        in_channels=3,
        output_stride=32,
        encoder=False,
        drop_rate=0,
        drop_connect_rate=0,
        stem_size=32,
        norm_layer="abn",
        norm_act="swish",
        match_tf_same_padding=False,
    ):
        super().__init__()
        norm_layer = bn_from_name(norm_layer)
        self.norm_layer = norm_layer
        self.norm_act = norm_act
        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        stem_size = make_divisible(stem_size * width_multiplier)
        self.conv_stem = conv3x3(in_channels, stem_size, stride=2)
        self.bn1 = norm_layer(stem_size, activation=norm_act)
        in_channels = stem_size
        self.blocks = nn.ModuleList([])
        # modify block args to account for output_stride strategy
        blocks_args = _patch_block_args(blocks_args, output_stride)
        for block_idx, block_arg in enumerate(blocks_args):
            block = []
            block_arg["in_channels"] = make_divisible(block_arg["in_channels"] * self.width_multiplier)
            block_arg["out_channels"] = make_divisible(block_arg["out_channels"] * self.width_multiplier)
            block_arg["norm_layer"] = norm_layer
            block_arg["norm_act"] = norm_act
            # linearly scale keep prob
            block_arg["keep_prob"] = 1 - drop_connect_rate * block_idx / len(blocks_args)
            repeats = block_arg.pop("num_repeat")
            repeats = int(math.ceil(repeats * self.depth_multiplier))
            # when dilating conv with stride 2 we want it to have dilation // 2
            # it prevents checkerboard artifacts with OS=16 and OS=8
            dilation = block_arg.get("dilation", 1)  # save block values
            if block_arg.pop("no_first_dilation", False):
                block_arg["dilation"] = max(1, block_arg["dilation"] // 2)
            block.append(InvertedResidual(**block_arg))
            # only first layer in block is strided
            block_arg["stride"] = 1
            block_arg["dilation"] = dilation
            block_arg["in_channels"] = block_arg["out_channels"]
            for _ in range(repeats - 1):
                block.append(InvertedResidual(**block_arg))

            self.blocks.append(nn.Sequential(*block))

        # Head

        if encoder:
            self.forward = self.encoder_features
        else:
            out_channels = block_arg["out_channels"]
            num_features = make_divisible(1280 * width_multiplier)
            self.conv_head = conv1x1(out_channels, num_features)
            self.bn2 = norm_layer(num_features, activation=norm_act)
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(drop_rate, inplace=True)
            self.last_linear = nn.Linear(num_features, num_classes)

        patch_bn_tf(self)  # adjust epsilon
        initialize(self)
        if match_tf_same_padding:
            conv_to_same_conv(self)
            maxpool_to_same_maxpool(self)

    def encoder_features(self, x):
        x0 = self.conv_stem(x)
        x0 = self.bn1(x0)
        x0 = self.blocks[0](x0)
        x1 = self.blocks[1](x0)
        x2 = self.blocks[2](x1)
        x3 = self.blocks[3](x2)
        x3 = self.blocks[4](x3)
        x4 = self.blocks[5](x3)
        x4 = self.blocks[6](x4)
        return [x4, x3, x2, x1, x0]

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
        x = self.last_linear(x)
        return x

    def load_state_dict(self, state_dict, **kwargs):
        valid_weights = []
        for key, value in state_dict.items():
            if "num_batches_tracked" in key:
                continue
            valid_weights.append(value)
        new_sd = OrderedDict(zip(self.state_dict().keys(), valid_weights))
        super().load_state_dict(new_sd, **kwargs)


EFFNET_BLOCKARGS = [
    "r1_k3_s11_e1_i32_o16_se0.25",
    "r2_k3_s22_e6_i16_o24_se0.25",
    "r2_k5_s22_e6_i24_o40_se0.25",
    "r3_k3_s22_e6_i40_o80_se0.25",
    "r3_k5_s11_e6_i80_o112_se0.25",
    "r4_k5_s22_e6_i112_o192_se0.25",
    "r1_k3_s11_e6_i192_o320_se0.25",
]


def _patch_block_args(blocks_args, output_stride=32):
    """iterate over block args and change `stride` and `dilation` according to `output_stride`"""
    if output_stride not in [8, 16, 32]:
        raise ValueError("Output stride should be in [8, 16, 32]")
    if output_stride == 32:
        return blocks_args
    dilation = 4 if output_stride == 8 else 2
    for ba in reversed(blocks_args):
        ba["dilation"] = dilation
        if ba["stride"][0] == 2:
            ba["stride"] = (1, 1)
            ba["no_first_dilation"] = True
            dilation //= 2
        if dilation == 1:
            break
    return blocks_args


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
        attn_type="se" if "se" in options else None,
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
PRETRAIN_SETTINGS = DEFAULT_IMAGENET_SETTINGS
PRETRAIN_SETTINGS["interpolation"] = "bicubic"
PRETRAIN_SETTINGS["crop_pct"] = 0.875

CFGS = {
    # All pretrained models were trained on TF by Google and ported to PyTorch by Ross Wightman @rwightman
    # Due to framework little differences (BN epsilon and different padding in convs) this weights give slightly
    # worse performance when loaded into model above but the drop is only about ~1% on Imagenet and doesn't really 
    # mater for transfer learning 
    # upd. by default weights from Noisy Student paper are loaded due to a much better predictions
    "efficientnet-b0": {
        "default": {
            "params": {
                "blocks_args": EFFNET_BLOCKARGS, "width_multiplier": 1.0, "depth_multiplier": 1.0}, 
                **PRETRAIN_SETTINGS,
                "input_size": [3, 224, 224],
            },
        "imagenet": {
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_ns-c0e6a31c.pth",
        },
        "imagenet2": {
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_aa-827b6e33.pth",
        },
    },
    "efficientnet-b1": {
        "default": {
            "params": {
                "blocks_args": EFFNET_BLOCKARGS, "width_multiplier": 1.0, "depth_multiplier": 1.1}, 
                **PRETRAIN_SETTINGS,
                "input_size": [3, 240, 240], 
                "crop_pct": 0.882,
            },
        "imagenet": {
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_ns-99dd0c41.pth",
        },
        "imagenet2": {
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_aa-ea7a6ee0.pth",
        },
    },
    "efficientnet-b2": {
        "default": {
            "params": {"blocks_args": EFFNET_BLOCKARGS, "width_multiplier": 1.1, "depth_multiplier": 1.2},
            **DEFAULT_IMAGENET_SETTINGS,
            "input_size": [3, 260, 260],
            "crop_pct": 0.890,
        },
        "imagenet": { # noisy student. original: Acc@1: 81.97. Acc@5: 96.10. My: Acc@1: 81.41. Acc@5: 95.84
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_ns-00306e48.pth",
        },
        "imagenet2": { # auto augment
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_aa-60c94f97.pth",
        },
    },
    "efficientnet-b3": {
        "default": {
            "params": {"blocks_args": EFFNET_BLOCKARGS, "width_multiplier": 1.2, "depth_multiplier": 1.4},
            **DEFAULT_IMAGENET_SETTINGS,
            "input_size": [3, 300, 300],
            "crop_pct": 0.904,
        },
        "imagenet": { # noisy student. original: Acc@1: 83.61. Acc@5: 96.78. My: gives Acc@1: 82.23. Acc@5: 95
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_ns-9d44bf68.pth",
        },
        "imagenet2": {
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_aa-84b4657e.pth",
        },
    },
    "efficientnet-b4": {
        "default": {
            "params": {"blocks_args": EFFNET_BLOCKARGS, "width_multiplier": 1.4, "depth_multiplier": 1.8},
            **DEFAULT_IMAGENET_SETTINGS,
            "input_size": [3, 380, 380],
            "crop_pct": 0.922,
            },
        "imagenet": {
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_ns-d6313a46.pth",
        },
        "imagenet2": {
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_aa-818f208c.pth",
        },
    },
    "efficientnet-b5": {
        "default": {
            "params": {"blocks_args": EFFNET_BLOCKARGS, "width_multiplier": 1.6, "depth_multiplier": 2.2},
            **DEFAULT_IMAGENET_SETTINGS,
            "input_size": [3, 456, 456],
            "crop_pct": 0.934,
            },
        "imagenet": { # noisy student. original: Acc@1: 85.79. Acc@5: 97.72. my: Acc@1: 85.89. Acc@5: 97.63
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ns-6f26d0cf.pth",
        },
        "imagenet2": {
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ra-9a3e5369.pth",
        },
    },
    "efficientnet-b6": {
        "default": {
            "params": {"blocks_args": EFFNET_BLOCKARGS, "width_multiplier": 1.8, "depth_multiplier": 2.6},
            **DEFAULT_IMAGENET_SETTINGS,
            "input_size": [3, 528, 528],
            "crop_pct": 0.942,
            },
        "imagenet": {
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ns-51548356.pth",
        },
        "imagenet2": {
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_aa-80ba17e4.pth",
        },
    },
    "efficientnet-b7": {
        "default": {
            "params": {"blocks_args": EFFNET_BLOCKARGS, "width_multiplier": 2.0, "depth_multiplier": 3.1},
            **DEFAULT_IMAGENET_SETTINGS,
            "input_size": [3, 600, 600],
            "crop_pct": 0.949,
            },
        "imagenet": {
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth",
        },
        "imagenet2": {
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ra-6c08e654.pth",
        },
    },
}

# fmt: on


def patch_bn_tf(module):
    """TF ported weights use slightly different eps in BN. Need to adjust for better performance"""
    if isinstance(module, ABN):
        module.eps = 1e-3
        module.momentum = 1e-2
    for m in module.children():
        patch_bn_tf(m)


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
    if common_args:
        logging.warning(
            f"Args {common_args} are going to be overwritten by default params for {pretrained} weights"
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
            state_dict["classifier.weight"] = model.state_dict()["last_linear.weight"]
            state_dict["classifier.bias"] = model.state_dict()["last_linear.bias"]
        if kwargs.get("in_channels", 3) != 3:  # support pretrained for custom input channels
            state_dict["conv_stem.weight"] = repeat_channels(
                state_dict["conv_stem.weight"], kwargs["in_channels"]
            )
        model.load_state_dict(state_dict)
    setattr(model, "pretrained_settings", cfg_settings)
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
