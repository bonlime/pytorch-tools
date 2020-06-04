import logging
from copy import deepcopy
from functools import wraps, partial

import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

from pytorch_tools.models.resnet import ResNet
from pytorch_tools.modules import TBasicBlock, TBottleneck
from pytorch_tools.modules import FastGlobalAvgPool2d, BlurPool
from pytorch_tools.modules import SpaceToDepth
from pytorch_tools.modules.residual import conv1x1, conv3x3
from pytorch_tools.modules import bn_from_name
from pytorch_tools.modules import ABN
from pytorch_tools.utils.misc import add_docs_for
from pytorch_tools.utils.misc import repeat_channels

# avoid overwriting doc string
wraps = partial(wraps, assigned=("__module__", "__name__", "__qualname__", "__annotations__"))


class TResNet(ResNet):
    """TResNet M / TResNet L / XL


    Ref: 
        * TResNet paper: https://arxiv.org/abs/2003.13630


    Args:
        layers (List[int]):
            Numbers of layers in each block.
        pretrained (str, optional):
            If not, returns a model pre-trained on 'str' dataset. `imagenet` is available
            for every model but some have more. Check the code to find out.
        num_classes (int):
            Number of classification classes. Defaults to 1000.
        in_channels (int):
            Number of input (color) channels. Defaults to 3.
        width_factor (int): 
            Stem width is 64 * width_factor. neede to make larger models
        output_stride (List[8, 16, 32]): Applying dilation strategy to pretrained ResNet. Typically used in
            Semantic Segmentation. Defaults to 32.
            NOTE: Don't use this arg with `antialias` and `pretrained` together. it may produce weird results
        norm_layer (str):
            Normalization layer to use. One of 'abn', 'inplaceabn'. The inplace version lowers memory footprint.
            But increases backward time. Defaults to 'inplaceabn'.
        norm_act (str):
            Activation for normalizion layer. It's reccomended to use `leacky_relu` with `inplaceabn`.
        encoder (bool):
            Flag to overwrite forward pass to return 5 tensors with different resolutions. Defaults to False.
            NOTE: TResNet first features have resolution 4x times smaller than input, not 2x as all other models. 
            So it CAN'T be used as encoder in Unet and Linknet models 
        drop_rate (float):
            Dropout probability before classifier, for training. Defaults to 0.
        drop_connect_rate (float):
            Drop rate for StochasticDepth. Randomly removes samples each block. Used as regularization during training. Ref: https://arxiv.org/abs/1603.09382
    """

    def __init__(
        self,
        layers=None,
        pretrained=None,  # not used. here for proper signature
        num_classes=1000,
        in_channels=3,
        width_factor=1.0,
        output_stride=32,
        norm_layer="inplaceabn",
        norm_act="leaky_relu",
        encoder=False,
        drop_rate=0.0,
        drop_connect_rate=0.0,
    ):
        nn.Module.__init__(self)
        stem_width = int(64 * width_factor)
        norm_layer = bn_from_name(norm_layer)
        self.inplanes = stem_width
        self.num_classes = num_classes
        self.groups = 1  # not really used but needed inside _make_layer
        self.base_width = 64  # used inside _make_layer
        self.norm_act = norm_act
        self.block_idx = 0
        self.num_blocks = sum(layers)
        self.drop_connect_rate = drop_connect_rate

        self._make_stem("space2depth", stem_width, in_channels, norm_layer, norm_act)

        if output_stride not in [8, 16, 32]:
            raise ValueError("Output stride should be in [8, 16, 32]")
        # TODO add OS later
        # if output_stride == 8:
        # stride_3, stride_4, dilation_3, dilation_4 = 1, 1, 2, 4
        # elif output_stride == 16:
        # stride_3, stride_4, dilation_3, dilation_4 = 2, 1, 1, 2
        # elif output_stride == 32:
        stride_3, stride_4, dilation_3, dilation_4 = 2, 2, 1, 1

        largs = dict(attn_type="se", norm_layer=norm_layer, norm_act=norm_act, antialias=True)
        self.block = TBasicBlock
        self.expansion = TBasicBlock.expansion
        self.layer1 = self._make_layer(stem_width, layers[0], stride=1, **largs)
        self.layer2 = self._make_layer(stem_width * 2, layers[1], stride=2, **largs)

        self.block = TBottleneck  # first 2 - Basic, last 2 - Bottleneck
        self.expansion = TBottleneck.expansion
        self.layer3 = self._make_layer(
            stem_width * 4, layers[2], stride=stride_3, dilation=dilation_3, **largs
        )
        largs.update(attn_type=None)  # no se in last layer
        self.layer4 = self._make_layer(
            stem_width * 8, layers[3], stride=stride_4, dilation=dilation_4, **largs
        )
        self.global_pool = FastGlobalAvgPool2d(flatten=True)
        self.num_features = stem_width * 8 * self.expansion
        self.encoder = encoder
        if not encoder:
            self.dropout = nn.Dropout(p=drop_rate, inplace=True)
            self.last_linear = nn.Linear(self.num_features, num_classes)
        else:
            self.forward = self.encoder_features

        self._initialize_weights(init_bn0=True)

    def load_state_dict(self, state_dict, **kwargs):
        if self.encoder:
            state_dict.pop("last_linear.weight")
            state_dict.pop("last_linear.bias")
        nn.Module.load_state_dict(self, state_dict, **kwargs)


# fmt: off
# images should be normalized to [0, 1]
PRETRAIN_SETTINGS = {
    "input_space": "RGB",
    "input_size": [3, 448, 448],
    "input_range": [0, 1],
    "mean": [0., 0., 0.],
    "std": [1., 1., 1.],
    "num_classes": 1000,
}
# for each model there are weights trained on 224 crops from imagenet and 
# weights finetuned on 448x448 crops. I'm using the latest as default because they work better for
# finetuning on large image sizes
CFGS = {
    "tresnetm": {
        "default": {"params": {"layers": [3, 4, 11, 3]}, **PRETRAIN_SETTINGS,},
        # Acc@1 81.712 Acc@5 95.502 on 448 crops and proper normalization
        "imagenet": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.2/patched_tresnet_m_448-a236aa30.pth"},
        # Acc@1 80.644 Acc@5 94.762 on 224 crops
        "imagenet2": {
            "url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.2/patched_tresnet_m-80fe4f81.pth",
            "input_size": [3, 224, 224],
        },
    },
    "tresnetl": {
        "default": {"params": {"layers": [4, 5, 18, 3], "width_factor": 1.2}, **PRETRAIN_SETTINGS,},
        # Acc@1 82.266 Acc@5 95.938
        "imagenet": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.2/patched_tresnet_l_448-9401272c.pth"},
        "imagenet2": {
            "url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.2/patched_tresnet_l-f37fa24a.pth",
            "input_size": [3, 224, 224],
        },
    },
    "tresnetxl": {
        "default": {"params": {"layers": [4, 5, 24, 3], "width_factor": 1.3}, **PRETRAIN_SETTINGS,},
        # Acc@1 83.046 Acc@5 96.172
        "imagenet": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.2/patched_tresnet_xl_448-11841266.pth"},
        # Acc@1 81.988 Acc@5 95.896
        "imagenet2": {
            "url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.2/patched_tresnet_xl-493cb618.pth",
            "input_size": [3, 224, 224],
        },
    },
}
# fmt: on


def patch_bn(module):
    """changes weight from InplaceABN to be compatible with usual ABN"""
    if isinstance(module, ABN):
        module.weight = nn.Parameter(module.weight.abs() + 1e-5)
    for m in module.children():
        patch_bn(m)


def _resnet(arch, pretrained=None, **kwargs):
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
    model = TResNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(cfgs[arch][pretrained]["url"], check_hash=True)
        kwargs_cls = kwargs.get("num_classes", None)
        if kwargs_cls and kwargs_cls != cfg_settings["num_classes"]:
            logging.warning(
                "Using model pretrained for {} classes with {} classes. Last layer is initialized randomly".format(
                    cfg_settings["num_classes"], kwargs_cls
                )
            )
            # if there is last_linear in state_dict, it's going to be overwritten
            state_dict["last_linear.weight"] = model.state_dict()["last_linear.weight"]
            state_dict["last_linear.bias"] = model.state_dict()["last_linear.bias"]
        if kwargs.get("in_channels", 3) != 3:  # support pretrained for custom input channels
            state_dict["conv1.1.weight"] = repeat_channels(
                state_dict["conv1.1.weight"], kwargs["in_channels"] * 16, 3 * 16
            )
        model.load_state_dict(state_dict)
        patch_bn(model)
    setattr(model, "pretrained_settings", cfg_settings)
    return model


@wraps(TResNet)
@add_docs_for(TResNet)
def tresnetm(**kwargs):
    r"""Constructs a TResnetM model."""
    return _resnet("tresnetm", **kwargs)


@wraps(TResNet)
@add_docs_for(TResNet)
def tresnetl(**kwargs):
    r"""Constructs a TResnetL model."""
    return _resnet("tresnetl", **kwargs)


@wraps(TResNet)
@add_docs_for(TResNet)
def tresnetxl(**kwargs):
    r"""Constructs a TResnetXL model."""
    return _resnet("tresnetxl", **kwargs)
