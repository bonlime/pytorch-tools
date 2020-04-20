"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants added by Ross Wightman
"""
import logging
from copy import deepcopy
from collections import OrderedDict
from functools import wraps, partial

import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

from pytorch_tools.modules import BasicBlock, Bottleneck
from pytorch_tools.modules import GlobalPool2d, BlurPool
from pytorch_tools.modules.residual import conv1x1, conv3x3
from pytorch_tools.modules import bn_from_name
from pytorch_tools.utils.misc import add_docs_for
from pytorch_tools.utils.misc import DEFAULT_IMAGENET_SETTINGS
from pytorch_tools.utils.misc import repeat_channels

# avoid overwriting doc string
wraps = partial(wraps, assigned=("__module__", "__name__", "__qualname__", "__annotations__"))


class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt and SE-ResNeXt that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on 'Bag of Tricks' paper:
    https://arxiv.org/pdf/1812.01187.


    Args:
        block (Block):
            Class for the residual block. Options are BasicBlock, Bottleneck.
        layers (List[int]):
            Numbers of layers in each block.
        pretrained (str, optional):
            If not, returns a model pre-trained on 'str' dataset. `imagenet` is available
            for every model but some have more. Check the code to find out.
        num_classes (int):
            Number of classification classes. Defaults to 1000.
        in_channels (int):
            Number of input (color) channels. Defaults to 3.
        use_se (bool):
            Enable Squeeze-Excitation module in blocks.
        groups (int):
            Number of convolution groups for 3x3 conv in Bottleneck. Defaults to 1.
        base_width (int):
            Factor determining bottleneck channels. `planes * base_width / 64 * groups`. Defaults to 64.
        deep_stem (bool):
            Whether to replace the 7x7 conv1 with 3 3x3 convolution layers. Defaults to False.
        output_stride (List[8, 16, 32]): Applying dilation strategy to pretrained ResNet. Typically used in
            Semantic Segmentation. Defaults to 32.
            NOTE: Don't use this arg with `antialias` and `pretrained` together. it may produce weird results
        norm_layer (str):
            Normalization layer to use. One of 'abn', 'inplaceabn'. The inplace version lowers memory footprint.
            But increases backward time. Defaults to 'abn'.
        norm_act (str):
            Activation for normalizion layer. It's reccomended to use `leacky_relu` with `inplaceabn`.
        antialias (bool):
            Flag to turn on Rect-2 antialiasing from https://arxiv.org/abs/1904.11486. Defaults to False.
        encoder (bool):
            Flag to overwrite forward pass to return 5 tensors with different resolutions. Defaults to False.
        drop_rate (float):
            Dropout probability before classifier, for training. Defaults to 0.0.
        drop_connect_rate (float):
            Drop rate for StochasticDepth. Randomly removes samples each block. Used as regularization during training. 
            keep prob will be linearly decreased from 1 to 1 - drop_connect_rate each block. Ref: https://arxiv.org/abs/1603.09382
        global_pool (str):
            Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'. Defaults to 'avg'.
        init_bn0 (bool):
            Zero-initialize the last BN in each residual branch, so that the residual
            branch starts with zeros, and each residual block behaves like an identity.
            This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677. Defaults to True.
    """

    def __init__(
        self,
        block=None,
        layers=None,
        pretrained=None,  # not used. here for proper signature
        num_classes=1000,
        in_channels=3,
        use_se=False,
        groups=1,
        base_width=64,
        deep_stem=False,
        output_stride=32,
        norm_layer="abn",
        norm_act="relu",
        antialias=False,
        encoder=False,
        drop_rate=0.0,
        drop_connect_rate=0.0,
        global_pool="avg",
        init_bn0=True,
    ):

        stem_width = 64
        norm_layer = bn_from_name(norm_layer)
        self.inplanes = stem_width
        self.num_classes = num_classes
        self.groups = groups
        self.base_width = base_width
        self.block = block
        self.expansion = block.expansion
        self.norm_act = norm_act
        self.block_idx = 0
        self.num_blocks = sum(layers)
        self.drop_connect_rate = drop_connect_rate
        super(ResNet, self).__init__()

        if deep_stem:
            self.conv1 = nn.Sequential(
                conv3x3(in_channels, stem_width // 2, 2),
                norm_layer(stem_width // 2, activation=norm_act),
                conv3x3(stem_width // 2, stem_width // 2),
                norm_layer(stem_width // 2, activation=norm_act),
                conv3x3(stem_width // 2, stem_width),
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, stem_width, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(stem_width, activation=norm_act)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0 if use_se else 1, ceil_mode=True if use_se else False,
        )
        if output_stride not in [8, 16, 32]:
            raise ValueError("Output stride should be in [8, 16, 32]")
        if output_stride == 8:
            stride_3, stride_4, dilation_3, dilation_4 = 1, 1, 2, 4
        elif output_stride == 16:
            stride_3, stride_4, dilation_3, dilation_4 = 2, 1, 1, 2
        elif output_stride == 32:
            stride_3, stride_4, dilation_3, dilation_4 = 2, 2, 1, 1
        largs = dict(use_se=use_se, norm_layer=norm_layer, norm_act=norm_act, antialias=antialias)
        self.layer1 = self._make_layer(64, layers[0], stride=1, **largs)
        self.layer2 = self._make_layer(128, layers[1], stride=2, **largs)
        self.layer3 = self._make_layer(256, layers[2], stride=stride_3, dilation=dilation_3, **largs)
        self.layer4 = self._make_layer(512, layers[3], stride=stride_4, dilation=dilation_4, **largs)
        self.global_pool = GlobalPool2d(global_pool)
        self.num_features = 512 * self.expansion
        self.encoder = encoder
        if not encoder:
            self.dropout = nn.Dropout(p=drop_rate, inplace=True)
            self.last_linear = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes)
        else:
            self.forward = self.encoder_features

        self._initialize_weights(init_bn0)

    def _make_layer(
        self,
        planes,
        blocks,
        stride=1,
        dilation=1,
        use_se=None,
        norm_layer=None,
        norm_act=None,
        antialias=None,
    ):
        downsample = None

        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample_layers = []
            if antialias and stride == 2:  # using OrderedDict to preserve ordering and allow loading
                downsample_layers += [("blur", nn.AvgPool2d(2, 2))]
            downsample_layers += [
                ("0", conv1x1(self.inplanes, planes * self.expansion, stride=1 if antialias else stride)),
                ("1", norm_layer(planes * self.expansion, activation="identity")),
            ]
            downsample = nn.Sequential(OrderedDict(downsample_layers))
        # removes first dilation to avoid checkerboard artifacts
        first_dilation = max(1, dilation // 2)
        layers = [
            self.block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                use_se=use_se,
                dilation=first_dilation,
                norm_layer=norm_layer,
                norm_act=norm_act,
                antialias=antialias,
                keep_prob=self.keep_prob,
            )
        ]

        self.inplanes = planes * self.expansion
        for _ in range(1, blocks):
            layers.append(
                self.block(
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    use_se=use_se,
                    dilation=first_dilation,
                    norm_layer=norm_layer,
                    norm_act=norm_act,
                    antialias=antialias,
                    keep_prob=self.keep_prob,
                )
            )
        return nn.Sequential(*layers)

    def _initialize_weights(self, init_bn0=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        if init_bn0:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def encoder_features(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x4, x3, x2, x1, x0]

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

    def load_state_dict(self, state_dict, **kwargs):
        keys = list(state_dict.keys())
        # filter classifier and num_batches_tracked
        for k in keys:
            if (k.startswith("fc") or k.startswith("last_linear")) and self.encoder:
                state_dict.pop(k)
            elif k.startswith("fc"):
                state_dict[k.replace("fc", "last_linear")] = state_dict.pop(k)
            if k.startswith("layer0"):
                state_dict[k.replace("layer0.", "")] = state_dict.pop(k)
        super().load_state_dict(state_dict, **kwargs)

    @property
    def keep_prob(self):
        keep_prob = 1 - self.drop_connect_rate * self.block_idx / self.num_blocks
        self.block_idx += 1
        return keep_prob

# fmt: off
CFGS = {
    # RESNET MODELS
    "resnet18": {
        "default": {"params": {"block": BasicBlock, "layers": [2, 2, 2, 2]}, **DEFAULT_IMAGENET_SETTINGS,},
        "imagenet": {"url": "https://download.pytorch.org/models/resnet18-5c106cde.pth"},
        # EXAMPLE
        # 'imagenet_inplaceabn': {
        #     'params': {'block': BasicBlock, 'layers': [2, 2, 2, 2], 'norm_layer': 'inplaceabn', 'deepstem':True, 'antialias':True},
        #     'url' : 'pathtomodel',
        #     **DEFAULT_IMAGENET_SETTINGS,
        # }
    },
    "resnet34": {
        "default": {"params": {"block": BasicBlock, "layers": [3, 4, 6, 3]}, **DEFAULT_IMAGENET_SETTINGS,},
        "imagenet": {  #                          Acc@1: 71.80. Acc@5: 90.37
            "url": "https://download.pytorch.org/models/resnet34-333f7ec4.pth"
        },
        "imagenet2": {  # weigths from rwightman. Acc@1: 73.25. Acc@5: 91.32
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth",
        },
    },
    "resnet50": {
        "default": {"params": {"block": Bottleneck, "layers": [3, 4, 6, 3]}, **DEFAULT_IMAGENET_SETTINGS,},
        "imagenet": {"url": "https://download.pytorch.org/models/resnet50-19c8e357.pth"},
    },
    "resnet101": {
        "default": {"params": {"block": Bottleneck, "layers": [3, 4, 23, 3]}, **DEFAULT_IMAGENET_SETTINGS,},
        "imagenet": {"url": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"},
    },
    "resnet152": {
        "default": {"params": {"block": Bottleneck, "layers": [3, 8, 36, 3]}, **DEFAULT_IMAGENET_SETTINGS,},
        "imagenet": {"url": "https://download.pytorch.org/models/resnet152-b121ed2d.pth"},
    },
    # WIDE RESNET MODELS
    "wide_resnet50_2": {
        "default": {
            "params": {"block": Bottleneck, "layers": [3, 4, 6, 3], "base_width": 128},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet": {"url": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth"},
    },
    "wide_resnet101_2": {
        "default": {
            "params": {"block": Bottleneck, "layers": [3, 4, 23, 3], "base_width": 128},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet": {"url": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth"},
    },
    # RESNEXT MODELS
    "resnext50_32x4d": {
        "default": {
            "params": {"block": Bottleneck, "layers": [3, 4, 6, 3], "base_width": 4, "groups": 32,},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet": {  # Acc@1: 75.80. Acc@5: 92.71.
            "url": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth"
        },
        # weights from rwightman
        "imagenet2": {
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50d_32x4d-103e99f8.pth"
        },
    },
    "resnext101_32x4d": {
        "default": {
            "params": {"block": Bottleneck, "layers": [3, 4, 23, 3], "base_width": 4, "groups": 32,},
            **DEFAULT_IMAGENET_SETTINGS,
        },  # No pretrained
    },
    "resnext101_32x8d": {
        "default": {
            "params": {"block": Bottleneck, "layers": [3, 4, 23, 3], "base_width": 8, "groups": 32,},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet": {"url": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth"},
        # pretrained on weakly labeled instagram and then tuned on Imagenet
        "imagenet_ig": {"url": "https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth"},
    },
    "resnext101_32x16d": {
        "default": {
            "params": {"block": Bottleneck, "layers": [3, 4, 23, 3], "base_width": 16, "groups": 32,},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        # pretrained on weakly labeled instagram and then tuned on Imagenet
        "imagenet_ig": {"url": "https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth"},
    },
    "resnext101_32x32d": {
        "default": {
            "params": {"block": Bottleneck, "layers": [3, 4, 23, 3], "base_width": 32, "groups": 32,},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        # pretrained on weakly labeled instagram and then tuned on Imagenet
        "imagenet_ig": {"url": "https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth"},
    },
    "resnext101_32x48d": {
        "default": {  # actually it's imagenet_ig. pretrained on weakly labeled instagram and then tuned on Imagenet
            "params": {"block": Bottleneck, "layers": [3, 4, 23, 3], "base_width": 48, "groups": 32,},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet_ig": {"url": "https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth"},
    },
    # SE RESNET MODELS
    "se_resnet34": {
        "default": {
            "params": {"block": BasicBlock, "layers": [3, 4, 6, 3], "use_se": True},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        # NO WEIGHTS
    },
    "se_resnet50": {
        "default": {
            "params": {"block": Bottleneck, "layers": [3, 4, 6, 3], "use_se": True},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet": {"url": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth"},
    },
    "se_resnet101": {
        "default": {
            "params": {"block": Bottleneck, "layers": [3, 4, 23, 3], "use_se": True},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet": {"url": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth"},
    },
    "se_resnet152": {
        "default": {
            "params": {"block": Bottleneck, "layers": [3, 4, 36, 3], "use_se": True},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet": {"url": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth"},
    },
    # SE RESNEXT MODELS
    "se_resnext50_32x4d": {
        "default": {
            "params": {
                "block": Bottleneck,
                "layers": [3, 4, 6, 3],
                "base_width": 4,
                "groups": 32,
                "use_se": True,
            },
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet": {"url": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth"},
    },
    "se_resnext101_32x4d": {
        "default": {
            "params": {
                "block": Bottleneck,
                "layers": [3, 4, 23, 3],
                "base_width": 4,
                "groups": 32,
                "use_se": True,
            },
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet": {"url": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth"},
    },
}
# fmt: on


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
    assert (
        common_args == set()
    ), "Args {} are going to be overwritten by default params for {} weights".format(
        common_args, pretrained
    )
    kwargs.update(cfg_params)
    model = ResNet(**kwargs)
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
        # layer0. is needed to support se_resne(x)t weights
        if kwargs.get("in_channels", 3) != 3:
            old_weights = state_dict.get("conv1.weight")
            old_weights = state_dict.get("layer0.conv1.weight") if old_weights is None else old_weights
            state_dict["layer0.conv1.weight"] = repeat_channels(old_weights, kwargs["in_channels"])
        model.load_state_dict(state_dict)
    setattr(model, "pretrained_settings", cfg_settings)
    return model


@wraps(ResNet)
@add_docs_for(ResNet)
def resnet18(**kwargs):
    r"""Constructs a ResNet-18 model."""
    return _resnet("resnet18", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def resnet34(**kwargs):
    r"""Constructs a ResNet-34 model."""
    return _resnet("resnet34", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def resnet50(**kwargs):
    r"""Constructs a ResNet-50 model."""
    return _resnet("resnet50", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def resnet101(**kwargs):
    r"""Constructs a ResNet-101 model."""
    return _resnet("resnet101", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def resnet152(**kwargs):
    """Constructs a ResNet-152 model."""
    return _resnet("resnet152", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def wide_resnet50_2(**kwargs):
    r"""Constructs a Wide ResNet-50-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    return _resnet("wide_resnet50_2", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def wide_resnet101_2(**kwargs):
    r"""Constructs a Wide ResNet-101-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same."""
    return _resnet("wide_resnet101_2", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def resnext50_32x4d(**kwargs):
    r"""Constructs a ResNeXt50-32x4d model."""
    return _resnet("resnext50_32x4d", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def resnext101_32x4d(**kwargs):
    r"""Constructs a ResNeXt101-32x4d model."""
    return _resnet("resnext101_32x4d", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def resnext101_32x8d(**kwargs):
    r"""Constructs a ResNeXt101-32x8d model."""
    return _resnet("resnext101_32x8d", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def ig_resnext101_32x8d(**kwargs):
    r"""Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/"""
    return _resnet("ig_resnext101_32x8d", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def ig_resnext101_32x16d(**kwargs):
    r"""Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data."""
    return _resnet("ig_resnext101_32x16d", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def ig_resnext101_32x32d(**kwargs):
    r"""Constructs a ResNeXt-101 32x32 model pre-trained on weakly-supervised data."""
    return _resnet("ig_resnext101_32x32d", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def ig_resnext101_32x48d(**kwargs):
    r"""Constructs a ResNeXt-101 32x48 model pre-trained on weakly-supervised data."""
    return _resnet("ig_resnext101_32x48d", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def se_resnet34(**kwargs):
    """TODO: Add Doc"""
    return _resnet("se_resnet34", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def se_resnet50(**kwargs):
    """TODO: Add Doc"""
    return _resnet("se_resnet50", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def se_resnet101(**kwargs):
    """TODO: Add Doc"""
    return _resnet("se_resnet101", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def se_resnet152(**kwargs):
    """TODO: Add Doc"""
    return _resnet("se_resnet152", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def se_resnext50_32x4d(**kwargs):
    """TODO: Add Doc"""
    return _resnet("se_resnext50_32x4d", **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def se_resnext101_32x4d(**kwargs):
    """TODO: Add Doc"""
    return _resnet("se_resnext101_32x4d", **kwargs)
