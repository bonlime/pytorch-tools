"""PyTorch DenseNet

Copied from https://github.com/gpleiss/efficient_densenet_pytorch
and then modified to support additional parameters

Models fully compatible with default torchvision weights
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torchvision.models.utils import load_state_dict_from_url
from functools import wraps, partial
from pytorch_tools.modules.residual import conv1x1, conv3x3
from pytorch_tools.modules import GlobalPool2d
from pytorch_tools.utils.misc import initialize
from pytorch_tools.utils.misc import add_docs_for
from pytorch_tools.utils.misc import DEFAULT_IMAGENET_SETTINGS
from pytorch_tools.modules import bn_from_name
from pytorch_tools.modules import activation_from_name
from pytorch_tools.modules import ABN
from copy import deepcopy
import re
import logging

# avoid overwriting doc string
wraps = partial(wraps, assigned=("__module__", "__name__", "__qualname__", "__annotations__"))


class _Transition(nn.Module):
    r"""
    Transition Block as described in [DenseNet](https://arxiv.org/abs/1608.06993)
    
    - Activation
    - 1x1 Convolution (with optional compression of the number of channels)
    - 2x2 Average Pooling
    """

    def __init__(self, in_planes, out_planes, norm_layer=ABN, norm_act="relu"):
        super(_Transition, self).__init__()
        self.norm = norm_layer(in_planes, activation=norm_act)
        self.conv = conv1x1(in_planes, out_planes)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.norm(x)
        out = self.conv(out)
        out = self.pool(out)
        return out


def _bn_function_factory(norm, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(norm(concated_features))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    expansion = 4

    def __init__(
        self, in_planes, growth_rate, drop_rate=0.0, memory_efficient=False, norm_layer=ABN, norm_act="relu",
    ):
        super(_DenseLayer, self).__init__()

        width = growth_rate * self.expansion
        self.norm1 = norm_layer(in_planes, activation=norm_act)
        self.conv1 = conv1x1(in_planes, width)
        self.norm2 = norm_layer(width, activation=norm_act)
        self.conv2 = conv3x3(width, growth_rate)
        self.dropout = nn.Dropout(p=drop_rate, inplace=True)
        self.memory_efficient = memory_efficient

    def forward(self, *inputs):
        bn_function = _bn_function_factory(self.norm1, self.conv1)
        if self.memory_efficient and any(x.requires_grad for x in inputs):
            out = cp.checkpoint(bn_function, *inputs)
        else:
            out = bn_function(*inputs)
        out = self.conv2(self.norm2(out))
        out = self.dropout(out)
        return out


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_planes, growth_rate, **kwargs):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_planes + i * growth_rate, growth_rate=growth_rate, **kwargs)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, x):
        out = [x]
        for name, layer in self.named_children():
            new_out = layer(*out)
            out.append(new_out)
        return torch.cat(out, 1)


class DenseNet(nn.Module):
    r"""

    Args:
        growth_rate (int): 
            How many filters to add each layer (`k` in paper).
        block_config (List[int]): 
            How many layers in each pooling block.
        pretrained (str): 
            if present, returns a model pre-trained on 'str' dataset
        num_classes (int): 
            Number of classification classes. Defaults to 1000.
        drop_rate (float): 
            Dropout probability after each DenseLayer. Defaults to 0.0.
        in_channels (int): 
            Number of input (color) channels. Defaults to 3.
        norm_layer (str): 
            Nomaliztion layer to use. One of 'abn', 'inplaceabn'. The inplace version lowers memory footprint. 
            But increases backward time. Defaults to 'abn'.
        norm_act (str): 
            Activation for normalizion layer. It's reccomended to use `relu` with `abn`.
        deep_stem (bool): 
            Whether to replace the 7x7 conv1 with 3 3x3 convolution layers. Defaults to False.
        stem_width (int):
            Number of filters in the input stem
        encoder (bool): 
            Flag to overwrite forward pass to return 5 tensors with different resolutions. Defaults to False.
        global_pool (str): 
            Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'. Defaults to 'avg'.
        memory_efficient (bool):
            Use checkpointing. Much more memory efficient, but slower.
            See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_. Defaults to True.
    """

    def __init__(
        self,
        growth_rate=None,
        block_config=None,
        pretrained=None,  # not used. here for proper signature
        num_classes=1000,
        drop_rate=0.0,
        in_channels=3,
        norm_layer="abn",
        norm_act="relu",
        deep_stem=False,
        stem_width=64,
        encoder=False,
        global_pool="avg",
        memory_efficient=True,
        output_stride=32,  # not used! only here to allow using as encoder
    ):

        super(DenseNet, self).__init__()
        norm_layer = bn_from_name(norm_layer)
        self.num_classes = num_classes
        if deep_stem:
            self.conv0 = nn.Sequential(
                conv3x3(in_channels, stem_width // 2, 2),
                norm_layer(stem_width // 2, activation=norm_act),
                conv3x3(stem_width // 2, stem_width // 2),
                norm_layer(stem_width // 2, activation=norm_act),
                conv3x3(stem_width // 2, stem_width, 2),
            )
        else:
            self.conv0 = nn.Conv2d(in_channels, stem_width, kernel_size=7, stride=2, padding=3, bias=False)

        self.norm0 = norm_layer(stem_width, activation=norm_act)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

        largs = dict(
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient,
            norm_layer=norm_layer,
            norm_act=norm_act,
        )
        in_planes = stem_width
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, in_planes, **largs)
            setattr(self, f"denseblock{i+1}", block)
            in_planes += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    in_planes=in_planes, out_planes=in_planes // 2, norm_layer=norm_layer, norm_act=norm_act
                )
                setattr(self, f"transition{i+1}", trans)
                in_planes //= 2

        # Final normalization
        self.norm5 = nn.BatchNorm2d(in_planes)

        # Linear layer
        self.encoder = encoder
        if not encoder:
            self.global_pool = GlobalPool2d(global_pool)
            self.classifier = nn.Linear(in_planes, num_classes)
        else:
            assert len(block_config) == 4, "Need 4 blocks to use as encoder"
            self.forward = self.encoder_features
        initialize(self)

    def encoder_features(self, x):
        """
        Return 5 feature maps before maxpooling layers
        """
        x0 = self.norm0(self.conv0(x))
        x1 = self.denseblock1(self.pool0(x0))
        x2 = self.denseblock2(self.transition1(x1))
        x3 = self.denseblock3(self.transition2(x2))
        x4 = self.norm5(self.denseblock4(self.transition3(x3)))
        return [x4, x3, x2, x1, x0]

    def features(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.pool0(x)
        x = self.denseblock1(x)
        x = self.transition1(x)
        x = self.denseblock2(x)
        x = self.transition2(x)
        x = self.denseblock3(x)
        x = self.transition3(x)
        x = self.denseblock4(x)
        x = self.norm5(x)
        return x

    def logits(self, x):
        x = F.relu(x, inplace=True)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

    def load_state_dict(self, state_dict, **kwargs):
        pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )
        for key in list(state_dict.keys()):
            if key.startswith("classifier") and self.encoder:
                state_dict.pop(key)
            if key.startswith("features"):
                state_dict[key[9:]] = state_dict.pop(key)
            key = key[9:]
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict.pop(key)
        super().load_state_dict(state_dict, **kwargs)


CFGS = {
    "densenet121": {
        "default": {
            "params": {"growth_rate": 32, "block_config": (6, 12, 24, 16), "stem_width": 64,},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet": {"url": "https://download.pytorch.org/models/densenet121-a639ec97.pth"},
        # EXAMPLE RESNET
        # 'imagenet_inplaceabn': {
        #     'params': {'block': BasicBlock, 'layers': [2, 2, 2, 2], 'norm_layer': 'inplaceabn', 'deepstem':True, 'antialias':True},
        #     'url' : 'pathtomodel',
        #     **DEFAULT_IMAGENET_SETTINGS,
        # }
    },
    "densenet161": {
        "default": {
            "params": {"growth_rate": 48, "block_config": (6, 12, 36, 24), "stem_width": 96},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet": {"url": "https://download.pytorch.org/models/densenet161-8d451a50.pth"},
    },
    "densenet169": {
        "default": {
            "params": {"growth_rate": 32, "block_config": (6, 12, 32, 32)},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet": {"url": "https://download.pytorch.org/models/densenet169-b2777c0a.pth"},
    },
    "densenet201": {
        "default": {
            "params": {"growth_rate": 32, "block_config": (6, 12, 48, 32)},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet": {"url": "https://download.pytorch.org/models/densenet201-c1103571.pth"},
    },
    # DenseNet_BC
    "densenet121_bc": {
        "default": {"params": {"growth_rate": 32, "layers": (6, 12, 24, 16)}, **DEFAULT_IMAGENET_SETTINGS,},
    },
    "densenet161_bc": {
        "default": {
            "params": {"growth_rate": 48, "block_config": (6, 12, 36, 24), "stem_width": 96},
            **DEFAULT_IMAGENET_SETTINGS,
        },
    },
    "densenet169_bc": {
        "default": {
            "params": {"growth_rate": 32, "block_config": (6, 12, 32, 32)},
            **DEFAULT_IMAGENET_SETTINGS,
        },
    },
    "densenet201_bc": {
        "default": {
            "params": {"growth_rate": 32, "block_config": (6, 12, 48, 32)},
            **DEFAULT_IMAGENET_SETTINGS,
        },
    },
}


def _densenet(arch, pretrained=None, **kwargs):
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
    model = DenseNet(**kwargs)

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
            state_dict["classifier.weight"] = model.state_dict()["classifier.weight"]
            state_dict["classifier.bias"] = model.state_dict()["classifier.bias"]
        model.load_state_dict(state_dict)

    setattr(model, "pretrained_settings", cfg_settings)
    return model


@wraps(DenseNet)
@add_docs_for(DenseNet)
def densenet121(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    return _densenet("densenet121", **kwargs)


@wraps(DenseNet)
@add_docs_for(DenseNet)
def densenet161(**kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    return _densenet("densenet161", **kwargs)


@wraps(DenseNet)
@add_docs_for(DenseNet)
def densenet169(**kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    return _densenet("densenet169", **kwargs)


@wraps(DenseNet)
@add_docs_for(DenseNet)
def densenet201(**kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    return _densenet("densenet201", **kwargs)
