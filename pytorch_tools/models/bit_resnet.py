# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Bottleneck ResNet v2 with GroupNorm and Weight Standardization."""
import os
import numpy as np
from copy import deepcopy
from functools import wraps
from urllib.parse import urlparse
from collections import OrderedDict  # pylint: disable=g-importing-member

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_tools.modules.weight_standartization import WS_Conv2d as StdConv2d


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


def tf2th(conv_weights):
    """Possibly convert HWIO to OIHW."""
    if conv_weights.ndim == 4:
        conv_weights = conv_weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(conv_weights)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.

    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cin)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid)
        self.conv2 = conv3x3(cmid, cmid, stride)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cmid)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride)

    def forward(self, x):
        out = self.relu(self.gn1(x))

        # Residual branch
        residual = x
        if hasattr(self, "downsample"):
            residual = self.downsample(out)

        # Unit's branch
        out = self.conv1(out)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))

        return out + residual

    def load_from(self, weights, prefix=""):
        convname = "standardized_conv2d"
        with torch.no_grad():
            self.conv1.weight.copy_(tf2th(weights[f"{prefix}a/{convname}/kernel"]))
            self.conv2.weight.copy_(tf2th(weights[f"{prefix}b/{convname}/kernel"]))
            self.conv3.weight.copy_(tf2th(weights[f"{prefix}c/{convname}/kernel"]))
            self.gn1.weight.copy_(tf2th(weights[f"{prefix}a/group_norm/gamma"]))
            self.gn2.weight.copy_(tf2th(weights[f"{prefix}b/group_norm/gamma"]))
            self.gn3.weight.copy_(tf2th(weights[f"{prefix}c/group_norm/gamma"]))
            self.gn1.bias.copy_(tf2th(weights[f"{prefix}a/group_norm/beta"]))
            self.gn2.bias.copy_(tf2th(weights[f"{prefix}b/group_norm/beta"]))
            self.gn3.bias.copy_(tf2th(weights[f"{prefix}c/group_norm/beta"]))
            if hasattr(self, "downsample"):
                w = weights[f"{prefix}a/proj/{convname}/kernel"]
                self.downsample.weight.copy_(tf2th(w))


# this models are designed for trasfer learning only! not for training from scratch
class ResNetV2(nn.Module):
    """
    Implementation of Pre-activation (v2) ResNet mode.
    Used to create Bit-M-50/101/152x1/2/3/4 models
    
    Args:
        num_classes (int): Number of classification classes. Defaults to 5
    """

    def __init__(
        self,
        block_units,
        width_factor,
        # in_channels=3, # TODO: add later
        num_classes=5,  # just a random number
        # encoder=False, # TODO: add later
    ):
        super().__init__()
        wf = width_factor  # shortcut 'cause we'll use it a lot.

        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        # fmt: off
        self.root = nn.Sequential(OrderedDict([
                ('conv', StdConv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False)),
                ('pad', nn.ConstantPad2d(1, 0)),
                ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
                # The following is subtly not the same!
                # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.body = nn.Sequential(OrderedDict([
                ('block1', nn.Sequential(OrderedDict(
                        [('unit01', PreActBottleneck(cin=64*wf, cout=256*wf, cmid=64*wf))] +
                        [(f'unit{i:02d}', PreActBottleneck(cin=256*wf, cout=256*wf, cmid=64*wf)) for i in range(2, block_units[0] + 1)],
                ))),
                ('block2', nn.Sequential(OrderedDict(
                        [('unit01', PreActBottleneck(cin=256*wf, cout=512*wf, cmid=128*wf, stride=2))] +
                        [(f'unit{i:02d}', PreActBottleneck(cin=512*wf, cout=512*wf, cmid=128*wf)) for i in range(2, block_units[1] + 1)],
                ))),
                ('block3', nn.Sequential(OrderedDict(
                        [('unit01', PreActBottleneck(cin=512*wf, cout=1024*wf, cmid=256*wf, stride=2))] +
                        [(f'unit{i:02d}', PreActBottleneck(cin=1024*wf, cout=1024*wf, cmid=256*wf)) for i in range(2, block_units[2] + 1)],
                ))),
                ('block4', nn.Sequential(OrderedDict(
                        [('unit01', PreActBottleneck(cin=1024*wf, cout=2048*wf, cmid=512*wf, stride=2))] +
                        [(f'unit{i:02d}', PreActBottleneck(cin=2048*wf, cout=2048*wf, cmid=512*wf)) for i in range(2, block_units[3] + 1)],
                ))),
        ]))
        # pylint: enable=line-too-long

        self.head = nn.Sequential(OrderedDict([
                ('gn', nn.GroupNorm(32, 2048*wf)),
                ('relu', nn.ReLU(inplace=True)),
                ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
                ('conv', nn.Conv2d(2048*wf, num_classes, kernel_size=1, bias=True)),
        ]))
        # fmt: on

    def features(self, x):
        return self.body(self.root(x))

    def logits(self, x):
        return self.head(x)

    def forward(self, x):
        x = self.logits(self.features(x))
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0]

    def load_from(self, weights, prefix="resnet/"):
        with torch.no_grad():
            self.root.conv.weight.copy_(
                tf2th(weights[f"{prefix}root_block/standardized_conv2d/kernel"])
            )  # pylint: disable=line-too-long
            self.head.gn.weight.copy_(tf2th(weights[f"{prefix}group_norm/gamma"]))
            self.head.gn.bias.copy_(tf2th(weights[f"{prefix}group_norm/beta"]))
            # always zero_head
            nn.init.zeros_(self.head.conv.weight)
            nn.init.zeros_(self.head.conv.bias)

            for bname, block in self.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, prefix=f"{prefix}{bname}/{uname}/")


KNOWN_MODELS = OrderedDict(
    [
        ("BiT-M-R50x1", lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
        ("BiT-M-R50x3", lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
        ("BiT-M-R101x1", lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
        ("BiT-M-R101x3", lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
        ("BiT-M-R152x2", lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
        ("BiT-M-R152x4", lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
        ("BiT-S-R50x1", lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
        ("BiT-S-R50x3", lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
        ("BiT-S-R101x1", lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
        ("BiT-S-R101x3", lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
        ("BiT-S-R152x2", lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
        ("BiT-S-R152x4", lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
    ]
)


PRETRAIN_SETTINGS = {
    "input_space": "RGB",
    "input_size": [3, 448, 448],
    "input_range": [0, 1],
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "num_classes": None,
}

# fmt: off
CFGS = {
    # weights are loaded by default
    "bit_m_50x1": {
        "default": {
            "params": {"block_units": [3, 4, 6, 3], "width_factor": 1},
            "url": "https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz",
            **PRETRAIN_SETTINGS
        },
    },
    "bit_m_50x3": {
        "default": {
            "params": {"block_units": [3, 4, 6, 3], "width_factor": 3},
            "url": "https://storage.googleapis.com/bit_models/BiT-M-R50x3.npz",
            **PRETRAIN_SETTINGS,
        },
    },
    "bit_m_101x1": {
        "default": {
            "params": {"block_units": [3, 4, 23, 3], "width_factor": 1},
            "url": "https://storage.googleapis.com/bit_models/BiT-M-R101x1.npz",
            **PRETRAIN_SETTINGS,
        },
    },
    "bit_m_101x3": {
        "default": {
            "params": {"block_units": [3, 4, 23, 3], "width_factor": 3},
            "url": "https://storage.googleapis.com/bit_models/BiT-M-R101x3.npz",
            **PRETRAIN_SETTINGS,
        },
    },
    "bit_m_152x2": {
        "default": {
            "params": {"block_units": [3, 8, 36, 3], "width_factor": 2},
            "url": "https://storage.googleapis.com/bit_models/BiT-M-R152x2.npz",
            **PRETRAIN_SETTINGS,
        },
    },
    "bit_m_152x4": {
        "default": {
            "params": {"block_units": [3, 8, 36, 3], "width_factor": 4},
            "url": "https://storage.googleapis.com/bit_models/BiT-M-R152x4.npz",
            **PRETRAIN_SETTINGS
        },
    },
}

# fmt: on
def _bit_resnet(arch, pretrained=None, **kwargs):
    cfgs = deepcopy(CFGS)
    cfg_settings = cfgs[arch]["default"]
    cfg_params = cfg_settings.pop("params")
    cfg_url = cfg_settings.pop("url")
    kwargs.pop("pretrained", None)
    kwargs.update(cfg_params)
    model = ResNetV2(**kwargs)
    # load weights to torch checkpoints folder
    try:
        torch.hub.load_state_dict_from_url(cfg_url)
    except RuntimeError:
        pass  # to avoid RuntimeError: Only one file(not dir) is allowed in the zipfile
    filename = os.path.basename(urlparse(cfg_url).path)
    torch_home = torch.hub._get_torch_home()
    cached_file = os.path.join(torch_home, "checkpoints", filename)
    weights = np.load(cached_file)
    model.load_from(weights)
    return model


# only want M versions of models for fine-tuning
@wraps(ResNetV2)
def bit_m_50x1(**kwargs):
    return _bit_resnet("bit_m_50x1", **kwargs)


@wraps(ResNetV2)
def bit_m_50x3(**kwargs):
    return _bit_resnet("bit_m_50x3", **kwargs)


@wraps(ResNetV2)
def bit_m_101x1(**kwargs):
    return _bit_resnet("bit_m_101x1", **kwargs)


@wraps(ResNetV2)
def bit_m_101x3(**kwargs):
    return _bit_resnet("bit_m_101x3", **kwargs)


@wraps(ResNetV2)
def bit_m_152x2(**kwargs):
    return _bit_resnet("bit_m_152x2", **kwargs)


@wraps(ResNetV2)
def bit_m_152x4(**kwargs):
    return _bit_resnet("bit_m_152x4", **kwargs)
