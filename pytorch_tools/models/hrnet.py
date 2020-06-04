# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
# Then additionally modified by @bonlime to match other models in pytorch-tools repo

import logging
from copy import deepcopy
from functools import wraps, partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url

from pytorch_tools.modules.residual import conv1x1
from pytorch_tools.modules.residual import conv3x3
from pytorch_tools.modules.residual import BasicBlock
from pytorch_tools.modules.residual import Bottleneck
from pytorch_tools.modules import ABN
from pytorch_tools.modules import bn_from_name
from pytorch_tools.utils.misc import initialize
from pytorch_tools.utils.misc import add_docs_for
from pytorch_tools.utils.misc import repeat_channels
from pytorch_tools.utils.misc import DEFAULT_IMAGENET_SETTINGS

# avoid overwriting doc string
wraps = partial(wraps, assigned=("__module__", "__name__", "__qualname__", "__annotations__"))


# simplified version of _make_layer from Resnet
def make_layer(inplanes, planes, blocks, norm_layer=ABN, norm_act="relu"):
    block = Bottleneck
    bn_args = {"norm_layer": norm_layer, "norm_act": norm_act}
    if inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion),
            norm_layer(planes * block.expansion, activation=norm_act),
        )

    layers = []
    layers.append(block(inplanes, planes, downsample=downsample, **bn_args))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, **bn_args))
    return nn.Sequential(*layers)


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,  # number of parallel branches
        num_blocks,  # number of blocks
        num_channels,
        norm_layer=ABN,
        norm_act="relu",
    ):
        super(HighResolutionModule, self).__init__()
        self.block = BasicBlock
        self.num_branches = num_branches  # used in forward
        self.num_inchannels = num_channels
        self.bn_args = {"norm_layer": norm_layer, "norm_act": norm_act}
        branches = [self._make_branch(n_bl, n_ch) for n_bl, n_ch in zip(num_blocks, num_channels)]
        self.branches = nn.ModuleList(branches)
        self.fuse_layers = self._make_fuse_layers(norm_layer, norm_act)
        self.relu = nn.ReLU(False)

    def _make_branch(self, b_blocks, b_channels):
        return nn.Sequential(*[self.block(b_channels, b_channels, **self.bn_args) for _ in range(b_blocks)])

    # fmt: off
    # don't want to rewrite this piece it's too fragile
    def _make_fuse_layers(self, norm_layer, norm_act):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        conv1x1(num_inchannels[j], num_inchannels[i]),
                        norm_layer(num_inchannels[i], activation="identity"),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                conv3x3(num_inchannels[j], num_outchannels_conv3x3, 2),
                                norm_layer(num_outchannels_conv3x3, activation="identity")))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                conv3x3(num_inchannels[j], num_outchannels_conv3x3, 2),
                                norm_layer(num_outchannels_conv3x3, activation=norm_act)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)
    # fmt: on

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        x = [branch(x_i) for branch, x_i in zip(self.branches, x)]

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class TransitionBlock(nn.Module):
    """Transition is where new branches for smaller resolution are born
    -- ==> --
    
    -- ==> --
      \
       \=> --
    """

    def __init__(self, prev_channels, current_channels, norm_layer=ABN, norm_act="relu"):
        super().__init__()
        transition_layers = []
        for prev_ch, curr_ch in zip(prev_channels, current_channels):
            if prev_ch != curr_ch:
                # this case only happens between 1st and 2nd stage
                layers = [conv3x3(prev_ch, curr_ch), norm_layer(curr_ch, activation=norm_act)]
                transition_layers.append(nn.Sequential(*layers))
            else:
                transition_layers.append(nn.Identity())

        if len(current_channels) > len(prev_channels):  # only works for ONE extra branch
            layers = [
                conv3x3(prev_channels[-1], current_channels[-1], 2),
                norm_layer(current_channels[-1], activation=norm_act),
            ]
            transition_layers.append(nn.Sequential(*layers))
        self.trans_layers = nn.ModuleList(transition_layers)

    def forward(self, x):  # x is actually an array
        out_x = [trans_l(x_i) for x_i, trans_l in zip(x, self.trans_layers)]
        out_x.append(self.trans_layers[-1](x[-1]))
        return out_x


class HRClassificationHead(nn.Module):
    def __init__(self, pre_channels, norm_layer=ABN, norm_act="relu"):
        super().__init__()
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]
        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for (pre_c, head_c) in zip(pre_channels, head_channels):
            incre_modules.append(make_layer(pre_c, head_c, 1, norm_layer, norm_act))
        self.incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_channels) - 1):
            in_ch = head_channels[i] * head_block.expansion
            out_ch = head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(
                conv3x3(in_ch, out_ch, 2, bias=True), norm_layer(out_ch, activation=norm_act)
            )
            downsamp_modules.append(downsamp_module)
        self.downsamp_modules = nn.ModuleList(downsamp_modules)

        self.final_layer = nn.Sequential(
            conv1x1(head_channels[3] * head_block.expansion, 2048, bias=True),
            norm_layer(2048, activation=norm_act),
        )

    def forward(self, x):
        x = [self.incre_modules[i](x[i]) for i in range(4)]
        for i in range(1, 4):
            x[i] = x[i] + self.downsamp_modules[i - 1](x[i - 1])
        return self.final_layer(x[3])


class HighResolutionNet(nn.Module):
    """HighResolution Nets constructor
    Supports any width and small version of the network 

    Ref: 
        * HRNet paper https://arxiv.org/abs/1908.07919


    Args:
        width (int):
            Width of the branch with highest resolution. Other branches are scales accordingly. Better don't pass mannualy.
        small (bool): Flag to construct smaller version of the model with less number of blocks in each stage.
        pretrained (str, optional):
            If not, returns a model pre-trained on 'str' dataset. `imagenet` is available
            for every model but some have more. Check the code to find out.
        num_classes (int):
            Number of classification classes. Defaults to 1000.
        in_channels (int):
            Number of input (color) channels. Defaults to 3.
        norm_layer (str):
            Normalization layer to use. One of 'abn', 'inplaceabn'. The inplace version lowers memory footprint.
            But increases backward time. Defaults to 'abn'.
        norm_act (str):
            Activation for normalizion layer. It's reccomended to use `leacky_relu` with `inplaceabn`. Default: 'relu'
        encoder (bool):
            Flag to overwrite forward pass to return 5 tensors with different resolutions. Defaults to False.
            NOTE: HRNet first features have resolution 4x times smaller than input, not 2x as all other models. 
            So it CAN'T be used as encoder in Unet and Linknet models 
    """

    # drop_rate (float):
    #     Dropout probability before classifier, for training. Defaults to 0.
    def __init__(
        self,
        width=18,
        small=False,
        pretrained=None,  # not used. here for proper signature
        num_classes=1000,
        in_channels=3,
        norm_layer="abn",
        norm_act="relu",
        encoder=False,
    ):
        super(HighResolutionNet, self).__init__()
        stem_width = 64
        norm_layer = bn_from_name(norm_layer)
        self.bn_args = bn_args = {"norm_layer": norm_layer, "norm_act": norm_act}
        self.conv1 = conv3x3(in_channels, stem_width, stride=2)
        self.bn1 = norm_layer(stem_width, activation=norm_act)

        self.conv2 = conv3x3(stem_width, stem_width, stride=2)
        self.bn2 = norm_layer(stem_width, activation=norm_act)

        channels = [width, width * 2, width * 4, width * 8]
        n_blocks = [2 if small else 4] * 4

        self.layer1 = make_layer(stem_width, stem_width, n_blocks[0], **bn_args)

        self.transition1 = TransitionBlock([stem_width * Bottleneck.expansion], channels[:2], **bn_args)
        self.stage2 = self._make_stage(n_modules=1, n_branches=2, n_blocks=n_blocks[:2], n_chnls=channels[:2])

        self.transition2 = TransitionBlock(channels[:2], channels[:3], **bn_args)
        self.stage3 = self._make_stage(  # 3 if small else 4
            n_modules=(4, 3)[small], n_branches=3, n_blocks=n_blocks[:3], n_chnls=channels[:3]
        )

        self.transition3 = TransitionBlock(channels[:3], channels, **bn_args)
        self.stage4 = self._make_stage(  # 2 if small else 3
            n_modules=(3, 2)[small], n_branches=4, n_blocks=n_blocks, n_chnls=channels,
        )

        self.encoder = encoder
        if encoder:
            self.forward = self.encoder_features
        else:
            # Classification Head
            self.cls_head = HRClassificationHead(channels, **bn_args)
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.last_linear = nn.Linear(2048, num_classes)
        # initialize weights
        initialize(self)

    def _make_stage(self, n_modules, n_branches, n_blocks, n_chnls):
        modules = []
        for i in range(n_modules):
            modules.append(HighResolutionModule(n_branches, n_blocks, n_chnls, **self.bn_args,))
        return nn.Sequential(*modules)

    def encoder_features(self, x):
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.layer1(x)

        x = self.transition1([x])  # x is actually a list now
        x = self.stage2(x)

        x = self.transition2(x)
        x = self.stage3(x)

        x = self.transition3(x)
        x = self.stage4(x)
        if self.encoder:  # want to return from lowest resolution to highest
            x = [x[3], x[2], x[1], x[0], x[0]]
        return x

    def features(self, x):
        x = self.encoder_features(x)
        x = self.cls_head(x)
        return x

    def logits(self, x):
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        #         x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

    def load_state_dict(self, state_dict, **kwargs):
        self_keys = list(self.state_dict().keys())
        sd_keys = list(state_dict.keys())
        sd_keys = [k for k in sd_keys if "num_batches_tracked" not in k]  # filter
        new_state_dict = {}
        for new_key, old_key in zip(self_keys, sd_keys):
            new_state_dict[new_key] = state_dict[old_key]
        super().load_state_dict(new_state_dict, **kwargs)


# fmt: off
CFGS = {
    "hrnet_w18_small": {
        "default": {"params": {"width": 18, "small": True}, **DEFAULT_IMAGENET_SETTINGS,},
        "imagenet": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.2/hrnet_w18_small_model_v2-a6eb6c92.pth"},
    },
    "hrnet_w18": {
        "default": {"params": {"width": 18}, **DEFAULT_IMAGENET_SETTINGS,},
        "imagenet": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.2/hrnetv2_w18_imagenet_pretrained-00eb2006.pth"},
    },
    "hrnet_w30": {
        "default": {"params": {"width": 30}, **DEFAULT_IMAGENET_SETTINGS,},
        "imagenet": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.2/hrnetv2_w30_imagenet_pretrained-11fb7730.pth"},
    },
    "hrnet_w32": {
        "default": {"params": {"width": 32}, **DEFAULT_IMAGENET_SETTINGS,},
        "imagenet": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.2/hrnetv2_w32_imagenet_pretrained-dc9eeb4f.pth"},
    },
    "hrnet_w40": {
        "default": {"params": {"width": 40}, **DEFAULT_IMAGENET_SETTINGS,},
        "imagenet": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.2/hrnetv2_w40_imagenet_pretrained-ed0b031c.pth"},
    },
    "hrnet_w44": {
        "default": {"params": {"width": 44}, **DEFAULT_IMAGENET_SETTINGS,},
        "imagenet": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.2/hrnetv2_w44_imagenet_pretrained-8c55086c.pth"},
    },
    "hrnet_w48": {
        "default": {"params": {"width": 48}, **DEFAULT_IMAGENET_SETTINGS,},
        "imagenet": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.2/hrnetv2_w48_imagenet_pretrained-0efec102.pth"},
    },
    "hrnet_w64": { # there are weights for this model too I just didn't add them
        "default": {"params": {"width": 60}, **DEFAULT_IMAGENET_SETTINGS,},
        "imagenet": {"url": None},
    },
}

# fmt:on


def _hrnet(arch, pretrained=None, **kwargs):
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
    model = HighResolutionNet(**kwargs)
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
            state_dict["classifier.weight"] = model.state_dict()["last_linear.weight"]
            state_dict["classifier.bias"] = model.state_dict()["last_linear.bias"]
        # support pretrained for custom input channels
        if kwargs.get("in_channels", 3) != 3:
            old_weights = state_dict.get("conv1.weight")
            state_dict["conv1.weight"] = repeat_channels(old_weights, kwargs["in_channels"])
        model.load_state_dict(state_dict)
    setattr(model, "pretrained_settings", cfg_settings)
    return model


@wraps(HighResolutionNet)
@add_docs_for(HighResolutionNet)
def hrnet_w18_small(**kwargs):
    r"""Constructs a HRNetv2-18 small model."""
    return _hrnet("hrnet_w18_small", **kwargs)


@wraps(HighResolutionNet)
@add_docs_for(HighResolutionNet)
def hrnet_w18(**kwargs):
    r"""Constructs a HRNetv2-18 model."""
    return _hrnet("hrnet_w18", **kwargs)


@wraps(HighResolutionNet)
@add_docs_for(HighResolutionNet)
def hrnet_w30(**kwargs):
    r"""Constructs a HRNetv2-30 model."""
    return _hrnet("hrnet_w30", **kwargs)


@wraps(HighResolutionNet)
@add_docs_for(HighResolutionNet)
def hrnet_w32(**kwargs):
    r"""Constructs a HRNetv2-32 model."""
    return _hrnet("hrnet_w32", **kwargs)


@wraps(HighResolutionNet)
@add_docs_for(HighResolutionNet)
def hrnet_w40(**kwargs):
    r"""Constructs a HRNetv2-40 model."""
    return _hrnet("hrnet_w40", **kwargs)


@wraps(HighResolutionNet)
@add_docs_for(HighResolutionNet)
def hrnet_w44(**kwargs):
    r"""Constructs a HRNetv2-44 model."""
    return _hrnet("hrnet_w44", **kwargs)


@wraps(HighResolutionNet)
@add_docs_for(HighResolutionNet)
def hrnet_w48(**kwargs):
    r"""Constructs a HRNetv2-48 model."""
    return _hrnet("hrnet_w48", **kwargs)


@wraps(HighResolutionNet)
@add_docs_for(HighResolutionNet)
def hrnet_w64(**kwargs):
    r"""Constructs a HRNetv2-64 model."""
    return _hrnet("hrnet_w64", **kwargs)
