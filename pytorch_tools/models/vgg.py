import logging
from copy import deepcopy
from functools import wraps, partial

import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

from pytorch_tools.modules import BlurPool
from pytorch_tools.modules import bn_from_name
from pytorch_tools.utils.misc import initialize
from pytorch_tools.utils.misc import add_docs_for
from pytorch_tools.utils.misc import DEFAULT_IMAGENET_SETTINGS

# avoid overwriting doc string
wraps = partial(wraps, assigned=("__module__", "__name__", "__qualname__", "__annotations__"))


class VGG(nn.Module):
    """
    Args:
        pretrained (str): if present, returns a model pre-trained on 'str' dataset
        num_classes (int, optional): [description]. Defaults to 1000.
        in_channels (int): Number of input (color) channels. Defaults to 3.
        norm_layer (ABN, optional): Which version of ABN to use. Choices are:
            'ABN' - dropin replacement for BN+Relu.
            'InplaceABN' - efficient version. If used with `pretrain` Weights still 
                will be loaded but performance may be worse than with ABN. 
        encoder (bool, optional): Flag to return features with different resolution. 
            Defaults to False.
        antialias (bool, optional): Flag to turn on Rect-2 antialiasing 
            from https://arxiv.org/abs/1904.11486. Defaults to False.
    """

    def __init__(
        self,
        layers,
        pretrained=None,  # not used. here for proper signature.
        num_classes=1000,
        in_channels=3,
        encoder=False,
        antialias=False,
        norm_layer="abn",
        norm_act="relu",
    ):

        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.norm_act = norm_act
        self.norm_layer = bn_from_name(norm_layer)
        self.encoder = encoder
        self.antialias = antialias
        self.features = self._make_layers(layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if not encoder:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        else:
            self.forward = self.encoder_features

        initialize(self)

    def encoder_features(self, x):
        features = []
        for module in self.features:
            if isinstance(module, nn.MaxPool2d):
                features.append(x)
            x = module(x)
        features.append(x)

        features = features[1:]
        features = features[::-1]
        return features

    def logits(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for v in cfg:
            if v == "M":
                if self.antialias:
                    layers += [nn.MaxPool2d(kernel_size=2, stride=1), BlurPool()]
                else:
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, self.norm_layer(v, activation=self.norm_act)]
                in_channels = v
        return nn.Sequential(*layers)

    def load_state_dict(self, state_dict, **kwargs):
        keys = list(state_dict.keys())
        # filter classifier and num_batches_tracked
        for k in keys:
            if k.startswith("classifier") and self.encoder:
                state_dict.pop(k)
        # there is a mismatch in feature layers names, so need this mapping
        self_feature_names = [i for i in self.state_dict().keys() if "features" in i]
        load_feature_names = [i for i in state_dict.keys() if "features" in i]
        features_map = {load_f: self_f for (load_f, self_f) in zip(load_feature_names, self_feature_names)}
        for k in keys:
            if k.startswith("features"):
                state_dict[features_map[k]] = state_dict.pop(k)
        super().load_state_dict(state_dict, **kwargs)


# fmt: off
CFGS = {
    "vgg11_bn": {
        "default": {
            "params": {"layers": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet": {"url": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth"},
    },
    "vgg13_bn": {
        "default": {
            "params": {"layers": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet": {"url": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth"},
    },
    "vgg16_bn": {
        "default": {
            "params": {
                "layers": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
            },
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet": {"url": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"},
    },
    "vgg19_bn": {
        "default": {
            "params": {
                "layers": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
            },
            **DEFAULT_IMAGENET_SETTINGS,
        },
        "imagenet": {"url": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",},
    },
}
# fmt: on


def _vgg(arch, pretrained=None, **kwargs):
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
    model = VGG(**kwargs)
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
            state_dict["classifier.6.weight"] = model.state_dict()["classifier.6.weight"]
            state_dict["classifier.6.bias"] = model.state_dict()["classifier.6.bias"]
        model.load_state_dict(state_dict)
    setattr(model, "pretrained_settings", cfg_settings)
    return model


@wraps(VGG)
@add_docs_for(VGG)
def vgg11_bn(**kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg("vgg11_bn", **kwargs)


@wraps(VGG)
@add_docs_for(VGG)
def vgg13_bn(**kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg("vgg13_bn", **kwargs)


@wraps(VGG)
@add_docs_for(VGG)
def vgg16_bn(**kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg("vgg16_bn", **kwargs)


@wraps(VGG)
@add_docs_for(VGG)
def vgg19_bn(**kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg("vgg19_bn", **kwargs)
