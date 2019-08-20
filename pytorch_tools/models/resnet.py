"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants added by Ross Wightman
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url

from pytorch_tools.modules import BasicBlock, Bottleneck, SEModule
from pytorch_tools.modules import GlobalPool2d, BlurPool
from pytorch_tools.modules.residual import conv1x1, conv3x3
from pytorch_tools.utils.misc import bn_from_name
from pytorch_tools.utils.misc import add_docs_for
from pytorch_tools.utils.misc import DEFAULT_IMAGENET_SETTINGS
from collections import OrderedDict
from functools import wraps, partial
# avoid overwriting doc string
wraps = partial(wraps, assigned=('__module__', '__name__', '__qualname__', '__annotations__'))

class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants:
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32
      * d - 3 layer deep 3x3 stem, stem_width = 32, average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64, average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard groups and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, groups=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlock, Bottleneck.
    layers : list of int
        Numbers of layers in each block
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    use_se : bool, default False
        Enable Squeeze-Excitation module in blocks
    groups : int, default 1
        Number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64
        Factor determining bottleneck channels. `planes * base_width / 64 * groups`
    deep_stem : bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    stem_width : int, default 64
        Number of channels in stem convolutions
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    global_pool : str, default 'avg'
        Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    init_bn0 : Zero-initialize the last BN in each residual branch,
        so that the residual branch starts with zeros, and each residual block behaves like an identity.
        This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

    """
    def __init__(self, block=None, layers=None,
                     pretrained=None,
                     num_classes=1000, in_chans=3, use_se=False,
                     groups=1, base_width=64,
                     deep_stem=False,
                     block_reduce_first=1, down_kernel_size=1,
                     dilated=False,
                     norm_layer='abn',
                     antialias=False,
                     encoder=False,
                     drop_rate=0.0,
                     global_pool='avg',
                     init_bn0=True):

        stem_width = 64
        norm_act = 'relu' if norm_layer.lower() == 'abn' else 'leaky_relu'
        norm_layer = bn_from_name(norm_layer)
        self.inplanes = stem_width
        self.num_classes = num_classes
        self.groups = groups
        self.base_width = base_width
        self.drop_rate = drop_rate
        self.block = block
        self.expansion = block.expansion
        self.dilated = dilated
        self.norm_act = norm_act
        super(ResNet, self).__init__()

        # no antialias in stem to avoid huge computations
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv3x3(in_chans, stem_width // 2, 2),
                norm_layer(stem_width // 2, activation=norm_act),
                conv3x3(stem_width, stem_width // 2),
                norm_layer(stem_width // 2, activation=norm_act),
                conv3x3(stem_width // 2, stem_width, 2)
            )
        else:
            self.conv1 = nn.Conv2d(in_chans, stem_width, kernel_size=7, stride=2,
                                   padding=3, bias=False)
        self.bn1 = norm_layer(stem_width, activation=norm_act)
        if antialias:
            self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                         BlurPool())
        else:
            # for se resnets fist maxpool is slightly different
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
                                        padding=0 if use_se else 1,
                                        ceil_mode=True if use_se else False)
        # Output stride is 8 with dilated and 32 without
        stride_3_4 = 1 if self.dilated else 2
        dilation_3 = 2 if self.dilated else 1
        dilation_4 = 4 if self.dilated else 1
        largs = dict(use_se=use_se, norm_layer=norm_layer,
                     norm_act=norm_act, antialias=antialias)
        self.layer1 = self._make_layer(64, layers[0], stride=1, **largs)
        self.layer2 = self._make_layer(128, layers[1], stride=2, **largs)
        self.layer3 = self._make_layer(256, layers[2], stride=stride_3_4, dilation=dilation_3, **largs)
        self.layer4 = self._make_layer(512, layers[3], stride=stride_3_4, dilation=dilation_4, **largs)
        self.global_pool = GlobalPool2d(global_pool)
        self.num_features = 512 * self.expansion
        self.encoder = encoder
        if not encoder:
            self.last_linear = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes)
        else:
            self.forward = self.encoder_features

        self._initialize_weights(init_bn0)

    def _make_layer(self, planes, blocks, stride=1, dilation=1,
                    use_se=None, norm_layer=None, norm_act=None, antialias=None):
        downsample = None

        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample_layers = []
            if antialias and stride == 2:  # using OrderedDict to preserve ordering and allow loading
                downsample_layers += [('blur', BlurPool())]
            downsample_layers += [
                ('0', conv1x1(self.inplanes, planes * self.expansion, stride=1 if antialias else stride)),
                ('1', norm_layer(planes * self.expansion, activation='identity'))]
            downsample = nn.Sequential(OrderedDict(downsample_layers))

        layers = [self.block(
            self.inplanes, planes, stride, downsample, self.groups,
            self.base_width, use_se, dilation, norm_layer, norm_act, antialias)]

        self.inplanes = planes * self.expansion
        for _ in range(1, blocks):
            layers.append(self.block(
                self.inplanes, planes, 1, None, self.groups, self.base_width,
                use_se, dilation, norm_layer, norm_act, antialias))
        return nn.Sequential(*layers)

    def _initialize_weights(self, init_bn0=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.norm_act)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
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
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.last_linear(x)

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

    def load_state_dict(self, state_dict, **kwargs):
        keys = list(state_dict.keys())
        # filter classifier and num_batches_tracked
        for k in keys:
            if k.startswith('fc') and self.encoder:
                state_dict.pop(k)
            elif k.startswith('fc'):
                state_dict[k.replace('fc', 'last_linear')] = state_dict.pop(k)
            if k.startswith('layer0'):
                state_dict[k.replace('layer0.', '')] = state_dict.pop(k)
        super().load_state_dict(state_dict, **kwargs)

cfgs = {
    # RESNET MODELS
    'resnet18': {
        'default': {
            'params' : {'block': BasicBlock, 'layers': [2, 2, 2, 2]},
            **DEFAULT_RESNET_SETTINGS,
        },
        'imagenet': {'url' : 'https://download.pytorch.org/models/resnet18-5c106cde.pth'},
        # EXAMPLE
        # 'imagenet_inplaceabn': {
        #     'params': {'block': BasicBlock, 'layers': [2, 2, 2, 2], 'norm_layer': 'inplaceabn', 'deepstem':True, 'antialias':True},
        #     'url' : 'pathtomodel',
        #     **DEFAULT_RESNET_SETTINGS,
        # }
    },
    'resnet34': {
        'default': {
            'params' : {'block': BasicBlock, 'layers': [3, 4, 6, 3]},
            **DEFAULT_RESNET_SETTINGS,
        }, 
        'imagenet': {'url' : 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'}, 
        'imagenet2': { # weigths from rwightman. TODO: test accuracy
            'url' : 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth',
        },
    },
    'resnet50': {
        'default': {
            'params' : {'block': Bottleneck, 'layers': [3, 4, 6, 3]},
            **DEFAULT_RESNET_SETTINGS,
        },
        'imagenet': {'url' : 'https://download.pytorch.org/models/resnet50-19c8e357.pth'},
    },
    'resnet101': {
        'default': {
            'params' : {'block': Bottleneck, 'layers': [3, 4, 23, 3]},
            **DEFAULT_RESNET_SETTINGS,
        },
        'imagenet': {'url' : 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'},
    },
    'resnet152': {
        'default': {
            'params' : {'block': Bottleneck, 'layers': [3, 8, 36, 3]},
            **DEFAULT_RESNET_SETTINGS,
        },
        'imagenet': {'url' : 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'},
    },
    # WIDE RESNET MODELS
    'wide_resnet50_2': {
        'default': {
            'params' : {'block': Bottleneck, 'layers': [3, 4, 6, 3], 'base_width': 128},
            **DEFAULT_RESNET_SETTINGS,
        },
        'imagenet': {'url' : 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth'},
    },
    'wide_resnet101_2': {
        'default': {
            'params' : {'block': Bottleneck, 'layers': [3, 4, 23, 3], 'base_width': 128},
            **DEFAULT_RESNET_SETTINGS,
        },
        'imagenet': {'url' : 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth'},
    },
    # RESNEXT MODELS
    'resnext50_32x4d': {
        'default': {
            'params' : {'block': Bottleneck, 'layers': [3, 4, 6, 3], 'base_width': 4, 'groups': 32},
            **DEFAULT_RESNET_SETTINGS,
        },
        'imagenet': {'url' : 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'},
        # weights from rwightman
        'imagenet2': {'url' : 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50d_32x4d-103e99f8.pth'},
    },
    'resnext101_32x8d' : {
        'default': {
            'params': {'block': Bottleneck, 'layers': [3, 4, 23, 3], 'base_width': 8, 'groups': 32},
            **DEFAULT_RESNET_SETTINGS,
        },
        'imagenet': {'url': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'},
        #pretrained on weakly labeled instagram and then tuned on Imagenet
        'imagenet_ig': {'url': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth'}
    },
    'resnext101_32x16d': {
        'default': { 
            'params' : {'block': Bottleneck, 'layers': [3, 4, 23, 3], 'base_width': 16, 'groups': 32},
            **DEFAULT_RESNET_SETTINGS,
        },
        #pretrained on weakly labeled instagram and then tuned on Imagenet
        'imagenet_ig': {'url' : 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth'}
    },
    'resnext101_32x32d': {
        'default': {
            'params' : {'block': Bottleneck, 'layers': [3, 4, 23, 3], 'base_width': 32, 'groups': 32},
            **DEFAULT_RESNET_SETTINGS,
        },
        #pretrained on weakly labeled instagram and then tuned on Imagenet
        'imagenet_ig': {'url' : 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth'}
    },
    'resnext101_32x48d': {
        'default': { #actually it's imagenet_ig. pretrained on weakly labeled instagram and then tuned on Imagenet 
            'params' : {'block': Bottleneck, 'layers': [3, 4, 23, 3], 'base_width': 48, 'groups': 32},
            **DEFAULT_RESNET_SETTINGS,
        },
        'imagenet_ig':{'url' : 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth'}
    },
    # SE RESNET MODELS
    'se_resnet50': {
        'default': {
            'params' : {'block': Bottleneck, 'layers': [3, 4, 6, 3], 'use_se': True},
            'url' : 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            **DEFAULT_RESNET_SETTINGS,
        },
        'imagenet': {'url' : 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth'},
    },
    'se_resnet101': {
        'default': {
            'params' : {'block': Bottleneck, 'layers': [3, 4, 23, 3], 'use_se': True},
            **DEFAULT_RESNET_SETTINGS,
        },
        'imagenet': {'url' : 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth'},
    },
    'se_resnet152': {
        'default': {
            'params' : {'block': Bottleneck, 'layers': [3, 4, 36, 3], 'use_se': True},
            **DEFAULT_RESNET_SETTINGS,
        },
        'imagenet': {'url' : 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth'},
    },
    # SE RESNEXT MODELS
    'se_resnext50_32x4d': {
        'default': {
            'params' : {'block': Bottleneck, 'layers': [3, 4, 6, 3], 'base_width': 4, 'groups': 32, 'use_se': True},
            **DEFAULT_RESNET_SETTINGS,
        },
        'imagenet': {'url' : 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth'}
    },
    'se_resnext101_32x4d': {
        'default': {
            'params' : {'block': Bottleneck, 'layers': [3, 4, 23, 3], 'base_width': 4, 'groups': 32, 'use_se': True},
            **DEFAULT_RESNET_SETTINGS,
        },
        'imagenet': {'url' : 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth'}
    },
}

def _resnet(arch, pretrained=None, **kwargs):
    cfg_params = cfgs[arch]['default']['params']
    if pretrained and cfgs[arch][pretrained].get('params'):
        cfg_params.update(cfgs[arch][pretrained]['params'])
    # TODO maybe change default params
    common_args = set(cfg_params.keys()).intersection(set(kwargs.keys()))
    assert common_args == set(), "Args {} are going to be overwritten by default params for {} weights".format(common_args.keys(), pretrained)
    kwargs.update(cfg_params)
    model = ResNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(cfgs[arch][pretrained]['url'])
        model.load_state_dict(state_dict)
    return model

@wraps(ResNet)
@add_docs_for(ResNet)
def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    return  _resnet('resnet18', **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    return  _resnet('resnet34', **kwargs)

@wraps(ResNet)
@add_docs_for(ResNet)
def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    return  _resnet('resnet50', **kwargs)

@wraps(ResNet)
@add_docs_for(ResNet)
def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    return  _resnet('resnet101', **kwargs)

@wraps(ResNet)
@add_docs_for(ResNet)
def resnet152(**kwargs):
    """Constructs a ResNet-152 model."""
    return  _resnet('resnet152', **kwargs)

@wraps(ResNet)
@add_docs_for(ResNet)
def wide_resnet50_2(**kwargs):
    """Constructs a Wide ResNet-50-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    return  _resnet('wide_resnet50_2', **kwargs)

@wraps(ResNet)
@add_docs_for(ResNet)
def wide_resnet101_2(**kwargs):
    """Constructs a Wide ResNet-101-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same."""
    return  _resnet('wide_resnet101_2', **kwargs)

@wraps(ResNet)
@add_docs_for(ResNet)
def resnext50_32x4d(**kwargs):
    """Constructs a ResNeXt50-32x4d model."""
    return  _resnet('resnext50_32x4d', **kwargs)

@wraps(ResNet)
@add_docs_for(ResNet)
def resnext101_32x8d(**kwargs):
    """Constructs a ResNeXt101-32x8d model."""
    return  _resnet('resnext101_32x8d', **kwargs)

@wraps(ResNet)
@add_docs_for(ResNet)
def ig_resnext101_32x8d(**kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/"""
    return  _resnet('ig_resnext101_32x8d', **kwargs)

@wraps(ResNet)
@add_docs_for(ResNet)
def ig_resnext101_32x16d(**kwargs):
    """Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data."""
    return  _resnet('ig_resnext101_32x16d', **kwargs)

@wraps(ResNet)
@add_docs_for(ResNet)
def ig_resnext101_32x32d(**kwargs):
    """Constructs a ResNeXt-101 32x32 model pre-trained on weakly-supervised data."""
    return  _resnet('ig_resnext101_32x32d', **kwargs)

@wraps(ResNet)
@add_docs_for(ResNet)
def ig_resnext101_32x48d(**kwargs):
    """Constructs a ResNeXt-101 32x48 model pre-trained on weakly-supervised data."""
    return  _resnet('ig_resnext101_32x48d', **kwargs)

# @wraps(ResNet)
# @add_docs_for(ResNet)
# def se_resnet34(**kwargs):
#     """TODO: Add Doc"""
#     return  _resnet('se_resnet34', **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def se_resnet50(**kwargs):
    """TODO: Add Doc"""
    return  _resnet('se_resnet50', **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def se_resnet101(**kwargs):
    """TODO: Add Doc"""
    return  _resnet('se_resnet101', **kwargs)


@wraps(ResNet) 
@add_docs_for(ResNet)
def se_resnet152(**kwargs):
    """TODO: Add Doc"""
    return  _resnet('se_resnet152', **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def se_resnext50_32x4d(**kwargs):
    """TODO: Add Doc"""
    return  _resnet('se_resnext50_32x4d', **kwargs)


@wraps(ResNet)
@add_docs_for(ResNet)
def se_resnext101_32x4d(**kwargs):
    """TODO: Add Doc"""
    return  _resnet('se_resnext101_32x4d', **kwargs)