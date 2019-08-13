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
from inplace_abn import ABN
from pytorch_tools.modules import BasicBlock, Bottleneck, SEModule
from pytorch_tools.modules import GlobalPool2d, BlurPool
from pytorch_tools.modules.residual import conv1x1, conv3x3
from collections import OrderedDict

# from .registry import register_model
# from .helpers import load_pretrained
# from .adaptive_avgmax_pool import SelectAdaptivePool2d
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# __all__ = ['ResNet']  # model_registry will add each entrypoint fn to this


# def _cfg(url='', **kwargs):
#     return {
#         'url': url,
#         'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
#         'crop_pct': 0.875, 'interpolation': 'bilinear',
#         'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
#         'first_conv': 'conv1', 'classifier': 'fc',
#         **kwargs
#     }


# default_cfgs = {
#     'resnet18': _cfg(url='https://download.pytorch.org/models/resnet18-5c106cde.pth'),
#     'resnet34': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth'),
#     'resnet26': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26-9aa10e23.pth',
#         interpolation='bicubic'),
#     'resnet26d': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26d-69e92c46.pth',
#         interpolation='bicubic'),
#     'resnet50': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/rw_resnet50-86acaeed.pth',
#         interpolation='bicubic'),
#     'resnet101': _cfg(url='https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'),
#     'resnet152': _cfg(url='https://download.pytorch.org/models/resnet152-b121ed2d.pth'),
#     'tv_resnet34': _cfg(url='https://download.pytorch.org/models/resnet34-333f7ec4.pth'),
#     'tv_resnet50': _cfg(url='https://download.pytorch.org/models/resnet50-19c8e357.pth'),
#     'wide_resnet50_2': _cfg(url='https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth'),
#     'wide_resnet101_2': _cfg(url='https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth'),
#     'resnext50_32x4d': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50_32x4d-068914d1.pth',
#         interpolation='bicubic'),
#     'resnext50d_32x4d': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50d_32x4d-103e99f8.pth',
#         interpolation='bicubic'),
#     'resnext101_32x4d': _cfg(url=''),
#     'resnext101_32x8d': _cfg(url='https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'),
#     'resnext101_64x4d': _cfg(url=''),
#     'tv_resnext50_32x4d': _cfg(url='https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'),
#     'ig_resnext101_32x8d': _cfg(url='https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth'),
#     'ig_resnext101_32x16d': _cfg(url='https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth'),
#     'ig_resnext101_32x32d': _cfg(url='https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth'),
#     'ig_resnext101_32x48d': _cfg(url='https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth'),
# }


# def _get_padding(kernel_size, stride, dilation=1):
#     padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
#     return padding


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
                 num_classes=1000, in_chans=3, use_se=False,
                 groups=1, base_width=64, 
                 deep_stem=False,
                 block_reduce_first=1, down_kernel_size=1, 
                 dilated=False,
                 norm_layer=ABN,
                 norm_act='relu',
                 antialias=False, 
                 encoder=False,
                 drop_rate=0.0, 
                 global_pool='avg',
                 init_bn0=True):

        stem_width = 64
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
            layer0_modules = [
                # as in https://arxiv.org/pdf/1812.01187.pdf Resnet-B
                ('conv1', conv3x3(in_chans, stem_width//2, 2)),
                ('bn1', norm_layer(stem_width // 2, activation=norm_act)),
                ('conv2', conv3x3(stem_width, stem_width//2)),
                ('bn2', norm_layer(stem_width // 2, activation=norm_act)),
                ('conv3', conv3x3(stem_width //2, stem_width, 2)),
                ('bn3', norm_layer(stem_width, activation=norm_act))]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(in_chans, stem_width, kernel_size=7, stride=2, 
                                    padding=3, bias=False)),
                ('bn1', norm_layer(stem_width, activation=norm_act))]
        if antialias:
            layer0_modules += [('maxpool', nn.MaxPool2d(kernel_size=3, padding=1)), 
                               ('blurpool', BlurPool())]
        else:
            layer0_modules += [('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
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
            self.forward = self.forward_features
        
        self._initialize_weights(init_bn0)

    def _make_layer(self, planes, blocks, stride=1, dilation=1, 
                    use_se=False, norm_layer=ABN, norm_act='relu', antialias=False):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample_layers = []
            if antialias:
                downsample_layers += [BlurPool()]
            downsample_layers += [
                conv1x1(self.inplanes, planes * self.expansion, stride=1 if antialias else stride),
                norm_layer(planes * self.expansion, activation='linear')]
            downsample = nn.Sequential(*downsample_layers)
        
        layers = [self.block(
            self.inplanes, planes, stride, downsample, self.groups, 
            self.base_width, use_se, dilation, norm_layer)]
        
        self.inplanes = planes * self.expansion
        for _ in range(1, blocks):
            layers.append(self.block(
                self.inplanes, planes, 1, None, self.groups, 
                self.base_width, use_se, dilation, norm_layer))
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
        raise NotImplementedError

    def forward_features(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        # 
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        if self.encoder:
            return [x4, x3, x2, x1, x0]
        return x4

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.last_linear(x)
        return x

    def load_state_dict(self, state_dict, **kwargs):
        keys = list(state_dict.keys())
        # filter classifier and num_batches_tracked
        for k in keys:
            if k.startswith('fc') and self.encoder:
                state_dict.pop(k)
            if k.endswith('num_batches_tracked'):
                state_dict.pop(k)
        super().load_state_dict(state_dict, **kwargs)

cfg = {
    'resnet18': {'block': BasicBlock, 'layers': [2, 2, 2, 2]},
    'resnet34': {'block': BasicBlock, 'layers': [3, 4, 6, 3]},
    'resnet50': {'block': Bottleneck, 'layers': [3, 4, 6, 3]},
    'resnet101': {'block': Bottleneck, 'layers': [3, 4, 23, 3]},
    'resnet152': {'block': Bottleneck, 'layers': [3, 8, 36, 3]},
    'wide_resnet50_2': {'block': Bottleneck, 'layers': [3, 4, 6, 3], 'base_width':128},
    'wide_resnet101_2': {'block': Bottleneck, 'layers': [3, 4, 23, 3], 'base_width':128},
    'resnext50_32x4d': {'block': Bottleneck, 'layers': [3, 4, 6, 3], 'base_width':4, 'groups':32},
    'resnext101_32x8d': {'block': Bottleneck, 'layers': [3, 4, 23, 3], 'base_width':8, 'groups':32},
    'ig_resnext101_32x8d': {'block': Bottleneck, 'layers': [3, 4, 23, 3], 'base_width':8, 'groups':32},
    'ig_resnext101_32x16d': {'block': Bottleneck, 'layers': [3, 4, 23, 3], 'base_width':16, 'groups':32},
    'ig_resnext101_32x32d': {'block': Bottleneck, 'layers': [3, 4, 23, 3], 'base_width':32, 'groups':32},
    'ig_resnext101_32x48d': {'block': Bottleneck, 'layers': [3, 4, 23, 3], 'base_width':48, 'groups':32},

    'se_resnet34': {'block': BasicBlock, 'layers': [3, 4, 6, 3], 'use_se': True},
    'se_resnet50': {'block': Bottleneck, 'layers': [3, 4, 6, 3], 'use_se': True},
    'se_resnet101': {'block': Bottleneck, 'layers': [3, 4, 23, 3], 'use_se': True},
    'se_resnet152': {'block': Bottleneck, 'layers': [3, 4, 36, 3], 'use_se': True},
    'se_resnext50_32x4d': {'block': Bottleneck, 'layers': [3, 4, 6, 3], 'base_width':4, 'groups':32, 'use_se':True},
    'se_resnext101_32x4d' {'block': Bottleneck, 'layers': [3, 4, 23, 3], 'base_width':4, 'groups':32, 'use_se':True},
}

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'ig_resnext101_32x8d': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth', #88M
    'ig_resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth', #193M
    'ig_resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth', #466M
    'ig_resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth', #829M
    
    'se_resnet50': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
    'se_resnet101': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
    'se_resnet152': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
    'se_resnext50_32x4d': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
    'se_resnext101_32x4d': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
}

def _resnet(arch, pretrained, **kwargs):
    # if pretrained:
    #     kwargs['init_weights'] = False
    model = ResNet(**cfg[arch], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=True)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model."""
    return _resnet('resnet18', pretrained, **kwargs)

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model."""
    return _resnet('resnet34', pretrained, **kwargs)

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model."""
    return _resnet('resnet50', pretrained, **kwargs)

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model."""
    return _resnet('resnet101', pretrained, **kwargs)

def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model."""
    return _resnet('resnet101', pretrained, **kwargs)

def wide_resnet50_2(pretrained=False, **kwargs):
    """Constructs a Wide ResNet-50-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    return _resnet('wide_resnet50_2', pretrained, **kwargs)

def wide_resnet101_2(pretrained=False, **kwargs):
    """Constructs a Wide ResNet-101-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same."""
    return _resnet('wide_resnet101_2', pretrained, **kwargs)

def resnext50_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt50-32x4d model."""
    return _resnet('resnext50_32x4d', pretrained, **kwargs)

def resnext101_32x8d(pretrained=False, **kwargs):
    """Constructs a ResNeXt101-32x8d model."""
    return _resnet('resnext101_32x8d', pretrained, **kwargs)

def ig_resnext101_32x8d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/"""
    return _resnet('ig_resnext101_32x8d', pretrained, **kwargs)

def ig_resnext101_32x16d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data."""
    return _resnet('ig_resnext101_32x16d', pretrained, **kwargs)

def ig_resnext101_32x32d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 32x32 model pre-trained on weakly-supervised data."""
    return _resnet('ig_resnext101_32x32d', pretrained, **kwargs)

def ig_resnext101_32x48d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 32x48 model pre-trained on weakly-supervised data."""
    return _resnet('ig_resnext101_32x48d', pretrained, **kwargs)

def se_resnet34(pretrained=False, **kwargs):
    return _resnet('se_resnet34', pretrained, **kwargs)

def se_resnet50(pretrained=False, **kwargs):
    return _resnet('se_resnet50', pretrained, **kwargs)

def se_resnet101(pretrained=False, **kwargs):
    return _resnet('se_resnet50', pretrained, **kwargs)

def se_resnet152(pretrained=False, **kwargs):
    return _resnet('se_resnet50', pretrained, **kwargs)

def se_resnext50_32x4d(pretrained=False, **kwargs):
    return _resnet('se_resnext50_32x4d', pretrained, **kwargs)

def se_resnext101_32x4d(pretrained=False, **kwargs):
    return _resnet('se_resnext101_32x4d', pretrained, **kwargs)

######################################
#     NOT YET ADDED WEIGHTS BELOW    #
#                                    #
######################################
def resnet26(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-26 model.
    """
    default_cfg = default_cfgs['resnet26']
    model = ResNet(Bottleneck, [2, 2, 2, 2], num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


# 
def resnet26d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNet-26 v1d model.
    This is technically a 28 layer ResNet, sticking with 'd' modifier from Gluon for now.
    """
    default_cfg = default_cfgs['resnet26d']
    model = ResNet(
        Bottleneck, [2, 2, 2, 2], stem_width=32, deep_stem=True, avg_down=True,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model

def resnext50d_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    """Constructs a ResNeXt50d-32x4d model. ResNext50 w/ deep stem & avg pool downsample
    """
    default_cfg = default_cfgs['resnext50d_32x4d']
    model = ResNet(
        Bottleneck, [3, 4, 6, 3], groups=32, base_width=4,
        stem_width=32, deep_stem=True, avg_down=True,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model