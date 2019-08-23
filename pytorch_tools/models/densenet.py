"""PyTorch DenseNet

Copied from https://github.com/gpleiss/efficient_densenet_pytorch
and then modified to support additional parameters

Models fully compatible with default torchvision weights
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url
import torch.utils.checkpoint as cp

from pytorch_tools.modules import BasicBlock, Bottleneck, SEModule, Flatten
from pytorch_tools.modules import GlobalPool2d, BlurPool, Transition
from pytorch_tools.modules.residual import conv1x1, conv3x3
from pytorch_tools.utils.misc import activation_from_name, bn_from_name
from pytorch_tools.utils.misc import add_docs_for
from pytorch_tools.utils.misc import DEFAULT_IMAGENET_SETTINGS
from collections import OrderedDict
from functools import wraps, partial
from copy import deepcopy
import logging

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)



class DenseNet(nn.Module):
    r"""DenseNet model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    
    DenseNet variants:
      * normal - 7x7 stem, num_init_features = 64 (96 for densenet161), same as torchvision DenseNet
      * bc - added compression and bottleneck layers

    Parameters
    ----------
    growth_rate :  int
        How many filters to add each layer (`k` in paper)
    block_config : (list of 4 ints) 
        How many layers in each pooling block
    num_init_features : int, default 64
        Number of filters to learn in the first convolution layer
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    deep_stem : bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    compression: float, default 0.
        Decrease number of feature maps in transition layer. 'theta' in paper
    antialias: bool, default False
        Use antialias
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    norm_layer : str, default 'bn'
        Normalization layer type. One of 'bn', 'abn', 'inplaceabn'
    norm_act : str, default 'relu'
        Normalization activation type. 
    global_pool : str, default 'avg'
        Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    init_bn0 : bool, default True
        Zero-initialize the last BN in each residual branch,
        so that the residual branch starts with zeros, and each residual block behaves like an identity.
        This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    memory_efficient : bool, default True
        Use checkpointing. Much more memory efficient, but slower.
        See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=None, block_config=None,
                 num_init_features=64,
                 num_classes=1000,
                 drop_rate=0.0,
                 in_chans=3,
                 deep_stem=False,
                 compression=0.,
                 antialias=False,
                #  encoder=False,
                 norm_layer='bn',
                 norm_act='relu',
                 global_pool='avg',
                 activation_param=None,
                 memory_efficient=False):

        self.num_classes = num_classes
        self.num_init_features = num_init_features
        self.drop_rate = drop_rate
        self.norm_layer = bn_from_name(norm_layer)
        self.norm_act = activation_from_name(norm_act, activation_param)
        self.activation_param = activation_param
        super(DenseNet, self).__init__()

        # Feature block
        if deep_stem:
            self.conv0 = nn.Sequential(
                conv3x3(in_chans, num_init_features // 2, 2),
                norm_layer(num_init_features // 2, activation=norm_act),
                conv3x3(num_init_features // 2, num_init_features // 2),
                norm_layer(num_init_features // 2, activation=norm_act),
                conv3x3(num_init_features // 2, num_init_features, 2)
            )
        else:
            self.conv0 = nn.Conv2d(in_chans, num_init_features, kernel_size=7, stride=2,
                                   padding=3, bias=False)
        self.norm0 = norm_layer(num_init_features)  # vanilla BatchNorm by default
        self.relu0 = self.norm_act
        if antialias:
            self.pool0 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                         BlurPool())
        else:
            self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

        self.global_pool = GlobalPool2d(global_pool)
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final normalization
        self.norm5 = norm_layer(self.num_features)  # Define num_features!!!

        largs = dict(use_se=use_se, norm_layer=norm_layer,
                     norm_act=norm_act, antialias=antialias)
        self.denseblock1 = self._make_layer(64, layers[0], stride=1, **largs)
        self.transition1 = 0
        self.denseblock2 = 0
        self.transition2 = 0
        self.denseblock3 = 0
        self.transition3 = 0
        self.denseblock4 = 0


        self.layer2 = self._make_layer(128, layers[1], stride=2, **largs)
        self.layer3 = self._make_layer(256, layers[2], stride=stride_3_4, dilation=dilation_3, **largs)
        self.layer4 = self._make_layer(512, layers[3], stride=stride_3_4, dilation=dilation_4, **largs)
        self.global_pool = GlobalPool2d(global_pool)
        self.num_features = 512 * self.expansion
        self.encoder = encoder

        # Linear layer
        if not encoder:
            self.classifier = nn.Linear(self.num_features, num_classes)
            self.last_linear = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes)
        else:
            self.forward = self.encoder_features
        self.classifier = nn.Linear(num_features, num_classes)

        self._initialize_weights(init_bn0)

    def features(self, x):
        x = self.relu0(self.norm0(self.conv0(x)))
        x = self.pool0(x)
        x = self.transition1(self.denseblock1(x))
        x = self.transition2(self.denseblock2(x))
        x = self.transition3(self.denseblock3(x))
        x = self.denseblock4(x)
        x = self.norm5(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x   

    def _initialize_weights(self):
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.norm_act)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

CFGS = {
    'densenet121': {
        'default': {
            'params': {'growth_rate': 32, 'layers': (6, 12, 24, 16), norm_layer='bn', },
            **DEFAULT_IMAGENET_SETTINGS,
        },
        'imagenet': {'url': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'},
        # EXAMPLE RESNET
        # 'imagenet_inplaceabn': {
        #     'params': {'block': BasicBlock, 'layers': [2, 2, 2, 2], 'norm_layer': 'inplaceabn', 'deepstem':True, 'antialias':True},
        #     'url' : 'pathtomodel',
        #     **DEFAULT_IMAGENET_SETTINGS,
        # }
    },
    'densenet161': {
        'default': {
            'params': {'growth_rate': 48, 'block_config': (6, 12, 36, 24), 'num_init_features': 96},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        'imagenet': {'url': 'https://download.pytorch.org/models/densenet161-8d451a50.pth'},
    },
    'densenet169': {
        'default': {
            'params': {'growth_rate': 32, 'block_config': (6, 12, 32, 32)},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        'imagenet': {'url': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth'},
    },
    'densenet201': {
        'default': {
            'params': {'growth_rate': 32, 'block_config': (6, 12, 48, 32)},
            **DEFAULT_IMAGENET_SETTINGS,
        },
        'imagenet': {'url': 'https://download.pytorch.org/models/densenet201-c1103571.pth'},
    },
    # DenseNet_BC 
    'densenet121_bc': {
        'default': {
            'params': {'growth_rate': 32, 'layers': (6, 12, 24, 16) },
            **DEFAULT_IMAGENET_SETTINGS,
        },
    },
    'densenet161_bc': {
        'default': {
            'params': {'growth_rate': 48, 'block_config': (6, 12, 36, 24), 'num_init_features': 96},
            **DEFAULT_IMAGENET_SETTINGS,
        },
    },
    'densenet169_bc': {
        'default': {
            'params': {'growth_rate': 32, 'block_config': (6, 12, 32, 32)},
            **DEFAULT_IMAGENET_SETTINGS,
        },
    },
    'densenet201_bc': {
        'default': {
            'params': {'growth_rate': 32, 'block_config': (6, 12, 48, 32)},
            **DEFAULT_IMAGENET_SETTINGS,
        },
    },
}


def _densenet(arch, pretrained=None, progress=True, **kwargs):
    """
        Args:
        pretrained (str or None): if present, returns a model pre-trained on 'str' dataset
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    cfgs = deepcopy(CFGS)
    cfg_settings = cfgs[arch]['default']
    cfg_params = cfg_settings.pop('params')
    if pretrained:
        pretrained_settings = cfgs[arch][pretrained]
        pretrained_params = pretrained_settings.pop('params', {})
        cfg_settings.update(pretrained_settings)
        cfg_params.update(pretrained_params)

    common_args = set(cfg_params.keys()).intersection(set(kwargs.keys()))
    assert common_args == set(), "Args {} are going to be overwritten by default params for {} weights".format(common_args.keys(), pretrained)
    kwargs.update(cfg_params)
    model = DenseNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(cfgs[arch][pretrained]['url'])
        kwargs_cls = kwargs.get('num_classes', None)
        if kwargs_cls and kwargs_cls != cfg_settings['num_classes']:
            logging.warning('Using model pretrained for {} classes with {} classes. Last layer is initialized randomly'.format(
                cfg_settings['num_classes'], kwargs_cls))
            # if there is last_linear in state_dict, it's going to be overwritten
            state_dict['fc.weight'] = model.state_dict()['last_linear.weight']
            state_dict['fc.bias'] = model.state_dict()['last_linear.bias']
        model.load_state_dict(state_dict)

    setattr(model, 'pretrained_settings', cfg_settings)
    return model


def densenet121(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    return _densenet('densenet121', **kwargs)


def densenet161(**kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    return _densenet('densenet161', **kwargs)


def densenet169(**kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    return _densenet('densenet169', **kwargs)


def densenet201(**kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    return _densenet('densenet201', **kwargs)
