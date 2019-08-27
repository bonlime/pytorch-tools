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
from pytorch_tools.modules import GlobalPool2d, BlurPool, Transition, DenseLayer
from pytorch_tools.modules.residual import conv1x1, conv3x3
from pytorch_tools.utils.misc import activation_from_name, bn_from_name
from pytorch_tools.utils.misc import add_docs_for
from pytorch_tools.utils.misc import DEFAULT_IMAGENET_SETTINGS
from collections import OrderedDict
from functools import wraps, partial
from copy import deepcopy
import logging


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
    bottle_neck: bool, default True
        Use bottleneck in DenseLayer with default expansion rate 4. (Width = 4 * growth_rate)
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
                 bottleneck=True,
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

        largs = dict(compression=compression, norm_layer=norm_layer,
                     norm_act=norm_act, antialias=antialias)

        num_features = num_init_features
        self.denseblock1, self.transition1 = self._make_layer(block_config[0],
                                                              growth_rate,
                                                              norm_act=norm_act,
                                                              norm_layer=norm_layer,
                                                              drop_rate=drop_rate,
                                                              compression=compression,
                                                              transition=True)
        num_features = int((num_features + block_config[0] * growth_rate) * compression)
        self.denseblock2, self.transition2 = self._make_layer(block_config[1],
                                                              growth_rate,
                                                              norm_act=norm_act,
                                                              norm_layer=norm_layer,
                                                              drop_rate=drop_rate,
                                                              compression=compression,
                                                              transition=True)
        num_features = int((num_features + block_config[0] * growth_rate) * compression)
        self.denseblock3, self.transition3 = self._make_layer(block_config[2],
                                                              growth_rate,
                                                              norm_act=norm_act,
                                                              norm_layer=norm_layer,
                                                              drop_rate=drop_rate,
                                                              compression=compression,
                                                              transition=True)
        num_features = int((num_features + block_config[0] * growth_rate) * compression)
        self.denseblock4 = self._make_layer(block_config[0],
                                            growth_rate,
                                            norm_act=norm_act,
                                            norm_layer=norm_layer,
                                            drop_rate=drop_rate,
                                            transition=False)

        num_features = int(num_features + block_config[0] * growth_rate)

        # Final normalization
        self.norm5 = norm_layer(num_features) 

        # Linear layer
        if not encoder:
            self.classifier = nn.Linear(num_features, num_classes)
        else:
            self.forward = self.encoder_features
        self.classifier = nn.Linear(num_features, num_classes)

        self._initialize_weights(init_bn0)

    def _make_layer(self, in_planes, blocks, growth_rate,
                    norm_layer,
                    norm_act,
                    drop_rate,
                    compression,
                    transition,
                    global_pool):
        """
        Returns DenseBlock
        """
        layers = nn.Sequential()
        for i in range(blocks):
            layer = DenseLayer(in_planes + i * growth_rate,
                               growth_rate=growth_rate,
                               norm_layer=norm_layer,
                               norm_act=norm_act,
                               drop_rate=drop_rate)
            layers.add_module("denselayer{}".format(i + 1), layer)
        in_planes = in_planes + blocks * growth_rate
        if transition:
            trans = Transition(in_planes=in_planes,
                               out_planes=int(in_planes * compression),
                               drop_rate=drop_rate,
                               norm_act=norm_act,
                               norm_layer=norm_layer,
                               global_pool=global_pool)
            return layers, trans
        else:
            return layers

    def features(self, x):
        x = self.relu0(self.norm0(self.conv0(x)))
        x = self.pool0(x)
        x = self.transition1(self.denseblock1(x))
        x = self.transition2(self.denseblock2(x))
        x = self.transition3(self.denseblock3(x))
        x = self.denseblock4(x)
        x = self.norm5(x)
        return x

    # def encoder_features(self, x):
    #     """
    #     Return 5 feature maps before maxpooling layers
    #     """
    #     x0 = self.relu0(self.norm0(self.conv0(x)))
    #     x1 = self.denseblock1(self.pool0(x0))
    #     x2 = self.denseblock2(self.transition1(x1))
    #     x3 = self.denseblock3(self.transition2(x1))
    #     x4 = self.norm5(self.denseblock4(self.transition3(x1)))
    #     return [x4, x3, x2, x1, x0]

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
            'params': {'growth_rate': 32, 'block_config': (6, 12, 24, 16), 'num_init_features': 64, 'norm_layer': 'bn'},
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
