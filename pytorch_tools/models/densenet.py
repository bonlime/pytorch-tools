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
from pytorch_tools.utils.misc import add_docs_for
from pytorch_tools.utils.misc import DEFAULT_IMAGENET_SETTINGS
from copy import deepcopy
import re
import logging

class _Transition(nn.Module):
    r"""
    Transition Block as described in [DenseNet](https://arxiv.org/abs/1608.06993)
    
    - Activation
    - 1x1 Convolution (with optional compression of the number of channels)
    - 2x2 Average Pooling
    """
    def __init__(self, in_planes, out_planes):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(in_planes)
        self.act = nn.ReLU(inplace=True)
        self.conv = conv1x1(in_planes, out_planes)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.norm(x)
        out = self.act(x)
        out = self.conv(out)
        out = self.pool(out)
        return out

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function

class _DenseLayer(nn.Module):
    expansion = 4

    def __init__(self, in_planes, growth_rate, drop_rate=0.0, memory_efficient=False):
        super(_DenseLayer, self).__init__()

        width = growth_rate * self.expansion
        self.norm1 = nn.BatchNorm2d(in_planes)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(in_planes, width)
        self.norm2 = nn.BatchNorm2d(width)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, growth_rate)
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *inputs):
        bn_function = _bn_function_factory(self.norm1, self.act1, self.conv1)
        if self.memory_efficient and any(x.requires_grad for x in inputs):
            out = cp.checkpoint(bn_function, *inputs)
        else:
            out = bn_function(*inputs)
        out = self.conv2(self.act2(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_planes, growth_rate, **kwargs):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                in_planes + i * growth_rate, 
                growth_rate=growth_rate, **kwargs)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        out = [x]
        for name, layer in self.named_children():
            new_out = layer(*out)
            out.append(new_out)
        return torch.cat(out, 1)


class DenseNet(nn.Module):
    r"""DenseNet model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    
    DenseNet variants:
      * normal - 7x7 stem, stem_width = 64 (96 for densenet161), same as torchvision DenseNet
      * bc - added compression and bottleneck layers

    Parameters
    ----------
    growth_rate :  int
        How many filters to add each layer (`k` in paper)
    block_config : (list of 4 ints) 
        How many layers in each pooling block
    stem_width : int, default 64
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
                 num_classes=1000,
                 stem_width=64,
                 drop_rate=0.0,
                 in_chans=3,
                 deep_stem=False,
                 encoder=False,
                 global_pool='avg',
                 memory_efficient=False):

        self.num_classes = num_classes
        self.stem_width = stem_width
        super(DenseNet, self).__init__()
        layers_dict = OrderedDict()
        if deep_stem:
            layers_dict['conv0'] = nn.Sequential(
                conv3x3(in_chans, stem_width // 2, 2),
                nn.BatchNorm2d(stem_width //2),
                nn.ReLU(inplace=True),
                conv3x3(stem_width // 2, stem_width // 2),
                nn.BatchNorm2d(stem_width //2),
                nn.ReLU(inplace=True),
                conv3x3(stem_width // 2, stem_width, 2)
            )
        else:
            layers_dict['conv0'] = nn.Conv2d(in_chans, stem_width, kernel_size=7, 
                                             stride=2, padding=3, bias=False)

        layers_dict['norm0'] = nn.BatchNorm2d(stem_width) 
        layers_dict['relu0'] = nn.ReLU(inplace=True)
        layers_dict['pool0'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        
        largs = {'growth_rate': growth_rate, 'drop_rate': drop_rate, 'memory_efficient': memory_efficient}
        in_planes = stem_width
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, in_planes, **largs)
            layers_dict['denseblock{}'.format(i+1)] = block
            in_planes += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(in_planes=in_planes, out_planes=in_planes // 2)
                layers_dict['transition{}'.format(i+1)] = trans
                in_planes //= 2

        # Final normalization
        layers_dict['norm5'] = nn.BatchNorm2d(in_planes)
        # define features
        self.features = nn.Sequential(layers_dict)

        # Linear layer
        if not encoder:
            self.global_pool = GlobalPool2d(global_pool)
            self.classifier = nn.Linear(in_planes, num_classes)
        else:
            assert len(block_config) == 4, 'Need 4 blocks to use as encoder'
            self.forward = self.encoder_features

    # def encoder_features(self, x):
    #     """
    #     Return 5 feature maps before maxpooling layers
    #     """
    #     x0 = self.norm0(self.conv0(x))
    #     x1 = self.denseblock1(self.pool0(x0))
    #     x2 = self.denseblock2(self.transition1(x1))
    #     x3 = self.denseblock3(self.transition2(x1))
    #     x4 = self.norm5(self.denseblock4(self.transition3(x1)))
    #     return [x4, x3, x2, x1, x0]

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

    def _initialize_weights(self):
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_state_dict(self, state_dict, **kwargs):
        pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict.pop(key)
        super().load_state_dict(state_dict, **kwargs)

CFGS = {
    'densenet121': {
        'default': {
            'params': {'growth_rate': 32, 'block_config': (6, 12, 24, 16), 'stem_width': 64, },
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
            'params': {'growth_rate': 48, 'block_config': (6, 12, 36, 24), 'stem_width': 96},
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
            'params': {'growth_rate': 48, 'block_config': (6, 12, 36, 24), 'stem_width': 96},
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
            #state_dict['fc.weight'] = model.state_dict()['last_linear.weight']
            #state_dict['fc.bias'] = model.state_dict()['last_linear.bias']
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
