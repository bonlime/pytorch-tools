import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from inplace_abn import ABN
from pytorch_tools.modules import BlurPool

model_urls = {
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):
    """VGG model builder. Only BN version is supported.  
        From: https://arxiv.org/pdf/1409.1556.pdf

    Args:
        cfg ([type]): [description]
        num_classes (int, optional): [description]. Defaults to 1000.
        norm_layer (ABN, optional): Which version of ABN to use. Choices are:
            'ABN' - dropin replacement for BN+Relu.
            'InplaceABN' - efficient version. If used with `pretrain` Weights still 
                will be loaded but performance may be worse than with ABN. 
        encoder (bool, optional): Flag to return features with different resolution. 
            Defaults to False.
        antialias (bool, optional): Flag to turn on Rect-2 antialiasing 
            from https://arxiv.org/abs/1904.11486. Defaults to False.
    """

    def __init__(self, 
                 cfg, 
                 num_classes=1000, 
                 norm_layer=ABN,
                 encoder=False,
                 antialias=False):

        super(VGG, self).__init__()
        self.norm_layer = norm_layer
        self.norm_act = 'relu' if isinstance(norm_layer, ABN) else 'leaky_relu'
        self.encoder = encoder
        self.antialias = antialias
        self.features = self._make_layers(cfgs[cfg])
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
        self._initialize_weights()

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

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                if self.antialias:
                    layers += [nn.MaxPool2d(kernel_size=2, stride=1), BlurPool(in_channels)]
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
            if k.startswith('classifier') and self.encoder:
                state_dict.pop(k)
            if k.endswith('num_batches_tracked'):
                state_dict.pop(k)
        # there is a mismatch in feature layers names, so need this mapping
        self_feature_names = [i for i in self.state_dict().keys() if 'features' in i]
        load_feature_names = [i for i in state_dict.keys() if 'features' in i]
        features_map = {load_f:self_f for (load_f, self_f) in zip(load_feature_names,
                                                        self_feature_names)}
        for k in keys:
            if k.startswith('features'):
                state_dict[features_map[k]] = state_dict.pop(k)
        # handle blurpool
        for k in list(self.state_dict().keys()):
            if k.endswith('filt'):
                state_dict[k] = self.state_dict()[k]
        super().load_state_dict(state_dict, **kwargs)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def _vgg(arch, cfg, pretrained, **kwargs):
    model = VGG(cfg, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=True)
        model.load_state_dict(state_dict)
    return model


def vgg11_bn(pretrained=False, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg('vgg11_bn', 'A', pretrained, **kwargs)


def vgg13_bn(pretrained=False, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg('vgg13_bn', 'B', pretrained, **kwargs)


def vgg16_bn(pretrained=False, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg('vgg16_bn', 'D', pretrained, **kwargs)


def vgg19_bn(pretrained=False, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg('vgg19_bn', 'E', pretrained, **kwargs)
