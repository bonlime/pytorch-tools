import pytest
import pytorch_tools.models as models

@pytest.mark.parametrize('arch', ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn'])
@pytest.mark.parametrize('pretrained', [None,'imagenet'])
def test_vgg_init(arch, pretrained):
    m = models.__dict__[arch](pretrained=pretrained)