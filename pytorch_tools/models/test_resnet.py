import pytest
import pytorch_tools.models as models


resnet_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and 'resne' in name and callable(models.__dict__[name]))

@pytest.mark.parametrize('arch', resnet_names)
@pytest.mark.parametrize('pretrained', [None,'imagenet'])
def test_resnet_init(arch, pretrained):
    m = models.__dict__[arch](pretrained=pretrained)