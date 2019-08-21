import pytest
import pytorch_tools.models as models
import torchvision.models as tv_models
from pytorch_tools.models import get_preprocessing_fn
import numpy as np
import torch


resnet_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and 'resne' in name and callable(models.__dict__[name]))

np_imgs = np.load('tests/models/test_dogs_42.npy') # HxWxC

@pytest.mark.parametrize('arch', resnet_names)
def test_resnet_init(arch):
    m = models.__dict__[arch](pretrained=None)