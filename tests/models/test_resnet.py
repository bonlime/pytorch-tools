import pytest
import pytorch_tools.models as models
import torchvision.models as tv_models
from pytorch_tools.models import get_preprocessing_fn
import numpy as np
import torch


resnet_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and "resne" in name and callable(models.__dict__[name])
)


@pytest.mark.parametrize("arch", resnet_names)
def test_resnet_init(arch):
    m = models.__dict__[arch](pretrained=None)


@pytest.mark.parametrize("arch", resnet_names[:5])  # test only part of the models
def test_resnet_imagenet(arch):
    m = models.__dict__[arch](pretrained="imagenet")


@pytest.mark.parametrize("arch", resnet_names[:5])  # test only part of the models
def test_resnet_imagenet_custom_cls(arch):
    m = models.__dict__[arch](pretrained="imagenet", num_classes=10)


@pytest.mark.parametrize("arch", resnet_names[:5])  # test only part of the models
def test_resnet_custom_in_channels(arch):
    m = models.__dict__[arch](in_channels=5)
