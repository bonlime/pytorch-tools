import pytest
import pytorch_tools.models as models


densenet_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and "dense" in name and callable(models.__dict__[name])
)


@pytest.mark.parametrize("arch", densenet_names[:2])
@pytest.mark.parametrize("pretrained", [None, "imagenet"])
def test_densenet_init(arch, pretrained):
    m = models.__dict__[arch](pretrained=pretrained)


@pytest.mark.parametrize("arch", densenet_names[:2])
def test_densenet_imagenet_custom_cls(arch):
    m = models.__dict__[arch](pretrained="imagenet", num_classes=10)


@pytest.mark.parametrize("arch", densenet_names[:2])  # test only part of the models
def test_densenet_custom_in_channels(arch):
    m = models.__dict__[arch](in_channels=5)
