import torch
import pytest
import pytorch_tools as pt
import pytorch_tools.models as models


efficientnet_names = sorted(
    name
    for name in models.__dict__
    if name.islower()
    and not name.startswith("__")
    and "efficient" in name
    and callable(models.__dict__[name])
)

INP = torch.ones(2, 3, 128, 128)


def _test_forward(model):
    with torch.no_grad():
        return model(INP)


@pytest.mark.parametrize("arch", efficientnet_names)
@pytest.mark.parametrize("pretrained", [None, "imagenet"])
def test_effnet_init(arch, pretrained):
    m = models.__dict__[arch](pretrained=pretrained)
    _test_forward(m)


@pytest.mark.parametrize("arch", efficientnet_names[:2])
def test_effnet_imagenet_custom_cls(arch):
    m = models.__dict__[arch](pretrained="imagenet", num_classes=10)
    _test_forward(m)


@pytest.mark.parametrize("arch", efficientnet_names[:2])  # test only part of the models
def test_effnet_custom_in_channels(arch):
    m = models.__dict__[arch](in_channels=5)
    with torch.no_grad():
        m(torch.ones(2, 5, 128, 128))


@pytest.mark.parametrize("arch", efficientnet_names[:2])  # test only part of the models
def test_effnet_inplace_abn(arch):
    """check than passing `inplaceabn` really changes all norm activations"""
    m = models.__dict__[arch](norm_layer="inplaceabn", norm_act="leaky_relu")
    _test_forward(m)

    def _check_bn(module):
        assert not isinstance(module, pt.modules.ABN)
        for child in module.children():
            _check_bn(child)

    _check_bn(m)
