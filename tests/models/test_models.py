import torch
import pytest
import numpy as np
import pytorch_tools as pt
import pytorch_tools.models as models

ALL_MODEL_NAMES = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("_") and callable(models.__dict__[name])
)

DENSENET_NAMES = [name for name in ALL_MODEL_NAMES if "dense" in name]

EFFNET_NAMES = [name for name in ALL_MODEL_NAMES if "efficient" in name]

VGG_NAMES = [name for name in ALL_MODEL_NAMES if "vgg" in name]

RESNET_NAMES = [name for name in ALL_MODEL_NAMES if ("resne" in name) and not ("tresnet" in name)]

TRESNET_NAMES = [name for name in ALL_MODEL_NAMES if "tresne" in name]

HRNET_NAMES = [name for name in ALL_MODEL_NAMES if "hrnet" in name]

# test only part of the models
TEST_MODEL_NAMES = (
    DENSENET_NAMES[:1]
    + EFFNET_NAMES[:1]
    + VGG_NAMES[:1]
    + RESNET_NAMES[:1]
    + TRESNET_NAMES[:1]
    + HRNET_NAMES[:1]
)
# TEST_MODEL_NAMES = HRNET_NAMES[:1]
INP = torch.ones(2, 3, 128, 128)


def _test_forward(model):
    with torch.no_grad():
        return model(INP)


@pytest.mark.parametrize("arch", TEST_MODEL_NAMES)
@pytest.mark.parametrize("pretrained", [None, "imagenet"])
def test_init(arch, pretrained):
    m = models.__dict__[arch](pretrained=pretrained)
    _test_forward(m)


@pytest.mark.parametrize("arch", TEST_MODEL_NAMES)
def test_imagenet_custom_cls(arch):
    m = models.__dict__[arch](pretrained="imagenet", num_classes=10)
    _test_forward(m)


@pytest.mark.parametrize("arch", TEST_MODEL_NAMES)
def test_custom_in_channels(arch):
    m = models.__dict__[arch](in_channels=5)
    with torch.no_grad():
        m(torch.ones(2, 5, 128, 128))


@pytest.mark.parametrize("arch", EFFNET_NAMES[:2] + RESNET_NAMES[:2])
def test_pretrained_custom_in_channels(arch):
    m = models.__dict__[arch](in_channels=5, pretrained="imagenet")
    with torch.no_grad():
        m(torch.ones(2, 5, 128, 128))


@pytest.mark.parametrize("arch", TEST_MODEL_NAMES)
def test_inplace_abn(arch):
    """check than passing `inplaceabn` really changes all norm activations"""
    m = models.__dict__[arch](norm_layer="inplaceabn", norm_act="leaky_relu")
    _test_forward(m)

    def _check_bn(module):
        assert not isinstance(module, pt.modules.ABN)
        for child in module.children():
            _check_bn(child)

    _check_bn(m)


@pytest.mark.parametrize("arch", EFFNET_NAMES[:2] + RESNET_NAMES[:2])
@pytest.mark.parametrize("output_stride", [8, 16])
def test_dilation(arch, output_stride):
    m = models.__dict__[arch](output_stride=output_stride)
    with torch.no_grad():
        res = m.features(INP)
    W, H = INP.shape[-2:]
    assert res.shape[-2:] == (W // output_stride, H // output_stride)


@pytest.mark.parametrize("arch", EFFNET_NAMES[:2] + RESNET_NAMES[:2])
def test_drop_connect(arch):
    m = models.__dict__[arch](drop_connect_rate=0.2)
    _test_forward(m)


NUM_PARAMS = {
    "tresnetm": 31389032,
    "tresnetl": 55989256,
    "tresnetxl": 78436244,
    "efficientnet_b0": 5288548,
    "efficientnet_b1": 7794184,
    "efficientnet_b2": 9109994,
    "efficientnet_b3": 12233232,
}


@pytest.mark.parametrize("name_num_params", zip(NUM_PARAMS.items()))
def test_num_parameters(name_num_params):
    name, num_params = name_num_params[0]
    m = models.__dict__[name]()
    assert pt.utils.misc.count_parameters(m)[0] == num_params


@pytest.mark.parametrize("stem_type", ["", "deep", "space2depth"])
def test_resnet_stem_type(stem_type):
    m = models.resnet50(stem_type=stem_type)
    _test_forward(m)
