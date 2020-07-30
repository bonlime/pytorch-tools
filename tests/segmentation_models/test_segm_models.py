import torch
import pytest
import numpy as np
import pytorch_tools as pt
import pytorch_tools.segmentation_models as pt_sm


INP = torch.ones(2, 3, 64, 64)
ENCODERS = ["resnet34", "se_resnet50", "efficientnet_b1", "densenet121"]
SEGM_ARCHS = [pt_sm.Unet, pt_sm.Linknet, pt_sm.DeepLabV3, pt_sm.SegmentationFPN, pt_sm.SegmentationBiFPN]

# this lines are usefull for quick tests
# ENCODERS = ["se_resnet50", "efficientnet_b1"]
# SEGM_ARCHS = [pt_sm.SegmentationBiFPN]


def _test_forward(model):
    with torch.no_grad():
        return model(INP)


@pytest.mark.parametrize("encoder_name", ENCODERS)
@pytest.mark.parametrize("model_class", SEGM_ARCHS)
def test_forward(encoder_name, model_class):
    m = model_class(encoder_name=encoder_name)
    _test_forward(m)


@pytest.mark.parametrize("encoder_name", ENCODERS)
@pytest.mark.parametrize("model_class", SEGM_ARCHS)
def test_inplace_abn(encoder_name, model_class):
    """check than passing `inplaceabn` really changes all norm activations"""
    kwargs = dict(
        encoder_norm_layer="inplaceabn",
        encoder_norm_act="leaky_relu",
        decoder_norm_layer="inplaceabn",
        decoder_norm_act="leaky_relu",
    )

    m = model_class(encoder_name=encoder_name, **kwargs)
    _test_forward(m)

    def check_bn(module):
        assert not isinstance(module, pt.modules.ABN)
        for child in module.children():
            check_bn(child)

    check_bn(m)


@pytest.mark.parametrize("encoder_name", ENCODERS)
@pytest.mark.parametrize("model_class", SEGM_ARCHS)
def test_num_classes(encoder_name, model_class):
    m = model_class(encoder_name=encoder_name, num_classes=5)
    out = _test_forward(m)
    assert out.size(1) == 5


@pytest.mark.parametrize("encoder_name", ENCODERS)
@pytest.mark.parametrize("model_class", SEGM_ARCHS)
def test_drop_rate(encoder_name, model_class):
    m = model_class(encoder_name=encoder_name, drop_rate=0.2)
    _test_forward(m)


@pytest.mark.parametrize("encoder_name", ENCODERS)
@pytest.mark.parametrize("model_class", [pt_sm.DeepLabV3])  # pt_sm.Unet, pt_sm.Linknet
@pytest.mark.parametrize("output_stride", [32, 16, 8])
def test_dilation(encoder_name, model_class, output_stride):
    if output_stride == 8 and model_class != pt_sm.DeepLabV3:
        return None  # OS=8 only supported for Deeplab
    m = model_class(encoder_name=encoder_name, output_stride=output_stride)
    _test_forward(m)


@pytest.mark.parametrize("model_class", [pt_sm.DeepLabV3, pt_sm.SegmentationFPN, pt_sm.SegmentationBiFPN])
def test_deeplab_last_upsample(model_class):
    m = model_class(last_upsample=True)
    out = _test_forward(m)
    assert out.shape[-2:] == INP.shape[-2:]

    m = model_class(last_upsample=False)
    out = _test_forward(m)
    W, H = INP.shape[-2:]
    # should be 4 times smaller
    assert tuple(out.shape[-2:]) == (W // 4, H // 4)


@pytest.mark.parametrize("merge_policy", ["add", "cat"])
def test_merge_policy(merge_policy):
    m = pt_sm.SegmentationFPN(merge_policy=merge_policy)
    _test_forward(m)


@pytest.mark.parametrize("attn_type", ["se", "scse"])
@pytest.mark.parametrize("model_class", [pt_sm.Unet, pt_sm.Linknet])
def test_attention(attn_type, model_class):
    """check that passing different attention works"""
    m = model_class(encoder_name="resnet34", decoder_attention_type=attn_type)
    _test_forward(m)
