import torch
import pytest
import numpy as np
import pytorch_tools.segmentation_models as pt_sm


INP = torch.ones(2,3,64,64)
ENCODERS = ["resnet34", "se_resnet50", "densenet121"]

def _test_forward(model):
    with torch.no_grad():
        model(INP)


@pytest.mark.parametrize("encoder_name", ENCODERS)
@pytest.mark.parametrize("model_class", [pt_sm.Unet, pt_sm.Linknet])
def test_forward(encoder_name, model_class):
    m = model_class(encoder_name=encoder_name)
    _test_forward(m)

@pytest.mark.parametrize("encoder_name", ENCODERS[:1])
@pytest.mark.parametrize("model_class", [pt_sm.Unet, pt_sm.Linknet])
def test_inplace_abn(encoder_name, model_class):
    m = model_class(encoder_name=encoder_name, norm_layer="inplaceabn", norm_act="leaky_relu")
    _test_forward(m)

# Fails for now
# @pytest.mark.parametrize("encoder_name", ENCODERS[:1])
# @pytest.mark.parametrize("model_class", [pt_sm.Unet, pt_sm.Linknet])
# def test_dilation(encoder_name, model_class):
#     m = model_class(encoder_name=encoder_name, dilated=True)
#     _test_forward(m)

