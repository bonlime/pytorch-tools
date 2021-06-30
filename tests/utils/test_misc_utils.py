from pytorch_tools import models
import pytest
import pytorch_tools as pt


def test_filter_bn():
    model = pt.models.resnet18()
    param_groups = pt.utils.misc.filter_bn_from_wd(model)
    assert param_groups[1]["weight_decay"] == 0
    assert len(param_groups[0]["params"]) == 21
    assert len(param_groups[1]["params"]) == 41

    model = pt.models.efficientnet_b0()
    param_groups = pt.utils.misc.filter_bn_from_wd(model)
    assert param_groups[1]["weight_decay"] == 0
    assert len(param_groups[0]["params"]) == 82
    assert len(param_groups[1]["params"]) == 131


def test_patch_bn_mom():
    """check that patching momentum works for all layers in the model"""
    model = pt.models.resnet18()
    new_mom = 0.02
    pt.utils.misc.patch_bn_mom(model, new_mom)
    assert model.bn1.momentum == new_mom
    assert model.layer4[1].bn2.momentum == model.bn1.momentum

