import pytest
import pytorch_tools as pt


def test_filter_bn():
    model = pt.models.resnet18()
    param_groups = pt.utils.misc.filter_bn_from_wd(model)
    assert param_groups[0]["weight_decay"] == 0
    assert len(param_groups[0]["params"]) == 41
    assert len(param_groups[1]["params"]) == 21

    model = pt.models.efficientnet_b0()
    param_groups = pt.utils.misc.filter_bn_from_wd(model)
    assert param_groups[0]["weight_decay"] == 0
    assert len(param_groups[0]["params"]) == 131
    assert len(param_groups[1]["params"]) == 82
