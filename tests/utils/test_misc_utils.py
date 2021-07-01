import torch
import torch.nn as nn
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

def test_init_fn_long_seq():
    """check that variance is preserved over long sequences"""
    seq = nn.Sequential(*[nn.Conv2d(64, 64, 3, groups=16, padding_mode='reflect', padding=1) for _ in range(5)])
    seq = seq.eval().requires_grad_(False)
    inp = torch.randn(64, 64, 64, 64)
    assert seq(inp).std() < 0.2 # always holds for default init
    pt.utils.misc.initialize(seq, gamma=1)
    assert seq(inp).std() > 0.9  # usually >0.5 but setting lower to pass test in 100%

    
def test_init_fn_groups():
    """check that init works for DW convs"""
    seq = nn.Conv2d(64, 64, 3, groups=4).requires_grad_(False)
    inp = torch.randn(64, 64, 64, 64)
    assert seq(inp).std() < 0.6 # always holds for default init
    pt.utils.misc.initialize(seq, gamma=1)
    assert seq(inp).std() > 0.9  # usually >0.5 but setting lower to pass test in 100%

def test_init_fn_diff_out():
    """check that init works in case of very different number of channels"""
    seq = nn.Conv2d(16, 256, 3).requires_grad_(False)
    inp = torch.randn(64, 16, 64, 64)
    assert seq(inp).std() < 0.6 # always holds for default init
    pt.utils.misc.initialize(seq, gamma=1)
    assert seq(inp).std() > 0.9  # usually >0.5 but setting lower to pass test in 100%

def test_init_iterator():
    """check that init function works on iterators"""
    m = pt.models.resnet18()
    pt.utils.misc.initialize(m.modules())