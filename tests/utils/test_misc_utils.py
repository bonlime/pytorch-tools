import torch
import torch.nn as nn
import pytorch_tools as pt


def test_filter_bn():
    model = pt.models.resnet18()
    param_groups = pt.utils.misc.filter_from_weight_decay(model)
    assert param_groups[1]["weight_decay"] == 0
    assert len(param_groups[0]["params"]) == 21
    assert len(param_groups[1]["params"]) == 41

    model = pt.models.efficientnet_b0()
    param_groups = pt.utils.misc.filter_from_weight_decay(model)
    assert param_groups[1]["weight_decay"] == 0
    assert len(param_groups[0]["params"]) == 82
    assert len(param_groups[1]["params"]) == 131

    # check that skip_list works as expected
    model = pt.models.resnet18(attn_type="eca")
    param_groups = pt.utils.misc.filter_from_weight_decay(model, skip_list=("se_module",))
    assert len(param_groups[1]["params"]) == 49


def test_patch_bn_mom():
    """check that patching momentum works for all layers in the model"""
    model = pt.models.resnet18()
    new_mom = 0.02
    pt.utils.misc.patch_bn_mom(model, new_mom)
    assert model.bn1.momentum == new_mom
    assert model.layer4[1].bn2.momentum == model.bn1.momentum


def test_init_fn_long_seq():
    """check that variance is preserved over long sequences"""
    seq = nn.Sequential(*[nn.Conv2d(64, 64, 3, groups=16, padding_mode="reflect", padding=1) for _ in range(5)])
    seq = seq.eval().requires_grad_(False)
    inp = torch.randn(64, 64, 64, 64)
    assert seq(inp).std() < 0.2  # always holds for default init
    pt.utils.misc.initialize(seq, gamma=1)
    assert seq(inp).std() > 0.9  # usually >0.5 but setting lower to pass test in 100%


def test_init_fn_groups():
    """check that init works for DW convs"""
    seq = nn.Conv2d(64, 64, 3, groups=4).requires_grad_(False)
    inp = torch.randn(64, 64, 64, 64)
    assert seq(inp).std() < 0.6  # always holds for default init
    pt.utils.misc.initialize(seq, gamma=1)
    assert seq(inp).std() > 0.9  # usually >0.5 but setting lower to pass test in 100%


def test_init_fn_diff_out():
    """check that init works in case of very different number of channels"""
    seq = nn.Conv2d(16, 256, 3).requires_grad_(False)
    inp = torch.randn(64, 16, 64, 64)
    assert seq(inp).std() < 0.6  # always holds for default init
    pt.utils.misc.initialize(seq, gamma=1)
    assert seq(inp).std() > 0.9  # usually >0.5 but setting lower to pass test in 100%


def test_init_iterator():
    """check that init function works on iterators"""
    m = pt.models.resnet18()
    pt.utils.misc.initialize(m.modules())


def test_normalize_conv_weight():
    """test that normalizing conv weights helps to remove large mean shifts"""
    seq = nn.Conv2d(16, 16, 3).requires_grad_(False)
    inp = torch.randn(2, 16, 16, 16) + 10  # simulate large mean shift
    out = seq(inp)
    # usual conv doesn't help with mean shift
    assert out.mean(dim=(0, 2, 3)).pow(2).mean().sqrt().item() > 3
    seq.weight.data.copy_(pt.utils.misc.normalize_conv_weight(seq.weight))
    out = seq(inp)
    # correctly normalized conv should remove mean shift
    assert out.mean(dim=(0, 2, 3)).pow(2).mean().sqrt().item() < 0.5


def test_zero_mean_conv_weight():
    """test that zero-mean of conv weights helps to remove large mean shifts"""
    seq = nn.Conv2d(32, 32, 3).requires_grad_(False)
    inp = torch.randn(2, 32, 16, 16) + 10  # simulate large mean shift
    out = seq(inp)
    # usual conv doesn't help with mean shift
    assert out.mean(dim=(0, 2, 3)).pow(2).mean().sqrt().item() > 3
    seq.weight.data = pt.utils.misc.zero_mean_conv_weight(seq.weight)
    out = seq(inp)
    # correctly normalized conv should remove mean shift
    assert out.mean(dim=(0, 2, 3)).pow(2).mean().sqrt().item() < 0.5


def test_update_dict():
    """Tests to make sure updating dict works as expected"""
    # simple update
    d_to = {"a": 10, "b": 20}
    d_from = {"a": 12, "c": 30}
    d_expected = {"a": 12, "b": 20, "c": 30}
    assert pt.utils.misc.update_dict(d_to, d_from) == d_expected

    # recursive update. dict.update would fail in this case
    d_to = {"foo": {"a": 10, "b": 20}}
    d_from = {"foo": {"a": 12, "c": 30}}
    d_expected = {"foo": {"a": 12, "b": 20, "c": 30}}
    assert pt.utils.misc.update_dict(d_to, d_from) == d_expected

    # when key is not present in `to`
    d_to = {"bar": 1}
    d_from = {"foo": {"a": 12, "c": 30}}
    d_expected = {"bar": 1, "foo": {"a": 12, "c": 30}}
    assert pt.utils.misc.update_dict(d_to, d_from) == d_expected
