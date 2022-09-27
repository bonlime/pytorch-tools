import torch
import pytest
import pytorch_tools as pt
import pytorch_tools.modules as modules

activations_name = ["Mish", "Mish_naive"]
INP = torch.ones(2, 10, 16, 16)


@pytest.mark.parametrize("activation", activations_name)
def test_activations_init(activation):
    act = modules.activation_from_name(activation)
    res = act(INP)
    assert res.mean()


def test_frozen_abn():
    l = modules.bn_from_name("frozen_abn")(10)
    assert list(l.parameters()) == []
    l = modules.ABN(10, frozen=True)
    assert list(l.parameters()) == []
    # check that passing tensor through frozen ABN won't update stats
    running_mean_original = l.running_mean.clone()
    running_var_original = l.running_var.clone()
    l(INP)
    assert torch.allclose(running_mean_original, l.running_mean)
    assert torch.allclose(running_var_original, l.running_var)


def test_estimated_abn():
    """Checks that init works and output is the same in eval mode"""
    est_bn = modules.bn_from_name("estimated_abn")(10).eval()
    bn = modules.bn_from_name("estimated_abn")(10).eval()
    est_bn.load_state_dict(bn.state_dict())
    assert torch.allclose(est_bn(INP), bn(INP))


def test_abn_repr():
    """Checks that activation param is present in repr"""
    l = modules.bn_from_name("frozen_abn")(10)
    expected = "ABN(10, eps=1e-05, momentum=0.1, affine=True, activation=ACT.LEAKY_RELU[0.01], frozen=True)"
    assert repr(l) == expected

    l2 = modules.bn_from_name("estimated_abn")(10, activation="relu")
    expected2 = "ABN(10, eps=1e-05, momentum=0.1, affine=True, activation=ACT.RELU, estimated_stats=True)"
    assert repr(l2) == expected2


def test_agn_repr():
    """Checks that repr for AGN includes number of groups"""
    l = modules.bn_from_name("agn")(10, num_groups=2, activation="leaky_relu")
    expected = "AGN(10, num_groups=2, eps=1e-05, affine=True, activation=ACT.LEAKY_RELU[0.01])"
    assert repr(l) == expected


def test_abcn():
    """Check that abcn init and forward works"""
    l = modules.bn_from_name("abcn")(10, num_groups=2)


def test_no_norm():
    """check that can init without params"""
    modules.bn_from_name("none")(10)
    modules.bn_from_name("none")()


def test_drop_connect_repr():
    """Check that keep_prob is shown correctly in DropConnect repr"""
    l = modules.residual.DropConnect(0.123)
    assert repr(l) == "DropConnect(keep_prob=0.12)"


# need to test and resnet and vgg because in resnet there are no Convs with bias
# and in VGG there are no Convs without bias
@pytest.mark.parametrize("norm_layer", ["abn", "agn"])
@pytest.mark.parametrize("arch", ["resnet18", "vgg11_bn"])
def test_weight_standardization(norm_layer, arch):
    m = pt.models.__dict__[arch](norm_layer=norm_layer)
    ws_m = modules.weight_standartization.conv_to_ws_conv(m)
    out = ws_m(torch.ones(2, 3, 224, 224))


def test_weight_standardization_depthwise():
    """check that depthwise convs are not converted"""
    m = pt.models.efficientnet_b0()
    ws_m = modules.weight_standartization.conv_to_ws_conv(m)
    assert type(ws_m.blocks[1][0].conv_dw) == torch.nn.Conv2d


def test_fpn_block():
    """Check that it works for different number of input feature maps"""
    inp_channels = [15, 16, 17, 18]
    inp = [torch.rand(1, in_ch, 2 * 2**idx, 2 * 2**idx) for idx, in_ch in enumerate(inp_channels)]
    # test for 4 features maps
    fpn4 = modules.fpn.FPN(inp_channels, pyramid_channels=55)
    res4 = fpn4(inp)
    for idx, r in enumerate(res4):
        assert r.shape == torch.Size([1, 55, 2 * 2**idx, 2 * 2**idx])

    # test that is also works for 3 feature maps
    fpn3 = modules.fpn.FPN(inp_channels[:3], pyramid_channels=55)
    res3 = fpn3(inp[:3])
    for idx, r in enumerate(res3):
        assert r.shape == torch.Size([1, 55, 2 * 2**idx, 2 * 2**idx])


def test_fpn_num_layers():
    # FPN should only support one layer
    with pytest.raises(AssertionError):
        modules.fpn.FPN([1, 2, 3, 4], num_layers=2)


def test_bifpn_block():
    """Check that it works for different number of input feature maps"""
    inp_channels = [15, 16, 17, 18, 19]
    inp = [torch.rand(1, in_ch, 2 * 2**idx, 2 * 2**idx) for idx, in_ch in enumerate(inp_channels)]
    # test for 5 features maps
    bifpn5 = modules.bifpn.BiFPN(inp_channels, pyramid_channels=55)
    res5 = bifpn5(inp)
    for idx, r in enumerate(res5):
        assert r.shape == torch.Size([1, 55, 2 * 2**idx, 2 * 2**idx])

    # test that is also works for 3 feature maps
    bifpn3 = modules.bifpn.BiFPN(inp_channels[:3], pyramid_channels=55)
    res3 = bifpn3(inp[:3])
    for idx, r in enumerate(res3):
        assert r.shape == torch.Size([1, 55, 2 * 2**idx, 2 * 2**idx])


def test_first_bifpn_layer():
    inp_channels = [15, 16, 17, 18, 19]
    inp = [torch.rand(1, in_ch, 2 * 2**idx, 2 * 2**idx) for idx, in_ch in enumerate(inp_channels)]
    layer = modules.bifpn.FirstBiFPNLayer(inp_channels, 55)
    res = layer(inp)
    for idx, r in enumerate(res):
        assert r.shape == torch.Size([1, 55, 2 * 2**idx, 2 * 2**idx])


def test_bifpn_num_layers():
    """Test different number of layers"""
    modules.bifpn.BiFPN([55, 55, 32, 30, 28], pyramid_channels=55, num_layers=2)
    modules.bifpn.BiFPN([1, 2, 3, 4], pyramid_channels=55, num_layers=2)


def test_space2depth():
    """Test that space2depth works as expected."""
    inp = torch.arange(16).view(1, 1, 4, 4)
    s2d_4 = modules.SpaceToDepth(block_size=4)
    out = s2d_4(inp)
    expected = torch.arange(16).view(1, -1, 1, 1)
    assert torch.allclose(out, expected)
    s2d_2 = modules.SpaceToDepth(block_size=2)
    out2 = s2d_2(inp)
    expected2 = torch.tensor([[[[0, 2], [8, 10]], [[1, 3], [9, 11]], [[4, 6], [12, 14]], [[5, 7], [13, 15]]]])
    assert torch.allclose(out2, expected2)


def test_drop_connect():
    """ "Tests that dropconnect works correctly for any number of dimensions"""
    drop_connect = modules.residual.DropConnect(0.5)
    inp1d = torch.rand(2, 10, 16)
    inp3d = torch.rand(2, 10, 16, 16, 16)
    assert drop_connect(INP).shape == INP.shape
    assert drop_connect(inp1d).shape == inp1d.shape
    assert drop_connect(inp3d).shape == inp3d.shape


def test_fused_vgg():
    """Test that Fused and not Fused version of VGG block are equal"""
    orig = modules.residual.RepVGGBlock(4, 6, act=torch.nn.SELU, alpha=0.1, n_heads=3)
    fused = modules.residual.FusedRepVGGBlock(4, 6, act=torch.nn.SELU)
    fused.load_state_dict(orig.state_dict())
    inp = torch.randn(1, 4, 3, 3)
    orig_out = orig(inp)
    fused_out = fused(inp)
    assert torch.allclose(orig_out, fused_out, atol=1e-6)


@pytest.mark.skip  # currently not working for some reason
def test_xca_module():
    """make sure that version that works on images and version that works on token are identical"""
    BS = 2
    DIM = 128
    SIZE = 7
    inp = torch.rand(BS, SIZE**2, DIM)
    inp_img = inp.transpose(1, 2).reshape(BS, DIM, SIZE, SIZE)
    xca_timm = modules.residual.XCA_Token(DIM, num_heads=4)
    xca_my = modules.residual.XCA(DIM, num_heads=4, qkv_bias=False)

    # load weights from Linear layer to conv1x1 layer
    xca_my.load_state_dict(xca_timm.state_dict())

    # make sure results are identical
    out_timm = xca_timm(inp)
    out_my = xca_my(inp_img)
    out_timm_reshaped = out_timm.transpose(1, 2).reshape(BS, DIM, SIZE, SIZE)
    assert torch.allclose(out_timm_reshaped, out_my, atol=1e-5)


def test_sesr_convs():
    expansion = 5
    sesr_e = pt.modules.conv.SESRBlockExpanded(INP.size(1), expansion)
    sesr_c = pt.modules.conv.SESRBlockCollapsed(INP.size(1), expansion)
    sesr_c.load_state_dict(sesr_e.state_dict())
    assert torch.allclose(sesr_e(INP), sesr_c(INP), atol=1e-6)
