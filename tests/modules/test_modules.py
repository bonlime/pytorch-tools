import torch
import pytest
import pytorch_tools as pt
import pytorch_tools.modules as modules

activations_name = ["Swish", "Swish_Naive", "Mish", "Mish_naive"]


@pytest.mark.parametrize("activation", activations_name)
def test_activations_init(activation):
    inp = torch.ones(10)
    act = modules.activation_from_name(activation)
    res = act(inp)
    assert res.mean()


def test_frozen_abn():
    l = modules.bn_from_name("frozen_abn")(10)
    assert list(l.parameters()) == []
    l = modules.ABN(10, frozen=True)
    assert list(l.parameters()) == []


def test_abn_repr():
    """Checks that activation param is present in repr"""
    l = modules.bn_from_name("frozen_abn")(10)
    expected = "ABN(10, eps=1e-05, momentum=0.1, affine=True, activation=ACT.LEAKY_RELU[0.01], frozen=True)"
    assert repr(l) == expected


# need to test and resnet and vgg because in resnet there are no Convs with bias
# and in VGG there are no Convs without bias
@pytest.mark.parametrize("norm_layer", ["abn", "agn"])
@pytest.mark.parametrize("arch", ["resnet18", "vgg11_bn"])
def test_weight_standardization(norm_layer, arch):
    m = pt.models.__dict__[arch](norm_layer=norm_layer)
    ws_m = modules.weight_standartization.conv_to_ws_conv(m)
    out = ws_m(torch.ones(2, 3, 224, 224))


def test_fpn_block():
    """Check that it works for different number of input feature maps"""
    inp_channels = [15, 16, 17, 18]
    inp = [torch.rand(1, in_ch, 2 * 2 ** idx, 2 * 2 ** idx) for idx, in_ch in enumerate(inp_channels)]
    # test for 4 features maps
    fpn4 = modules.fpn.FPN(inp_channels, pyramid_channels=55)
    res4 = fpn4(inp)
    for idx, r in enumerate(res4):
        assert r.shape == torch.Size([1, 55, 2 * 2 ** idx, 2 * 2 ** idx])

    # test that is also works for 3 feature maps
    fpn3 = modules.fpn.FPN(inp_channels[:3], pyramid_channels=55)
    res3 = fpn3(inp[:3])
    for idx, r in enumerate(res3):
        assert r.shape == torch.Size([1, 55, 2 * 2 ** idx, 2 * 2 ** idx])


def test_fpn_num_layers():
    # FPN should only support one layer
    with pytest.raises(AssertionError):
        modules.fpn.FPN([1, 2, 3, 4], num_layers=2)


def test_bifpn_block():
    """Check that it works for different number of input feature maps"""
    inp_channels = [15, 16, 17, 18, 19]
    inp = [torch.rand(1, in_ch, 2 * 2 ** idx, 2 * 2 ** idx) for idx, in_ch in enumerate(inp_channels)]
    # test for 5 features maps
    bifpn5 = modules.bifpn.BiFPN(inp_channels, pyramid_channels=55)
    res5 = bifpn5(inp)
    for idx, r in enumerate(res5):
        assert r.shape == torch.Size([1, 55, 2 * 2 ** idx, 2 * 2 ** idx])

    # test that is also works for 3 feature maps
    bifpn3 = modules.bifpn.BiFPN(inp_channels[:3], pyramid_channels=55)
    res3 = bifpn3(inp[:3])
    for idx, r in enumerate(res3):
        assert r.shape == torch.Size([1, 55, 2 * 2 ** idx, 2 * 2 ** idx])


def test_first_bifpn_layer():
    inp_channels = [15, 16, 17, 18, 19]
    inp = [torch.rand(1, in_ch, 2 * 2 ** idx, 2 * 2 ** idx) for idx, in_ch in enumerate(inp_channels)]
    layer = modules.bifpn.FirstBiFPNLayer(inp_channels, 55)
    res = layer(inp)
    for idx, r in enumerate(res):
        assert r.shape == torch.Size([1, 55, 2 * 2 ** idx, 2 * 2 ** idx])


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

