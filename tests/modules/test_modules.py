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

# need to test and resnet and vgg because in resnet there are no Convs with bias
# and in VGG there are no Convs without bias
@pytest.mark.parametrize("norm_layer", ["abn", "agn"])
@pytest.mark.parametrize("arch", ["resnet18", "vgg11_bn"])
def test_weight_standardization(norm_layer, arch):
    m = pt.models.__dict__[arch](norm_layer=norm_layer)
    ws_m = modules.weight_standartization.conv_to_ws_conv(m)
    out = ws_m(torch.ones(2, 3, 224, 224))
