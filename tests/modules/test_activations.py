import torch
import pytest
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