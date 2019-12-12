import pytest
import pytorch_tools.modules as modules

activations_name = ["SiLU", "SoftExponential", "Swish", "MemoryEfficientSwish"]


@pytest.mark.parametrize("activation", activations_name)
def test_activations_init(activation):
    a = modules.__dict__[activation]()
