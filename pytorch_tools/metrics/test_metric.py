from .accuracy import Accuracy
from .accuracy import BalancedAccuracy
#from ..utils.misc import to_numpy
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np 

def test_accuracy():
    output = torch.rand((16,4))
    output_np = output.numpy()
    target = torch.randint(0,4,(16,))
    target_np = target.numpy()
    expected = 100 * accuracy_score(target_np, np.argmax(output_np, 1))
    result = Accuracy()(output, target).flatten().numpy()
    assert np.allclose(expected, result)

def test_balanced_accuracy():
    output = torch.rand((16,4))
    output_np = output.numpy()
    target = torch.randint(0,4,(16,))
    target_np = target.numpy()
    expected = 100 * balanced_accuracy_score(target_np, np.argmax(output_np, 1))
    result = BalancedAccuracy()(output, target).flatten().numpy()
    assert np.allclose(expected, result)