import pytorch_tools as pt
from pytorch_tools.metrics import Accuracy
from pytorch_tools.metrics import BalancedAccuracy
from pytorch_tools.metrics import DiceScore, JaccardScore, PSNR

# from ..utils.misc import to_numpy
import pytest
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np

METRIC_NAMES = sorted(name for name in pt.metrics.__dict__ if not name.islower())

## Accuracy tests
def test_accuracy():
    output = torch.rand((16, 4))
    output_np = output.numpy()
    target = torch.randint(0, 4, (16,))
    target_np = target.numpy()
    expected = 100 * accuracy_score(target_np, np.argmax(output_np, 1))
    result = Accuracy()(output, target).item()
    assert np.allclose(expected, result)
    # check that it also works in case of extra dim for target
    result2 = Accuracy()(output, target.view(16, 1)).item()
    assert np.allclose(expected, result2)


def test_accuracy_topk():
    # fmt: off
    output = torch.tensor([
        [ 10,   5, -10,],
        [ 10, -10,   5,],
        [  5,  10, -10,],
        [-10,   5,  10,],
        [-10,   5,  10,],
    ])
    # fmt: on
    target = torch.tensor([1, 1, 1, 1, 1])
    result = Accuracy(2)(output, target).item()
    assert np.allclose(result, 4 / 5 * 100)
    target2 = torch.tensor([2, 2, 2, 2, 2])
    result2 = Accuracy(2)(output, target2).item()
    assert np.allclose(result2, 3 / 5 * 100)


def test_binary_accuracy():
    output = torch.rand((16, 1))
    output_np = output.numpy()
    target = torch.randint(0, 2, (16,))
    target_np = target.numpy()
    expected = 100 * accuracy_score(target_np, output_np > 0)
    result = Accuracy()(output, target).item()
    assert np.allclose(expected, result)


def test_binary_accuracy_image():
    output = torch.rand((16, 1, 20, 20)) - 0.5
    output_np = output.numpy()
    target = torch.randint(0, 2, (16, 1, 20, 20))
    target_np = target.numpy()
    expected = 100 * accuracy_score(target_np.flatten(), output_np.flatten() > 0)
    result = Accuracy()(output, target).item()
    assert np.allclose(expected, result)


def test_accuracy_image():
    output = torch.rand((16, 4, 20, 20))
    target = torch.randint(0, 2, (16, 1, 20, 20))
    expected = 100 * accuracy_score(target.flatten().numpy(), output.argmax(1).flatten())
    result = Accuracy()(output, target).item()
    assert np.allclose(expected, result)


def test_balanced_accuracy():
    output = torch.rand((16, 4))
    output_np = output.numpy()
    target = torch.randint(0, 4, (16,))
    target_np = target.numpy()
    expected = 100 * balanced_accuracy_score(target_np, np.argmax(output_np, 1))
    result = BalancedAccuracy()(output, target).flatten().numpy()
    assert np.allclose(expected, result)


IM_SIZE = 10
BS = 8
# only check that score == 1 - loss. see losses/test_losses for more tests
def test_dice_score():
    inp = torch.randn(BS, 1, IM_SIZE, IM_SIZE).float()
    target = torch.randint(0, 2, (BS, 1, IM_SIZE, IM_SIZE)).float()

    dice_score = DiceScore(mode="binary", from_logits=False)(inp, target)
    dice_loss = pt.losses.DiceLoss(mode="binary", from_logits=False)(inp, target)
    assert dice_score == 1 - dice_loss


def test_jaccard_score():
    inp = torch.randn(BS, 1, IM_SIZE, IM_SIZE).float()
    target = torch.randint(0, 2, (BS, 1, IM_SIZE, IM_SIZE)).float()

    jaccard_score = JaccardScore(mode="binary", from_logits=False)(inp, target)
    jaccard_loss = pt.losses.JaccardLoss(mode="binary", from_logits=False)(inp, target)
    assert jaccard_score == 1 - jaccard_loss


def test_psnr():
    img1 = torch.randint(low=0, high=256, size=(BS, 3, IM_SIZE, IM_SIZE), dtype=torch.float32)

    img2 = torch.randint(low=0, high=256, size=(BS, 3, IM_SIZE, IM_SIZE), dtype=torch.float32)

    psnr = PSNR()(img1, img2)
    assert psnr > 0.0


@pytest.mark.parametrize("metric", METRIC_NAMES)
def test_has_name(metric):
    m = pt.metrics.__dict__[metric]()
    assert hasattr(m, "name")
