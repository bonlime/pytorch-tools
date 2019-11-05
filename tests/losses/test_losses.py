import pytest
import torch

import pytorch_tools.losses.functional as F
import pytorch_tools.losses as losses
"""
Some test were taken from repo by BloodAxe
https://github.com/BloodAxe/pytorch-toolbelt/
"""

LOSSES_NAMES = sorted(name for name in losses.__dict__ if not name.islower())

def test_sigmoid_focal_loss():
    input_good = torch.Tensor([10, -10, 10]).float()
    input_bad = torch.Tensor([-1, 2, 0]).float()
    target = torch.Tensor([1, 0, 1])

    loss_good = F.sigmoid_focal_loss(input_good, target)
    loss_bad = F.sigmoid_focal_loss(input_bad, target)
    assert loss_good < loss_bad


def test_reduced_focal_loss():
    input_good = torch.Tensor([10, -10, 10]).float()
    input_bad = torch.Tensor([-1, 2, 0]).float()
    target = torch.Tensor([1, 0, 1])

    loss_good = F.reduced_focal_loss(input_good, target)
    loss_bad = F.reduced_focal_loss(input_bad, target)
    assert loss_good < loss_bad


def test_soft_jaccard_score():
    input_good = torch.Tensor([1, 0, 1]).float()
    input_bad = torch.Tensor([0, 0, 0]).float()
    target = torch.Tensor([1, 0, 1])
    eps = 1e-5

    jaccard_good = F.soft_jaccard_score(input_good, target, smooth=eps)
    assert float(jaccard_good) == pytest.approx(1.0, eps)

    jaccard_bad = F.soft_jaccard_score(input_bad, target, smooth=eps)
    assert float(jaccard_bad) == pytest.approx(0.0, eps)


def test_soft_dice_score():
    input_good = torch.Tensor([1, 0, 1]).float()
    input_bad = torch.Tensor([0, 0, 0]).float()
    target = torch.Tensor([1, 0, 1])
    eps = 1e-5

    dice_good = F.soft_dice_score(input_good, target, smooth=eps)
    assert float(dice_good) == pytest.approx(1.0, eps)

    dice_bad = F.soft_dice_score(input_bad, target, smooth=eps)
    assert float(dice_bad) == pytest.approx(0.0, eps)

@pytest.mark.parametrize('loss_name', LOSSES_NAMES)
def test_correct_inheritance(loss_name):
    """test that all models are inherited from our custom loss"""
    assert issubclass(losses.__dict__[loss_name], losses.base.Loss)

@pytest.mark.parametrize('loss_name', LOSSES_NAMES)
def test_losses_init(loss_name):
    l = losses.__dict__[loss_name]()

def test_loss_addition():
    l = losses.BinaryDiceLoss() * 0.5 + losses.BinaryFocalLoss() * 5
    inp = torch.ones(2,1,8,8)
    label = torch.zeros(2,1,8,8)
    res = l(inp, label)
