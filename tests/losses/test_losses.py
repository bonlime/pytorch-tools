import pytest
import torch
import numpy as np
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


EPS = 1e-3
@pytest.mark.parametrize(
    ["y_true", "y_pred", "expected"],
    [
        [[1, 1, 1, 1], [1, 1, 1, 1], 1.0],
        [[0, 1, 1, 0], [0, 1, 1, 0], 1.0],
        [[1, 1, 1, 1], [1, 1, 0, 0], 0.5],
    ],
)
def test_soft_jaccard_score(y_true, y_pred, expected):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    actual = F.soft_jaccard_score(y_pred, y_true)
    assert float(actual) == pytest.approx(expected, EPS)


@pytest.mark.parametrize(
    ["y_true", "y_pred", "expected"],
    [
        [[[1, 1, 0, 0], [0, 0, 1, 1]], [[1, 1, 0, 0], [0, 0, 1, 1]], 1.0],
        [[[1, 1, 0, 0], [0, 0, 1, 1]], [[0, 0, 1, 0], [0, 1, 0, 0]], 0.0],
        [[[1, 1, 0, 0], [0, 0, 0, 1]], [[1, 1, 0, 0], [0, 0, 0, 0]], 0.5],
    ],
)
def test_soft_jaccard_score_2(y_true, y_pred, expected):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    actual = F.soft_jaccard_score(y_pred, y_true, dims=[1])
    actual = actual.mean()
    assert float(actual) == pytest.approx(expected, abs=EPS)


@pytest.mark.parametrize(
    ["y_true", "y_pred", "expected"],
    [
        [[1, 1, 1, 1], [1, 1, 1, 1], 1.0],
        [[0, 1, 1, 0], [0, 1, 1, 0], 1.0],
        [[1, 1, 1, 1], [1, 1, 0, 0], 2.0 / 3.0],
    ],
)
def test_soft_dice_score(y_true, y_pred, expected):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    dice_good = F.soft_dice_score(y_pred, y_true)
    assert float(dice_good) == pytest.approx(expected, abs=EPS)


@torch.no_grad()
def test_dice_loss_binary():
    criterion = losses.DiceLoss(mode="binary", from_logits=False)

    # Ideal case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 1, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=EPS)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 0, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=EPS)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([0, 0, 0])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=EPS)

    # Worst case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 0, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    # It returns 1. due to internal smoothing
    assert float(loss) == pytest.approx(1.0, abs=EPS)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 1, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=EPS)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, -1)
    y_true = torch.tensor([1, 1, 1]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=EPS)


@torch.no_grad()
def test_binary_jaccard_loss():
    criterion = losses.JaccardLoss(mode="binary", from_logits=False)

    # Ideal case
    y_pred = torch.tensor([1.0]).view(1, 1, 1, 1)
    y_true = torch.tensor(([1])).view(1, 1, 1, 1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=EPS)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 0, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=EPS)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([0, 0, 0])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=EPS)

    # Worst case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 0, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    # It returns 1. due to internal smoothing
    assert float(loss) == pytest.approx(1.0, abs=EPS)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 1, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, EPS)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, -1)
    y_true = torch.tensor([1, 1, 1]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, EPS)


@torch.no_grad()
def test_multiclass_jaccard_loss():
    criterion = losses.JaccardLoss(mode="multiclass", from_logits=False)

    # Ideal case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = torch.tensor([[0, 0, 1, 1]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=EPS)

    # Worst case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = torch.tensor([[1, 1, 0, 0]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=EPS)

    # 1 - 1/3 case
    y_pred = torch.tensor([[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]])
    y_true = torch.tensor([[1, 1, 0, 0]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0 - 1.0 / 3.0, abs=EPS)


@torch.no_grad()
def test_multilabel_jaccard_loss():
    criterion = losses.JaccardLoss(mode="multilabel", from_logits=False)

    # Ideal case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=EPS)

    # Worst case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = 1 - y_pred
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=EPS)

    # 1 - 1/3 case
    y_pred = torch.tensor([[[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]])
    y_true = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0 - 1.0 / 3.0, abs=EPS)


@pytest.mark.parametrize('loss_name', LOSSES_NAMES)
def test_correct_inheritance(loss_name):
    """test that all models are inherited from our custom loss"""
    assert issubclass(losses.__dict__[loss_name], losses.base.Loss)

@pytest.mark.parametrize('loss_name', LOSSES_NAMES)
def test_losses_init(loss_name):
    l = losses.__dict__[loss_name]()

def test_loss_addition():
    l = losses.DiceLoss('binary') * 0.5 + losses.BinaryFocalLoss() * 5
    inp = torch.ones(2,1,8,8)
    label = torch.zeros(2,1,8,8)
    res = l(inp, label)

@torch.no_grad()
def test_cross_entropy():
    inp = torch.randn(100,10)
    target = torch.randint(0,10,(100,)).long()
    tar_one_hot = torch.zeros(target.size(0), 10, dtype=torch.float)
    tar_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
    c = np.random.beta(0.4, 0.4)
    perm = torch.randperm(100)
    tar_one_hot_2 = tar_one_hot * c + (1-c) * tar_one_hot[perm,:]

    torch_ce = torch.nn.CrossEntropyLoss()(inp, target)
    my_ce = losses.CrossEntropyLoss()(inp, target)
    assert torch.allclose(torch_ce, my_ce) 

    my_ce_oh = losses.CrossEntropyLoss()(inp, tar_one_hot)
    assert torch.allclose(torch_ce, my_ce_oh)

    my_ce_oh_2 = losses.CrossEntropyLoss()(inp, tar_one_hot_2)
    assert not torch.allclose(torch_ce, my_ce_oh_2)

    my_ce_sm = losses.CrossEntropyLoss(smoothing=0.1)(inp, target)
    assert not torch.allclose(my_ce_sm, my_ce)

