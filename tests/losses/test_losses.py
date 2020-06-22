import pytest
import numpy as np
import torch
import torch.nn.functional as F

import pytorch_tools as pt
import pytorch_tools.losses as losses
import pytorch_tools.losses.functional as pt_F
from pytorch_tools.utils.misc import set_random_seed

"""
Some test were taken from repo by BloodAxe
https://github.com/BloodAxe/pytorch-toolbelt/
"""

set_random_seed(42)

EPS = 1e-3
BS = 16
N_CLASSES = 10
IM_SIZE = 10
INP = torch.randn(BS, N_CLASSES)
INP_BINARY = torch.randn(BS).float()
INP_IMG = torch.rand(BS, N_CLASSES, IM_SIZE, IM_SIZE)
INP_IMG_BINARY = torch.rand(BS, 1, IM_SIZE, IM_SIZE)

TARGET = torch.randint(0, N_CLASSES, (BS,)).long()
TARGET_BINARY = torch.randint(0, 2, (BS,)).float()
TARGET_IMG_BINARY = torch.randint(0, 2, (BS, 1, IM_SIZE, IM_SIZE)).float()
TARGET_IMG_MULTICLASS = torch.randint(0, N_CLASSES, (BS, IM_SIZE, IM_SIZE)).long()

TARGET_MULTILABEL = torch.zeros_like(INP)
TARGET_MULTILABEL.scatter_(1, TARGET.unsqueeze(1), 1.0)
TARGET_IMG_MULTILABEL = torch.zeros_like(INP_IMG)
TARGET_IMG_MULTILABEL.scatter_(1, TARGET_IMG_MULTICLASS.unsqueeze(1), 1.0)

LOSSES_NAMES = sorted(name for name in losses.__dict__ if not name.islower())

# this tests are slow. it's usefull to sometimes remove them.
LOSSES_NAMES.pop(LOSSES_NAMES.index("ContentLoss"))
LOSSES_NAMES.pop(LOSSES_NAMES.index("StyleLoss"))

## Tests for Focal loss
def test_focal_loss_fn_basic():
    """explicit tests for values in two corner cases"""
    input_good = torch.Tensor([10, -10, 10]).float()
    input_bad = torch.Tensor([-1, 2, 0]).float()
    target = torch.Tensor([1, 0, 1])

    loss_good = pt_F.focal_loss_with_logits(input_good, target)
    loss_bad = pt_F.focal_loss_with_logits(input_bad, target)
    assert torch.allclose(loss_good, torch.tensor(0.0))
    assert torch.allclose(loss_bad, torch.tensor(0.4854), atol=1e-4)


@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
def test_focal_loss_fn_reduction(reduction):
    torch_ce = F.binary_cross_entropy_with_logits(INP_BINARY, TARGET_BINARY.float(), reduction=reduction)
    my_ce = pt_F.focal_loss_with_logits(INP_BINARY, TARGET_BINARY, alpha=0.5, gamma=0, reduction=reduction)
    assert torch.allclose(torch_ce, my_ce * 2)


def test_focal_loss_fn():
    # classification test
    torch_ce = F.binary_cross_entropy_with_logits(INP_BINARY, TARGET_BINARY.float())
    my_ce = pt_F.focal_loss_with_logits(INP_BINARY, TARGET_BINARY, alpha=-1, gamma=0)
    assert torch.allclose(torch_ce, my_ce)

    # check that smooth combination works
    my_ce_not_reduced = pt_F.focal_loss_with_logits(INP_BINARY, TARGET_BINARY, combine_thr=0)
    my_ce_reduced = pt_F.focal_loss_with_logits(INP_BINARY, TARGET_BINARY, combine_thr=0.2)
    my_ce_reduced2 = pt_F.focal_loss_with_logits(INP_BINARY, TARGET_BINARY, combine_thr=0.8)
    assert my_ce_not_reduced < my_ce_reduced
    assert my_ce_reduced < my_ce_reduced2

    # images test
    torch_ce = F.binary_cross_entropy_with_logits(INP_IMG_BINARY, TARGET_IMG_BINARY)
    my_ce = pt_F.focal_loss_with_logits(INP_IMG_BINARY, TARGET_IMG_BINARY, alpha=-1, gamma=0)
    assert torch.allclose(torch_ce, my_ce)


def test_focal_loss_fn_normalize():
    # simulate very accurate predictions
    inp = TARGET_BINARY * 5 - (1 - TARGET_BINARY) * 5
    my_ce = pt_F.focal_loss_with_logits(inp, TARGET_BINARY, normalized=False)
    my_ce_normalized = pt_F.focal_loss_with_logits(inp, TARGET_BINARY, normalized=True)
    assert my_ce_normalized > my_ce


def test_focal_loss_modes():
    # check that multilabel == one-hot-encoded multiclass
    fl_multiclass = losses.FocalLoss(mode="multiclass", reduction="sum")(INP_IMG, TARGET_IMG_MULTICLASS)
    fl_multilabel = losses.FocalLoss(mode="multilabel", reduction="sum")(INP_IMG, TARGET_IMG_MULTILABEL)
    assert fl_multiclass == fl_multilabel

    # check that ignore index works for multiclass
    fl = losses.FocalLoss(mode="multiclass", reduction="none")(INP_IMG, TARGET_IMG_MULTICLASS)
    loss_diff = fl[:, :, :2, :2].sum()
    y_true = TARGET_IMG_MULTICLASS.clone()
    y_true[:, :2, :2] = -100
    fl_i = losses.FocalLoss(mode="multiclass", reduction="sum", ignore_label=-100)(INP_IMG, y_true)
    assert torch.allclose(fl.sum() - loss_diff, fl_i)

    # check that ignore index works for binary
    fl = losses.FocalLoss(mode="binary", reduction="none")(INP_IMG_BINARY, TARGET_IMG_BINARY)
    loss_diff = fl[:, :, :2, :2].sum()
    y_true = TARGET_IMG_BINARY.clone()
    y_true[:, :, :2, :2] = -100
    fl_i = losses.FocalLoss(mode="binary", reduction="sum", ignore_label=-100)(INP_IMG_BINARY, y_true)
    assert torch.allclose(fl.sum() - loss_diff, fl_i)


def test_focal_incorrect_mode():
    with pytest.raises(ValueError):
        losses.FocalLoss(mode="some_mode")


def test_focal_incorrect_reduction():
    with pytest.raises(ValueError):
        losses.FocalLoss(reduction="some_reduction")


def test_focal_fn_is_scribtable():
    """check that script gives the same results"""
    input_bad = torch.Tensor([-1, 2, 0]).float()
    target = torch.Tensor([1, 0, 1])
    loss = pt_F.focal_loss_with_logits(input_bad, target)
    jit_func = torch.jit.script(pt_F.focal_loss_with_logits)
    loss_jit = jit_func(input_bad, target)
    assert torch.allclose(loss, loss_jit)


def test_focal_class_is_scribtable():
    """check that script gives the same results"""
    input_bad = torch.Tensor([-1, 2, 0]).float()
    target = torch.Tensor([1, 0, 1])
    loss = losses.FocalLoss()(input_bad, target)
    jit_class = torch.jit.script(losses.FocalLoss())
    loss_jit = jit_class(input_bad, target)
    assert torch.allclose(loss, loss_jit)


## Tests for JaccardLoss
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
    actual = pt_F.soft_jaccard_score(y_pred, y_true)
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
    actual = pt_F.soft_jaccard_score(y_pred, y_true, dims=[1])
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
    dice_good = pt_F.soft_dice_score(y_pred, y_true)
    assert float(dice_good) == pytest.approx(expected, abs=EPS)


@torch.no_grad()
def test_dice_loss_binary():
    criterion = losses.DiceLoss(mode="binary", from_logits=False, eps=1e-4)

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
    # It zeros loss if there is no y_true
    assert float(loss) == pytest.approx(0.0, abs=EPS)

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
    criterion = losses.JaccardLoss(mode="binary", from_logits=False, eps=1e-4)

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
    # It zeros loss if there is no y_true
    assert float(loss) == pytest.approx(0.0, abs=EPS)

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
    criterion = losses.JaccardLoss(mode="multiclass", from_logits=False, eps=1e-4)

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
    criterion = losses.JaccardLoss(mode="multilabel", from_logits=False, eps=1e-4)

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


@pytest.mark.parametrize("loss_name", LOSSES_NAMES)
def test_correct_inheritance(loss_name):
    """test that all models are inherited from our custom loss"""
    assert issubclass(losses.__dict__[loss_name], losses.base.Loss)


@pytest.mark.parametrize("loss_name", LOSSES_NAMES)
def test_losses_init(loss_name):
    l = losses.__dict__[loss_name]()


def test_loss_addition():
    d_l = losses.DiceLoss("binary")
    bf_l = losses.CrossEntropyLoss(mode="binary")
    l = losses.DiceLoss("binary") * 0.5 + losses.CrossEntropyLoss(mode="binary") * 5
    d_res = d_l(INP_IMG_BINARY, TARGET_IMG_BINARY)
    bf_res = bf_l(INP_IMG_BINARY, TARGET_IMG_BINARY)
    res = l(INP_IMG_BINARY, TARGET_IMG_BINARY)
    assert res.shape == d_res.shape
    assert torch.allclose(res, d_res * 0.5 + bf_res * 5)


# comment out because it requires loading GPU which is slow
# def test_loss_multiplication():
#     """checks that weights are moved to the loss device as well"""
#     l = (losses.CrossEntropyLoss(mode="binary") * 5).cuda()
#     res = l(INP_IMG_BINARY.cuda(), TARGET_IMG_BINARY.cuda())
#     assert res


@torch.no_grad()
def test_cross_entropy():
    c = np.random.beta(0.4, 0.4)
    perm = torch.randperm(BS)
    tar_one_hot_2 = TARGET_MULTILABEL * c + (1 - c) * TARGET_MULTILABEL[perm, :]
    my_ce_loss = losses.CrossEntropyLoss()
    torch_ce = torch.nn.CrossEntropyLoss()(INP, TARGET)
    my_ce = my_ce_loss(INP, TARGET)
    assert torch.allclose(torch_ce, my_ce)

    my_ce_oh = my_ce_loss(INP, TARGET_MULTILABEL)
    assert torch.allclose(torch_ce, my_ce_oh)

    my_ce_oh_2 = my_ce_loss(INP, tar_one_hot_2)
    assert not torch.allclose(torch_ce, my_ce_oh_2)

    my_ce_sm = losses.CrossEntropyLoss(smoothing=0.1)(INP, TARGET)
    assert not torch.allclose(my_ce_sm, my_ce)


@torch.no_grad()
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
def test_binary_cross_entropy(reduction):
    # classification test
    torch_ce = F.binary_cross_entropy_with_logits(INP_BINARY, TARGET_BINARY, reduction=reduction)
    my_ce_loss = losses.CrossEntropyLoss(mode="binary", reduction=reduction)
    my_ce = my_ce_loss(INP_BINARY, TARGET_BINARY)
    assert torch.allclose(torch_ce, my_ce)

    # test for images
    torch_ce = F.binary_cross_entropy_with_logits(INP_IMG_BINARY, TARGET_IMG_BINARY, reduction=reduction)
    my_ce = my_ce_loss(INP_IMG_BINARY, TARGET_IMG_BINARY)
    assert torch.allclose(torch_ce, my_ce)

    my_ce = my_ce_loss(INP_IMG_BINARY.squeeze(), TARGET_IMG_BINARY.squeeze())
    assert torch.allclose(torch_ce.squeeze(), my_ce.squeeze())

    # test for images with different y_true shape
    my_ce = my_ce_loss(INP_IMG_BINARY.squeeze(), TARGET_IMG_BINARY)
    assert torch.allclose(torch_ce.squeeze(), my_ce.squeeze())

    my_ce = my_ce_loss(INP_IMG_BINARY, TARGET_IMG_BINARY.squeeze())
    assert torch.allclose(torch_ce.squeeze(), my_ce.squeeze())


def test_binary_cross_entropy_from_logits():
    """Check that passing from_logits True and taking sigmoid manually gives the same result"""
    loss_1 = losses.CrossEntropyLoss(mode="binary")
    res_1 = loss_1(INP_BINARY, TARGET_BINARY)
    loss_2 = losses.CrossEntropyLoss(mode="binary", from_logits=False)
    res_2 = loss_2(INP_BINARY.sigmoid(), TARGET_BINARY)
    assert torch.allclose(res_1, res_2)


def test_cross_entropy_from_logits():
    """Check that passing from_logits True and taking softmax manually gives the same result"""
    loss_1 = losses.CrossEntropyLoss()
    res_1 = loss_1(INP, TARGET)
    loss_2 = losses.CrossEntropyLoss(from_logits=False)
    res_2 = loss_2(INP.softmax(1), TARGET)
    assert torch.allclose(res_1, res_2)


@torch.no_grad()
def test_cross_entropy_weight():
    weight_1 = torch.randint(1, 100, (N_CLASSES,)).float()
    weight_2 = weight_1.numpy().astype(int)
    weight_3 = list(weight_2)

    torch_ce_w = torch.nn.CrossEntropyLoss(weight=weight_1)(INP, TARGET)
    my_ce_w = losses.CrossEntropyLoss(weight=weight_1)(INP, TARGET)
    assert torch.allclose(torch_ce_w, my_ce_w)

    my_ce_w = losses.CrossEntropyLoss(weight=weight_2)(INP, TARGET)
    assert torch.allclose(torch_ce_w, my_ce_w)

    my_ce_w = losses.CrossEntropyLoss(weight=weight_3)(INP, TARGET)
    assert torch.allclose(torch_ce_w, my_ce_w)


# For lovasz tests only check that it doesn't fail, not that results are right


def test_binary_lovasz():
    loss = losses.LovaszLoss(mode="binary")(INP_IMG_BINARY, TARGET_IMG_BINARY)
    # try other target shape
    target = TARGET_IMG_BINARY.view(BS, IM_SIZE, IM_SIZE)
    loss2 = losses.LovaszLoss(mode="binary")(INP_IMG_BINARY, target)
    assert torch.allclose(loss, loss2)


def test_multiclass_multilabel_lovasz():
    loss = losses.LovaszLoss(mode="multiclass")(INP_IMG, TARGET_IMG_MULTICLASS)
    loss2 = losses.LovaszLoss(mode="multilabel")(INP_IMG, TARGET_IMG_MULTILABEL)
    assert torch.allclose(loss, loss2)


def test_binary_hinge():
    assert losses.BinaryHinge()(INP_IMG_BINARY, TARGET_IMG_BINARY)


@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
def test_smoothl1(reduction):
    loss_my = losses.SmoothL1Loss(delta=1, reduction=reduction)(INP, TARGET_MULTILABEL)
    loss_torch = F.smooth_l1_loss(INP, TARGET_MULTILABEL, reduction=reduction)
    assert torch.allclose(loss_my, loss_torch)
    # check that delta = 0 turns it into l1
    loss_my = losses.SmoothL1Loss(delta=0, reduction=reduction)(INP, TARGET_MULTILABEL)
    loss_torch = F.l1_loss(INP, TARGET_MULTILABEL, reduction=reduction)
    assert torch.allclose(loss_my, loss_torch)


def test_detection_loss_is_scriptabble():
    # this is a piece of real target and box/cls outputs for COCO images
    # fmt:off
    target = torch.tensor([
        [[190., 114., 209., 170.,  0.], [  6., 134., 125., 210.,  0.]],
        [[  1.,  55., 469., 506.,  1.], [ -1.,  -1.,  -1.,  -1., -1.]],
        [[  3., 213., 324., 380.,  1.], [147., 109., 195., 183.,  1.]],
        [[123.,  74., 261., 230.,  1.], [126., 286., 149., 317.,  1.]],
        [[  1.,  45., 276., 422.,  0.], [  2., 223., 256., 506.,  0.]]
    ])
    anchors = torch.tensor([
        [-12.0000, -12.0000,  20.0000,  20.0000], 
        [-18.6274,  -7.3137,  26.6274,  15.3137]
    ])
    cls_out = torch.tensor([
        [[-3.4645, -4.9549], [-6.3456, -7.3695]], 
        [[-4.7375, -5.5270], [-5.4755, -6.7083]],
        [[-3.7339, -4.7242], [-6.2671, -6.8925]],
        [[-3.6217, -4.7811], [-5.6993, -6.7674]],
        [[-5.2376, -6.1608], [-6.1039, -7.0789]]
    ])
    box_out = torch.tensor([
        [[-0.0600,  6.1500, -21.8200,   0.2200], [ 0.0800,  0.0400,  -1.1300,  -1.4700]],
        [[ 7.5100, 13.2500,   0.2400, -17.7700], [ 0.2800,  0.0400,  -0.6500,  -3.1200]],
        [[-0.0000,  5.2600, -21.9700,   0.4900], [ 0.0700,  0.0500,  -1.2000,  -1.4800]],
        [[ 6.8300,  8.2000, -21.9800,  -0.5200], [ 0.1000, -0.0200,  -1.2300,  -1.4100]],
        [[ 4.0100,  5.3500, -21.8100,   0.9900], [ 0.1700, -0.1000,  -0.9500,  -1.6500]]
    ])

    # fmt: on
    loss = losses.DetectionLoss(anchors=anchors)
    loss_jit = torch.jit.script(loss)
    res = loss((cls_out, box_out), target)
    res_jit = loss_jit((cls_out, box_out), target)
    assert torch.allclose(res, res_jit)
    assert torch.allclose(res, torch.tensor(5e-07))
