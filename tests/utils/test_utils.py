import torch
import pytest
import pytorch_tools as pt


def random_boxes(mean_box, stdev, N):
    return torch.rand(N, 4) * stdev + torch.tensor(mean_box, dtype=torch.float)


# fmt: off
DEVICE_DTYPE = [
    ("cpu", torch.float),
    ("cuda", torch.float),
    ("cuda", torch.half)
]
# fmt: on
# check that it works for all combinations of dtype and device
@pytest.mark.parametrize("device_dtype", DEVICE_DTYPE)
def test_clip_bboxes(device_dtype):
    device, dtype = device_dtype
    # fmt: off
    bboxes = torch.tensor(
        [
            [-5, -10, 50, 100],
            [10, 15, 20, 25],
        ],
        device=device,
        dtype=dtype,
    )
    expected_bboxes = torch.tensor(
        [
            [0, 0, 40, 60],
            [10, 15, 20, 25],
        ],
        device=device,
        dtype=dtype,
    )
    # fmt: on
    size = (60, 40)
    # test single bbox clip
    res1 = pt.utils.box.clip_bboxes(bboxes, size)
    assert torch.allclose(res1, expected_bboxes)
    # test single bbox clip passing torch.Size
    res2 = pt.utils.box.clip_bboxes(bboxes, torch.Size(size))
    assert torch.allclose(res2, expected_bboxes)

    BS = 4
    batch_bboxes = bboxes.unsqueeze(0).expand(BS, -1, -1)
    batch_expected = expected_bboxes.unsqueeze(0).expand(BS, -1, -1)
    batch_sizes = torch.tensor(size).repeat(BS, 1)
    # test batch clipping
    res3 = pt.utils.box.clip_bboxes_batch(batch_bboxes.clone(), batch_sizes)
    assert torch.allclose(res3, batch_expected)

    # check that even in batch mode we can pass single size
    res4 = pt.utils.box.clip_bboxes_batch(batch_bboxes.clone(), torch.tensor(size))
    assert torch.allclose(res4, batch_expected)

    jit_clip = torch.jit.script(pt.utils.box.clip_bboxes_batch)
    # check that function is JIT script friendly
    res5 = jit_clip(batch_bboxes.clone(), batch_sizes)
    assert torch.allclose(res5, batch_expected)


@pytest.mark.parametrize("device_dtype", DEVICE_DTYPE)
def test_delta2box(device_dtype):
    device, dtype = device_dtype
    # fmt: off
    anchors = torch.tensor(
        [
            [ 0.,  0.,  1.,  1.],
            [ 0.,  0.,  1.,  1.],
            [ 0.,  0.,  1.,  1.],
            [ 5.,  5.,  5.,  5.]
        ],
        device=device,
        dtype=dtype,
    )
    deltas = torch.tensor(
        [
            [  0.,   0.,   0.,   0.],
            [  1.,   1.,   1.,   1.],
            [  0.,   0.,   2.,  -1.],
            [ 0.7, -1.9, -0.5,  0.3]
        ],
        device=device,
        dtype=dtype,
    )
    # by default we don't expect results to be clipped
    expected_res = torch.tensor(
        [
            [0.0000, 0.0000, 1.0000, 1.0000],
            [0.1409, 0.1409, 2.8591, 2.8591],
            [-3.1945, 0.3161, 4.1945, 0.6839],
            [5.0000, 5.0000, 5.0000, 5.0000],
        ],
        device=device,
        dtype=torch.float32, # should always be float!
    )
    # fmt: on
    res1 = pt.utils.box.delta2box(deltas, anchors)
    assert torch.allclose(res1, expected_res, atol=3e-4)

    BS = 4
    batch_anchors = anchors.unsqueeze(0).expand(BS, -1, -1)
    batch_deltas = deltas.unsqueeze(0).expand(BS, -1, -1)
    batch_expected = expected_res.unsqueeze(0).expand(BS, -1, -1)

    # test applying to batch
    res2 = pt.utils.box.delta2box(batch_deltas.clone(), batch_anchors)
    assert torch.allclose(res2, batch_expected, atol=3e-4)

    # check that function is JIT script friendly
    jit_func = torch.jit.script(pt.utils.box.delta2box)
    res3 = jit_func(batch_deltas.clone(), batch_anchors)
    assert torch.allclose(res3, batch_expected, atol=3e-4)


@pytest.mark.parametrize("device_dtype", DEVICE_DTYPE)
def test_box2delta(device_dtype):
    ## this test only checks that encoding and decoding  gives the same result
    device, dtype = device_dtype
    boxes = random_boxes([10, 10, 20, 20], 10, 10).to(device).to(dtype)
    anchors = random_boxes([10, 10, 20, 20], 10, 10).to(device).to(dtype)
    deltas = pt.utils.box.box2delta(boxes, anchors)
    boxes_reconstructed = pt.utils.box.delta2box(deltas, anchors)
    # output of box2delta should always be float to avoid numerical instability
    assert torch.allclose(boxes.float(), boxes_reconstructed, atol=1e-6)

    # check that it's jit friendly
    jit_box2delta = torch.jit.script(pt.utils.box.box2delta)
    jit_delta2box = torch.jit.script(pt.utils.box.delta2box)
    deltas2 = jit_box2delta(boxes, anchors)
    boxes_reconstructed2 = jit_delta2box(deltas2, anchors)
    assert torch.allclose(boxes.float(), boxes_reconstructed2, atol=1e-6)


def test_generate_anchors():
    # check that anchor generation is not broken
    anchors = pt.utils.box.generate_anchors_boxes((64, 64))[0]
    # check number of anchors
    assert anchors.shape == (765, 4)
    # check that mean is the same
    assert torch.allclose(anchors.mean(0), torch.tensor([1.8604, 1.8604, 62.1393, 62.1393]), rtol=5e-5)
    # check that it is really xyxy order
    assert torch.allclose(anchors[-1], torch.tensor([-111.6751, -255.3503, 175.6751, 319.3503]), rtol=5e-5)

    # test anchors explicitly
    # fmt: off
    expected = torch.tensor(
        [
            [-12, -12, 20, 20],
            [ -4, -28, 12, 36],
            [-28,  -4, 36, 12],
            [ -4, -12, 28, 20],
            [  4, -28, 20, 36],
            [-20,  -4, 44, 12],
            [-12,  -4, 20, 28],
            [ -4, -20, 12, 44],
            [-28,   4, 36, 20],
            [ -4,  -4, 28, 28],
            [  4, -20, 20, 44],
            [-20,   4, 44, 20],
        ]
    ).float()
    # fmt: on
    generated, num_anchors = pt.utils.box.generate_anchors_boxes(
        16, num_scales=1, aspect_ratios=(1, 0.25, 4), pyramid_levels=[3,]
    )
    assert torch.allclose(expected, generated)
    assert num_anchors == 3

    # check that it is scriptable
    jit_generate = torch.jit.script(pt.utils.box.generate_anchors_boxes)
    generated, num_anchors = jit_generate(
        (16, 16), num_scales=1, aspect_ratios=(1, 0.25, 4), pyramid_levels=[3,]
    )
    assert torch.allclose(expected, generated)
    assert num_anchors == 3


def test_box_iou():
    """Make sure IoU is calculated correctly"""
    # fmt: off
    bboxes1 = torch.tensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 19],
        [32, 32, 38, 42],
    ]).float()
    bboxes2 = torch.tensor([
        [0, 0, 10, 9],
        [0, 5, 12, 19],
    ]).float()
    expected_res = torch.tensor([
        [0.9000, 0.2294], # 0.9 = 90 / 100; 0.2294 = 5*10 / (12 * 14 + 10*10 - 5*10)
        [0.0000, 0.0720],
        [0.0952, 0.4667],
        [0.0000, 0.0000]
    ]).float()
    # fmt: on
    res = pt.utils.box.box_iou(bboxes1, bboxes2)
    assert torch.allclose(res, expected_res, atol=1e-4)


def test_iou_for_zero_bbox():
    b1 = random_boxes([10, 10, 10, 10], 10, 7)
    b2 = torch.zeros(5, 4)
    iou = pt.utils.box.box_iou(b1, b2)
    assert torch.allclose(iou, torch.zeros(7, 5))


@pytest.mark.parametrize("device_dtype", DEVICE_DTYPE)
def test_batch_iou(device_dtype):
    """check that batch iou is the same as calculating it for every image separately"""
    device, dtype = device_dtype
    anchors = random_boxes([10, 10, 20, 20], 10, 7).to(device).to(dtype)
    b_bboxes2 = torch.stack([random_boxes([10, 10, 20, 20], 10, 15) for _ in range(5)]).to(device).to(dtype)

    batch_res = pt.utils.box.batch_box_iou(anchors, b_bboxes2)
    separ_res = torch.stack([pt.utils.box.box_iou(anchors, bb2) for bb2 in b_bboxes2])
    assert torch.allclose(batch_res, separ_res)

    # check that functions are scriptable
    jit_batch_iou = torch.jit.script(pt.utils.box.batch_box_iou)
    jit_iou = torch.jit.script(pt.utils.box.box_iou)

    jit_batch_res = jit_batch_iou(anchors, b_bboxes2)
    jit_separ_res = torch.stack([jit_iou(anchors, bb2) for bb2 in b_bboxes2])
    assert torch.allclose(jit_batch_res, jit_separ_res)

    assert torch.allclose(jit_batch_res, batch_res)


def test_generate_targets():
    """checks that generated targets are correct"""
    # fmt: off
    bboxes = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 19],
        [11, 18, 38, 42],
    ])
    gt_bboxes = torch.tensor([
        [0, 0, 10, 9],
        [0, 5, 12, 19],
    ])
    expected_cls_target = torch.tensor([[
        [0., 0., 1., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]
    ]])
    expected_matches_mask = torch.tensor([[1, 0, 0, 0]])
    expected_boxes = torch.tensor([[
        [ 0.,  0., 10.,  9.],
        [ 0.,  5., 12., 19.],
        [ 0.,  5., 12., 19.],
        [ 0.,  5., 12., 19.]
    ]])
    gt_labels = torch.tensor([2, 3])
    gt = torch.cat([gt_bboxes, gt_labels.unsqueeze(1)], 1)[None].float()
    # fmt: on

    # test with high unmatched iou
    box_target, cls_target, matches_mask = pt.utils.box.generate_targets(bboxes, gt, 4, unmatched_iou=0.5)
    assert torch.allclose(cls_target, expected_cls_target)
    assert torch.allclose(matches_mask, expected_matches_mask)
    # predicted target after regression should give one of ground truth bboxes
    regressed_target = pt.utils.box.delta2box(box_target, bboxes)
    assert torch.allclose(regressed_target, expected_boxes)

    # test lower unmatched iou. one bbox should be ignored in this case
    expected_matches_mask2 = torch.tensor([[1, 0, -1, 0]])
    _, _, matches_mask2 = pt.utils.box.generate_targets(bboxes, gt, 4, unmatched_iou=0.4)
    assert torch.allclose(expected_matches_mask2, matches_mask2)


def test_generate_empty_true_targets():
    """Test behaviour for empty true boxes"""
    bboxes = random_boxes([10, 10, 20, 20], 10, 10)
    gt = torch.ones((2, 5)) * -1  # empty
    _, cls_target, matches_mask = pt.utils.box.generate_targets(bboxes, gt[None], 4)
    # check that output contains only zeros
    assert torch.allclose(matches_mask, torch.zeros_like(matches_mask))
    assert torch.allclose(cls_target, torch.zeros_like(cls_target))


def test_generate_targes_is_scriptable():
    bboxes = random_boxes([10, 10, 20, 20], 10, 10)
    N_CLASSES = 10
    gt = torch.randint(N_CLASSES, (2, 5)).float()[None]
    gt[..., 2:4] += gt[..., :2]  # make sure bbox is correct
    jit_func = torch.jit.script(pt.utils.box.generate_targets)
    box_t, cls_t, matches = pt.utils.box.generate_targets(bboxes, gt, N_CLASSES)
    box_t_jit, cls_t_jit, matches_jit = jit_func(bboxes, gt, N_CLASSES)
    assert torch.allclose(box_t, box_t_jit)
    assert torch.allclose(cls_t, cls_t_jit)
    assert torch.allclose(matches, matches_jit)
