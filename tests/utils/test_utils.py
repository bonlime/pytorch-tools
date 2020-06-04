import torch
import pytest
import pytorch_tools as pt

def random_boxes(mean_box, stdev, N):
    return torch.rand(N, 4) * stdev + torch.tensor(mean_box, dtype=torch.float)

DEVICE_DTYPE =  [
    ("cpu", torch.float), 
    ("cuda", torch.float), 
    ("cuda", torch.half)
]
# check that it works for all combinations of dtype and device
@pytest.mark.parametrize("device_dtype", DEVICE_DTYPE)
def test_clip_bboxes(device_dtype):
    device, dtype = device_dtype
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
            [5.0000, 5.0000, 5.0000, 5.0000]
        ],
        device=device,
        dtype=dtype,
    )
    res1 =  pt.utils.box.delta2box(deltas, anchors)
    assert torch.allclose(res1, expected_res, atol=3e-4)

    BS = 4
    batch_anchors = anchors.unsqueeze(0).expand(BS, -1, -1)
    batch_deltas = deltas.unsqueeze(0).expand(BS, -1, -1)
    batch_expected = expected_res.unsqueeze(0).expand(BS, -1, -1)

    # test applying to batch 
    res2 =  pt.utils.box.delta2box(batch_deltas.clone(), batch_anchors)
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
    atol = 2e-2 if dtype == torch.half else 1e-6 # for fp16 sometimes error is large 
    assert torch.allclose(boxes, boxes_reconstructed, atol=atol) 

    # check that it's jit friendly
    jit_box2delta = torch.jit.script(pt.utils.box.box2delta)
    jit_delta2box = torch.jit.script(pt.utils.box.delta2box)
    deltas2 = jit_box2delta(boxes, anchors)
    boxes_reconstructed2 = jit_delta2box(deltas2, anchors)
    assert torch.allclose(boxes, boxes_reconstructed2,  atol=atol)