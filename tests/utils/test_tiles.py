import torch
import torch.nn.functional as F
import pytest
import pytorch_tools as pt
from pytorch_tools.utils.tiles import compute_pyramid_patch_weight_loss, TileInference


def test_pyramid_patch_weight():
    H, W, D = 10, 20, 30
    weight_2d = compute_pyramid_patch_weight_loss(H, W)
    assert weight_2d.shape == torch.Size([1, H, W])
    # check that it's normalized
    assert torch.allclose(weight_2d.sum().round().long(), torch.tensor(weight_2d.shape).prod())

    weight_3d = compute_pyramid_patch_weight_loss(H, W, D)
    assert weight_3d.shape == torch.Size([1, H, W, D])
    assert torch.allclose(weight_3d.sum().round().long(), torch.tensor(weight_3d.shape).prod())

    SZ = 9
    w_2d = compute_pyramid_patch_weight_loss(SZ, SZ)
    w_3d = compute_pyramid_patch_weight_loss(SZ, SZ, SZ)
    # middle of the weight cube should be proportional to 2d case
    assert torch.allclose((w_2d / w_3d[..., SZ // 2]), torch.tensor(0.7053), atol=1e-4)

    w_2d = compute_pyramid_patch_weight_loss(5, 5)
    # fmt: off
    expected_w_2d = torch.tensor(
        [[[0.4513, 0.5490, 0.6008, 0.5490, 0.4513],
          [0.5490, 1.5463, 1.8025, 1.5463, 0.5490],
          [0.6008, 1.8025, 3.0042, 1.8025, 0.6008],
          [0.5490, 1.5463, 1.8025, 1.5463, 0.5490],
          [0.4513, 0.5490, 0.6008, 0.5490, 0.4513]]]
    )
    # fmt: on
    assert torch.allclose(expected_w_2d, w_2d, atol=1e-4)


@pytest.mark.parametrize("fusion", ["mean", "pyramid"])
@pytest.mark.parametrize("overlap", [0, 4, 6])
@pytest.mark.parametrize(
    "func_scale",
    [
        (lambda x: x.pow(2), 1),
        (lambda x: F.interpolate(x[None], scale_factor=2)[0], 2),
        (lambda x: F.avg_pool2d(x[None], 2)[0], 0.5),
    ],
)
def test_tile_inference_2d(func_scale, overlap, fusion):
    func, scale = func_scale
    C, H, W = 3, 32, 32
    large_img = torch.rand(C, H, W)
    tiler = TileInference(tile_size=(16, 16), overlap=overlap, model=func, scale=scale, fusion=fusion)
    tiler_res = tiler(large_img)
    full_res = func(large_img)
    assert torch.allclose(full_res, tiler_res)


def test_tile_inference_2d_large_overlap():
    with pytest.raises(ValueError):
        TileInference(tile_size=(16, 16), overlap=8, model=lambda x: x.pow(2))


def test_tile_inference_2d_int_size():
    with pytest.raises(ValueError):
        TileInference(tile_size=16, overlap=2, model=lambda x: x.pow(2))


@pytest.mark.parametrize("fusion", ["mean", "pyramid"])
@pytest.mark.parametrize("overlap", [0, 2])
@pytest.mark.parametrize(
    "func_scale",
    [
        (lambda x: x.pow(2), 1),
        (lambda x: F.interpolate(x[None], scale_factor=2)[0], 2),
        (lambda x: F.avg_pool3d(x[None], 2)[0], 0.5),
    ],
)
def test_tile_inference_3d(func_scale, overlap, fusion):
    func, scale = func_scale
    C, H, W, D = 3, 32, 32, 16
    large_img = torch.rand(C, H, W, D)
    tiler = TileInference(tile_size=(16, 16, 8), overlap=overlap, model=func, scale=scale, fusion=fusion)
    tiler_res = tiler(large_img)
    full_res = func(large_img)
    assert torch.allclose(full_res, tiler_res)
