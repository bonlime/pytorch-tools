import torch
import torch.nn as nn

from pytorch_tools.utils.fuse import fuse_model

BS, IN_CHS, OUT_CHS, SZ, N_ITER = 2, 128, 64, 5, 2
inp1d = torch.rand(BS, IN_CHS)
inp2d = torch.rand(BS, IN_CHS, SZ, SZ)
inp3d = torch.rand(BS, IN_CHS, SZ, SZ, SZ)


def test_model_fuse_simple_1d():
    linear_bn = nn.Sequential(nn.Linear(IN_CHS, OUT_CHS), nn.BatchNorm1d(OUT_CHS))
    for _ in range(N_ITER):
        linear_bn(inp1d)
    linear_bn.eval()
    out = linear_bn(inp1d)
    out_f = fuse_model(linear_bn)(inp1d)
    assert torch.allclose(out, out_f, atol=1e-5)
    assert isinstance(linear_bn[1], nn.Identity)

    linear_bn = nn.Sequential(nn.Linear(IN_CHS, OUT_CHS, bias=False), nn.BatchNorm1d(OUT_CHS))
    for _ in range(N_ITER):
        linear_bn(inp1d)
    linear_bn.eval()
    out = linear_bn(inp1d)
    out_f = fuse_model(linear_bn)(inp1d)
    assert torch.allclose(out, out_f, atol=1e-5)
    assert isinstance(linear_bn[1], nn.Identity)


def test_model_fuse_simple_2d():
    conv_bn = nn.Sequential(nn.Conv2d(IN_CHS, OUT_CHS, 3), nn.BatchNorm2d(OUT_CHS))
    for _ in range(N_ITER):  # warmup buffers to have not identity values
        conv_bn(inp2d)
    conv_bn = conv_bn.eval()
    out = conv_bn(inp2d)
    out_f = fuse_model(conv_bn)(inp2d)
    assert torch.allclose(out, out_f, atol=1e-5)
    assert isinstance(conv_bn[1], nn.Identity)

    bn_conv = nn.Sequential(nn.BatchNorm2d(IN_CHS), nn.Conv2d(IN_CHS, OUT_CHS, 3))
    for _ in range(N_ITER):  # warmup buffers to have not identity values
        bn_conv(inp2d)
    bn_conv = bn_conv.eval()
    out = bn_conv(inp2d)
    out_f = fuse_model(bn_conv)(inp2d)
    assert torch.allclose(out, out_f, atol=1e-5)
    assert isinstance(bn_conv[0], nn.Identity)


def test_model_fuse_mupltiple_bn():
    tconv = nn.Sequential(
        nn.BatchNorm2d(IN_CHS),
        nn.ConvTranspose2d(IN_CHS, OUT_CHS, 3),
        nn.ConvTranspose2d(OUT_CHS, IN_CHS, 3),
        nn.BatchNorm2d(IN_CHS),
    )
    for _ in range(N_ITER):  # warmup buffers to have not identity values
        tconv(inp2d)
    tconv = tconv.eval()
    out = tconv(inp2d)
    out_f = fuse_model(tconv)(inp2d)
    assert torch.allclose(out, out_f, atol=1e-5)
    # first conv shouldn't be converted
    assert isinstance(tconv[0], nn.BatchNorm2d) and isinstance(tconv[3], nn.Identity)

    # same tests but for 3d convs
    conv3d = nn.Sequential(
        nn.BatchNorm3d(IN_CHS),
        nn.Conv3d(IN_CHS, OUT_CHS, 3),
        nn.Conv3d(OUT_CHS, IN_CHS, 3),
        nn.BatchNorm3d(IN_CHS),
        nn.ConvTranspose3d(IN_CHS, OUT_CHS, 3),
        nn.BatchNorm3d(OUT_CHS),
    )
    for _ in range(N_ITER):  # warmup buffers to have not identity values
        conv3d(inp3d)
    conv3d = conv3d.eval()
    out = conv3d(inp3d)
    out_f = fuse_model(conv3d)(inp3d)
    assert torch.allclose(out, out_f, atol=1e-5)
    assert (
        isinstance(conv3d[0], nn.Identity) and isinstance(conv3d[3], nn.Identity) and isinstance(conv3d[5], nn.Identity)
    )


def test_model_fuse_resnet():
    from torchvision.models import resnet34, resnet50

    r34 = resnet34().eval().requires_grad_(False)
    param_before = sum(x.numel() for x in r34.parameters())
    inp = torch.rand(2, 3, 64, 64)
    out = r34(inp)
    r34_fuse = fuse_model(r34)
    param_after = sum(x.numel() for x in r34_fuse.parameters())
    out_fuse = r34_fuse(inp)
    assert torch.allclose(out, out_fuse, atol=1e-5)
    assert param_after < param_before
    has_any_bn = any([isinstance(m, nn.BatchNorm2d) for m in r34_fuse.modules()])
    assert has_any_bn is False

    r50 = resnet50().eval().requires_grad_(False)
    param_before = sum(x.numel() for x in r50.parameters())
    inp = torch.rand(1, 3, 64, 64)
    out = r50(inp)
    r50_fuse = fuse_model(r50)
    param_after = sum(x.numel() for x in r50_fuse.parameters())
    out_fuse = r50_fuse(inp)
    assert torch.allclose(out, out_fuse, atol=1e-5)
    assert param_after < param_before
    has_any_bn = any([isinstance(m, nn.BatchNorm2d) for m in r50_fuse.modules()])
    assert has_any_bn is False


def test_model_fuse_grad_status():
    from torchvision.models import resnet34

    r34 = resnet34().eval()
    requires_grad = next(r34.parameters()).requires_grad
    r34_fuse = fuse_model(r34)
    assert next(r34_fuse.parameters()).requires_grad == requires_grad
    assert requires_grad is True

    r34 = r34.requires_grad_(False)
    requires_grad = next(r34.parameters()).requires_grad
    r34_fuse = fuse_model(r34)
    assert next(r34_fuse.parameters()).requires_grad == requires_grad
    assert requires_grad is False
