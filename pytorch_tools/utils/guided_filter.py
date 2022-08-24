"""
PyTorch Guided Filter

Code is based on https://github.com/lisabug/guided-filter
Paper: Guided Image Filtering. Kaiming He et al
TODO: add `scale` option
"""

import torch


def box_filter_2d(tensor: torch.Tensor, r: int = 7):
    assert tensor.ndim == 4, "box_filter_2d expects tensor.shape==BxCxHxW"
    cs_x = tensor.cumsum(dim=2)
    blur_x = torch.cat(
        [
            cs_x[:, :, r : 2 * r + 1],
            cs_x[:, :, 2 * r + 1 :] - cs_x[:, :, : -2 * r - 1],
            cs_x[:, :, -1:] - cs_x[:, :, -2 * r - 1 : -r - 1],
        ],
        dim=2,
    )

    cs_y = blur_x.cumsum(dim=3)

    blur_y = torch.cat(
        [
            cs_y[:, :, :, r : 2 * r + 1],
            cs_y[:, :, :, 2 * r + 1 :] - cs_y[:, :, :, : -2 * r - 1],
            cs_y[:, :, :, -1:] - cs_y[:, :, :, -2 * r - 1 : -r - 1],
        ],
        dim=3,
    )
    return blur_y


def guided_filter_2d(guide: torch.Tensor, inp: torch.Tensor, r: int = 2, eps: float = 0.01):
    """Each channel is guided separately"""
    I, p = guide, inp
    # step 1
    N = box_filter_2d(torch.ones_like(I), r)
    meanI = box_filter_2d(I, r) / N
    meanp = box_filter_2d(p, r) / N
    corrI = box_filter_2d(I * I, r) / N
    corrIp = box_filter_2d(I * p, r) / N
    # step 2
    varI = corrI - meanI * meanI
    covIp = corrIp - meanI * meanp
    # step 3
    a = covIp / (varI + eps)
    b = meanp - a * meanI
    # step 4
    meana = box_filter_2d(a, r=r) / N
    meanb = box_filter_2d(b, r=r) / N
    # step 5
    q = meana * I + meanb
    return q


def multidim_guided_filter_2d(guide: torch.Tensor, inp: torch.Tensor, r: int = 2, eps: float = 0.01):
    """Input is guided on all channels of `guide`. Better for color images"""
    I, p = guide, inp
    B, C, H, W = guide.shape
    N = box_filter_2d(torch.ones_like(I)[:, 0:1], r)
    meanI = box_filter_2d(I, r) / N  # (H, W, C) -> B, C, H, W
    meanp = box_filter_2d(p, r) / N  # (H, W, 1) -> B, 1, H, W
    # BxCx1xHxW @ Bx1xCxHxW -> BxCxCxHxW -> B x C * C x H x W
    corrI_ = torch.einsum("bichw,bxihw->bcxhw", I[:, None], I[:, :, None]).view(B, -1, H, W)
    corrI_ = box_filter_2d(corrI_, r).div(N).view(B, C, C, H, W)
    meanI_ = torch.einsum("bichw,bxihw->bcxhw", meanI[:, None], meanI[:, :, None])
    corrI = corrI_ - meanI_
    # -> B x C x C x H x W -> B x H x W x C x C -> invert -> B x C x C x H x W
    left = torch.linalg.inv(corrI.permute(0, 3, 4, 1, 2) + torch.eye(C, C) * eps).permute(0, 3, 4, 1, 2)
    corrIp = box_filter_2d(I * p, r) / N
    covIp = corrIp - meanI * meanp
    # B x C x C x H x W @ B x C x 1 x H x W  -> B x C x 1 x H x W
    a = torch.einsum("bxchw,bcihw->bxihw", left, covIp[:, :, None])
    axmeanI = torch.einsum("bichw,bcjhw->bijhw", meanI[:, None], a).view(B, 1, H, W)
    b = meanp - axmeanI
    a = a.view(B, C, H, W)
    meana = box_filter_2d(a, r) / N
    meanb = box_filter_2d(b, r) / N
    # B x 1 x C x H x W @ B x C x 1 x H x W
    q = torch.einsum("bichw,bcjhw->bijhw", meana[:, None], I[:, :, None]).view(B, 1, H, W) + meanb
    return q
