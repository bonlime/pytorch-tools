"""
Implementation of tile-based inference allowing to predict huge images that does not fit into GPU memory entirely
in a sliding-window fashion and merging prediction mask back to full-resolution.

Hacked together by @zakajd & @bonlime

Reference:
    https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/inference/tiles.py
    https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
    https://github.com/victorca25/iNNfer
"""
import itertools
from typing import Callable, Optional, List

import torch
from torch.nn import functional as F


def compute_pyramid_patch_weight_loss(width: int, height: int, depth: Optional[int] = None) -> torch.Tensor:
    """Compute a weight matrix that assigns bigger weight on pixels in center and
    less weight to pixels on image boundary. This weight matrix then used for merging
    individual tile predictions and helps dealing with prediction artifacts on tile boundaries.

    Args:
        width: Tile width
        height: Tile height
        depth: Tile depth (used in 3d case).

    Returns:
        Weight matrix of shape (1, width, height, [depth])

    """
    Dcx = torch.linspace(-width * 0.5 + 0.5, width * 0.5 - 0.5, width).square()
    Dcy = torch.linspace(-height * 0.5 + 0.5, height * 0.5 - 0.5, height).square()
    Dcz = 0
    if depth is not None:
        Dcz = torch.linspace(-depth * 0.5 + 0.5, depth * 0.5 - 0.5, depth).square()
        Dcz = Dcz[None, None, :]

    # eucl distance to center
    Dc = (Dcx[:, None, None] + Dcy[None, :, None] + Dcz).sqrt()

    De_x = torch.linspace(0.5, width - 0.5, width).abs()
    De_x = torch.minimum(De_x, De_x.flip(dims=(0,)))

    De_y = torch.linspace(0.5, height - 0.5, height).abs()
    De_y = torch.minimum(De_y, De_y.flip(dims=(0,)))

    De_z = torch.tensor(float("inf"))
    if depth is not None:
        De_z = torch.linspace(0.5, depth - 0.5, depth).abs()
        De_z = torch.minimum(De_z, De_z.flip(dims=(0,)))
        De_z = De_z[None, None, :]

    # distance to the closest border
    De = torch.minimum(De_x[:, None, None], De_y[None, :, None])
    De = torch.minimum(De, De_z)

    W = De / (Dc + De)
    # normalize weights
    depth_dim = 1 if depth is None else depth
    W = W / W.sum() * (width * height * depth_dim)

    # remove extra axis if not needed
    W = W[..., 0] if depth is None else W
    return W.unsqueeze(dim=0)


class TileInference:
    """Wrapper for models that implements tile-based inference allowing to predict huge images
    that does not fit into GPU memory entirely in a sliding-window fashion.

    Args:
        tile_size: int
        overlap: By how many pixels tiles are overlapped with each other
        model: Any callable used for processing single input patch.
        border_pad: padding from borders. if not given would be same as `overlap`
        scale: ratio of output size to input size
        fusion: One of {'mean', 'pyramid'}. Defines how overlapped patches are weighted.
    """

    def __init__(
        self,
        tile_size: List[int],
        overlap: List[int],
        model: Callable,
        border_pad: Optional[List[int]] = None,
        scale: float = 1,
        fusion: str = "mean",
    ):
        # Check input values
        if isinstance(tile_size, int):
            raise ValueError("TileInference expects `tile_size` as `List[int]` not `int` ")

        self.tile_size = torch.tensor(tile_size)
        self.overlap = torch.tensor(overlap)
        self.border_pad = torch.tensor(overlap if border_pad is None else border_pad)
        if not all(self.tile_size >= 2 * self.overlap):
            raise ValueError("Overlap can't be larger than tile size")

        self.tile_step = self.tile_size - self.overlap

        self.scale = scale
        self.model = model
        weights = {"mean": self._mean, "pyramid": self._pyramid}
        self.weight = weights[fusion]((self.tile_size * scale).long())

    def _mean(self, tile_size: List[int]):
        return torch.ones((1, *tile_size))

    def _pyramid(self, tile_size: List[int]):
        return compute_pyramid_patch_weight_loss(*tile_size)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Args:
            image: ND tensor with shape (C, H, W, [D])

        """

        # Number of tiles in all directions.
        shapes_tensor = torch.tensor(image.shape[1:], dtype=torch.long)
        n_tiles = (shapes_tensor - self.overlap + 2 * self.border_pad).div(self.tile_step).ceil().long()

        extra_pads = n_tiles * self.tile_step - shapes_tensor + self.overlap
        pad_l, pad_r = extra_pads.div(2, rounding_mode="floor"), extra_pads - extra_pads.div(2, rounding_mode="floor")
        pads = torch.stack([pad_r, pad_l], dim=1).flatten().flip(dims=(0,))

        # Make image divisible by `tile_size` and add border pixels if necessary
        # Other border types produce artifacts
        padded_image = F.pad(image, pads.tolist(), mode="reflect")
        tile_generator = self.iter_split(padded_image)

        # Empty output tensor
        out_shape = (torch.tensor(padded_image.shape[1:]) * self.scale).long()
        out_shape = [image.size(0), *out_shape.tolist()]

        # Empty output tensor
        output = image.new_zeros(out_shape)
        # Used to save weight mask used for blending
        norm_mask = image.new_zeros(out_shape)

        # Move weights to correct device and dtype
        w = self.weight.to(device=image.device, dtype=image.dtype)

        for tile, out_slice in tile_generator:
            # Process. add and remove batch dimension manually
            output_tile = self.model(tile[None])[0]
            channel_slice = slice(0, output_tile.size(0))
            output[[channel_slice, *out_slice]] += output_tile * w
            norm_mask[[channel_slice, *out_slice]] += w

        # Normalize by mask to weighten overlapped patches.
        output = torch.div(output, norm_mask)

        # Crop added margins if we have any
        if (pad_l + pad_r).sum() > 0:
            unpad_slices = [
                slice(int(l * self.scale), -int(r * self.scale)) for l, r in zip(pad_l.tolist(), pad_r.tolist())
            ]
            output = output[[channel_slice, *unpad_slices]]
        return output

    def iter_split(self, image: torch.Tensor):
        """
        Split image into partially overlapping patches.
        Pads the image twice:
            - first to have a size multiple of the patch size
            - then to have equal padding at the borders.
        """

        tile_size = self.tile_size
        tile_step = self.tile_step

        ranges = [range(0, s - sz + 1, st) for (s, sz, st) in zip(image.shape[1:], tile_size, tile_step)]
        # coordinates of top left pixel of the tile
        starts = list(itertools.product(*ranges))
        slices = [[slice(i, i + sz) for i, sz in zip(start, tile_size)] for start in starts]
        out_slices = [
            [slice(int(i * self.scale), int((i + sz) * self.scale)) for i, sz in zip(start, tile_size)]
            for start in starts
        ]
        channel_slice = slice(0, image.size(0))
        # Loop over all slices
        for in_slice, out_slice in zip(slices, out_slices):
            tile = image[[channel_slice, *in_slice]]
            yield tile, out_slice
