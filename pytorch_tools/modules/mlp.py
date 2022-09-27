"""Blocks inspired by MLP papers. Used as alternative to transformers"""

import torch
import torch.nn as nn
from pytorch_tools.modules.residual import DropConnect


class SpatialToSequence(nn.Module):
    """Turns BS x C x H x W -> BS x H * W x C"""

    def forward(self, x):
        return x.view(*x.shape[:2], -1).transpose(-1, -2)


class SequenceToSpatial(nn.Module):
    """Turns BS x H * W x C -> BS x C x H x W"""

    def forward(self, x):
        spatial_size = int(x.size(1) ** 0.5)
        return x.transpose(-1, -2).view(x.size(0), x.size(2), spatial_size, spatial_size)


class Attention(nn.Module):
    """Simple single-head attention
    Args:
        dim_in (int): input number of dimensions
        dim_inner (int): number of channels in attention
        dim_out (int): output number of channels
    """

    def __init__(self, dim_in, dim_inner, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim_in
        self.scale = dim_inner**-0.5
        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        return self.to_out(out)


# Every design choice for MLP is questionable
# PayAttention to MLP: conv1x1 DW conv1x1
# -> LN -> CHN -> Chunk -> LN -> SPATIAL * Residual -> CHN + Residual ->
# \______\ ________ \ ____________ + ____ / ________________/
#         \          \___________ / _____/
#          \____ Self-Attn ______/

# While in ResMLP they use: DW conv1x1 conv1x1
# -> Aff -> SPATIAL -> Aff + Residual -> Aff -> CHN -> Act -> CHN -> Aff+ Residual ->
#  \________________________/          \_________________________________/
# When LN is replaced with Aff in first alpha=1, beta=0, in second alpha is a small value

# In MLP-Mixer design is close to ResMLP: DW DW 1x1 1x1
# -> LN -> SPATIAL -> Act -> SPATIAL + Residual -> LN -> CHN -> Act -> CHN + Residual ->
#  \__________________________________/          \__________________________/


class SpatialGatingBlock(nn.Module):
    """Residual Block with Spatial Gating
    Args:
        dim_in (int): input dimension
        seq_len (int): sequence length
        attn_dim (Optional[int]): if given, adds single-head self-attention
        mlp_ratio (int): how much to increase number of channels inside block
            intuitively this is the same as bottleneck_ratio in ResNet bottleneck
        dim_out (Optional[int]): output number of dimensions. if not given would be same as input
        norm_layer (nn.Module): normalization layer to use
        act_layer: activation after first projection
        spatial_act_layer: activation after spatial gating unit but before summation with residual
            not present in paper so is Identity by default. need additional investigation
        drop_path (float): probability to drop some

    Ref: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """

    def __init__(
        self,
        dim_in,
        seq_len,
        attn_dim=None,
        mlp_ratio=2,
        dim_out=None,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        spatial_act_layer=nn.Identity,
        keep_prob=1,
        init_eps=1e-2,
    ):
        super().__init__()
        dim_inner = int(dim_in * mlp_ratio)
        dim_out = dim_out or dim_in
        self.pre_norm = norm_layer(dim_in)
        self.proj_in = nn.Linear(dim_in, dim_inner)
        self.act_in = act_layer()
        self.sgu_norm = norm_layer(dim_inner // 2)
        # Conv1d is faster than Transpose + Linear + Transpose
        self.proj_spatial = nn.Conv1d(seq_len, seq_len, 1)
        self.act_spatial = spatial_act_layer()
        self.proj_out = nn.Linear(dim_inner // 2, dim_out)

        self.attn = None
        if attn_dim is not None:
            self.attn = Attention(dim_in=dim_in, dim_inner=attn_dim, dim_out=dim_inner // 2)

        self.drop_path = DropConnect(keep_prob) if keep_prob < 1 else nn.Identity()

        # careful init for spatial projection
        init_eps /= seq_len
        nn.init.uniform_(self.proj_spatial.weight, -init_eps, init_eps)
        nn.init.constant_(self.proj_spatial.bias, 1.0)

    def forward(self, x):
        res = x
        x = self.pre_norm(x)
        attn_res = self.attn(x) if self.attn is not None else 0
        x = self.proj_in(x)
        # spatial gating unit
        u, v = x.chunk(2, dim=-1)
        v = self.sgu_norm(v)
        v = self.proj_spatial(v)

        v += attn_res
        v = self.act_spatial(v)
        x = self.proj_out(u * v)
        x += self.drop_path(res)
        return x
