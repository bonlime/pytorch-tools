##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## modified and simplified by @bonlime

import torch
from torch import nn

from pytorch_tools.modules import ABN
from pytorch_tools.modules.residual import conv1x1


class SpatialOCR_Gather(nn.Module):
    """Aggregate the context features according to the initial predicted probability distribution.
       Employ the soft-weighted method to aggregate the context.
    Input: 
        torch.Tensor (B x C_2 x H x W), torch.Tensor (B x C_1 x H x W)
    Returns:
        torch.Tensor (B x C_2 x C_1 x 1)
    """

    def forward(self, feats, probs):
        # C_1 is number of final classes. C_2 in number of features in `feats`
        probs = probs.view(probs.size(0), probs.size(1), -1)  # B x C_1 x H x W => B x C_1 x HW
        feats = feats.view(feats.size(0), feats.size(1), -1)  # B x C_2 x H x W => B x C_2 x HW
        feats = feats.permute(0, 2, 1)  # B x HW x C_2
        probs = probs.softmax(dim=2)  # B x C_1 x HW
        # B x C_1 x HW @ B x HW x C_2 => B x C_1 x C_2 => B x C_2 x C_1 => B x C_2 x C_1 x 1
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)
        return ocr_context


# class ObjectAttentionBlock2D(nn.Module):
"""
The basic implementation for object context block
Input:
    N X C X H X W
Parameters:
    in_channels       : the dimension of the input feature map
    key_channels      : the dimension after the key/query transform
Return:
    N X C X H X W
"""


class SpatialOCR(nn.Module):
    """
    Implementation of the object context block (OCR) module:
    We aggregate the global object representation to update the representation for each pixel.
    Args:
        in_channels (int): number of input channels
        key_channels (int): number of channels in the middle
        out_channels (int): number of output channels
        norm_layer (): Normalization layer to use
        norm_act (str): activation to use in `norm_layer`
    """

    def __init__(self, in_channels, key_channels, out_channels, norm_layer=ABN, norm_act="relu"):
        super().__init__()

        self.in_channels = in_channels
        self.key_channels = key_channels

        self.f_pixel = nn.Sequential(
            conv1x1(in_channels, key_channels, bias=True),
            norm_layer(key_channels, activation=norm_act),
            conv1x1(key_channels, key_channels, bias=True),
            norm_layer(key_channels, activation=norm_act),
        )
        self.f_object = nn.Sequential(
            conv1x1(in_channels, key_channels, bias=True),
            norm_layer(key_channels, activation=norm_act),
            conv1x1(key_channels, key_channels, bias=True),
            norm_layer(key_channels, activation=norm_act),
        )
        self.f_down = nn.Sequential(
            conv1x1(in_channels, key_channels, bias=True), norm_layer(key_channels, activation=norm_act),
        )
        self.f_up = nn.Sequential(
            conv1x1(key_channels, in_channels, bias=True), norm_layer(in_channels, activation=norm_act),
        )

        self.conv_bn = nn.Sequential(
            conv1x1(2 * in_channels, out_channels, bias=True), norm_layer(out_channels, activation=norm_act),
        )

    def forward(self, feats, proxy_feats):
        # feats B x 512 x H //4 x W//4
        # proxy feats B x 512 x num_classes (19) x 1
        batch_size = feats.size(0)
        # C = num classes; 512 = in_channels; 256 = key_channels
        # query.shape = B x 512 x H//4 x W//4 => B x 256 x H*W//16 => B x H*W//16 x 256
        query = self.f_pixel(feats).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        # key.shape = B x 512 x C x 1 => B x 256 x C x 1 => B x 256 x C
        key = self.f_object(proxy_feats).view(batch_size, self.key_channels, -1)
        # value.shape = B x 512 x C x 1 => B x 256 x C x 1 => B x C x 256
        value = self.f_down(proxy_feats).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        # sim_map.shape = B x H*W//16 x 256 @ B x 256 x C => B x H*W//16 x C
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -0.5) * sim_map
        sim_map = sim_map.softmax(dim=-1)

        # add bg context ...
        # context.shape = B x H*W//16 x C @ B x C x 256 => B x H*W//16 x 256
        context = torch.matmul(sim_map, value)
        # B x H*W//16 x 256 => B x 256 x H*W//16 => ... => B x 256 x H//4 x W//4
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *feats.size()[2:])
        context = self.f_up(context)
        # concat and project
        output = self.conv_bn(torch.cat([context, feats], 1))
        return output
