import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tools.utils.misc import initialize
from .activated_batch_norm import ABN
from .residual import conv3x3, conv1x1, DepthwiseSeparableConv, get_attn


class UnetDecoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, norm_layer=ABN, norm_act="relu", upsample=True, attn_type=None
    ):
        super(UnetDecoderBlock, self).__init__()

        conv1 = conv3x3(in_channels, out_channels)
        conv2 = conv3x3(out_channels, out_channels)
        abn1 = norm_layer(out_channels, activation=norm_act)
        abn2 = norm_layer(out_channels, activation=norm_act)
        self.block = nn.Sequential(conv1, abn1, conv2, abn2)
        self.attention = get_attn(attn_type)(out_channels, out_channels // 2)  # None == Identity
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear") if upsample else nn.Identity()

    def forward(self, x):
        x, skip = x
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        x = self.attention(x)
        return x


class TransposeX2(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=ABN, norm_act="relu"):
        super().__init__()
        conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        abn = norm_layer(out_channels, activation=norm_act)
        self.block = nn.Sequential(conv1, abn)

    def forward(self, x):
        return self.block(x)


class LinknetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=ABN, norm_act="relu", attn_type=None):
        super().__init__()
        middle_channels = in_channels // 4
        conv1 = conv1x1(in_channels, middle_channels)
        transpose = TransposeX2(middle_channels, middle_channels, norm_layer, norm_act)
        conv2 = conv1x1(middle_channels, out_channels)
        abn1 = norm_layer(middle_channels, activation=norm_act)
        abn2 = norm_layer(out_channels, activation=norm_act)
        self.block = nn.Sequential(conv1, abn1, transpose, conv2, abn2)
        self.attention = get_attn(attn_type)(out_channels, out_channels // 2)  # None == Identity

    def forward(self, x):
        x, skip = x
        x = self.block(x)
        x = self.attention(x)
        if skip is not None:
            x = x + skip
        return x


## DeepLab V3+ Modules


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_layer=ABN, norm_act="relu"):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(in_channels, out_channels),
            norm_layer(out_channels, activation=norm_act),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer=ABN, norm_act="relu"):
        super(ASPP, self).__init__()
        out_channels = 256
        norm_params = {"norm_layer": norm_layer, "norm_act": norm_act}
        self.conv0 = nn.Sequential(
            conv1x1(in_channels, out_channels), norm_layer(out_channels, activation=norm_act),
        )
        self.pool = ASPPPooling(in_channels, out_channels, **norm_params)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            norm_layer(out_channels, activation=norm_act),
            nn.Dropout(0.1),
        )

        rate1, rate2, rate3 = atrous_rates
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, dilation=rate1, **norm_params)
        self.conv2 = DepthwiseSeparableConv(in_channels, out_channels, dilation=rate2, **norm_params)
        self.conv3 = DepthwiseSeparableConv(in_channels, out_channels, dilation=rate3, **norm_params)

    def forward(self, x):
        res = [
            self.conv0(x),
            self.conv1(x),
            self.conv2(x),
            self.conv3(x),
            self.pool(x),
        ]
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabHead(nn.Module):
    def __init__(
        self,
        encoder_channels,
        num_classes,
        dilation_rates=[6, 12, 18],
        output_stride=16,
        drop_rate=0,
        norm_layer=ABN,
        norm_act="relu",
    ):
        PROJ_CONV_CHANNELS = 48
        OUT_CHANNELS = 256
        super().__init__()
        norm_params = {"norm_layer": norm_layer, "norm_act": norm_act}
        if output_stride == 8:
            dilation_rates = [i * 2 for i in dilation_rates]
        self.aspp = ASPP(encoder_channels[0], dilation_rates, norm_layer, norm_act)
        self.conv0 = nn.Sequential(
            conv3x3(OUT_CHANNELS, OUT_CHANNELS), norm_layer(OUT_CHANNELS, activation=norm_act),
        )
        self.proj_conv = nn.Sequential(
            conv1x1(encoder_channels[3], PROJ_CONV_CHANNELS),
            norm_layer(PROJ_CONV_CHANNELS, activation=norm_act),
        )

        self.sep_conv1 = DepthwiseSeparableConv(OUT_CHANNELS + PROJ_CONV_CHANNELS, 256, **norm_params)
        self.sep_conv2 = DepthwiseSeparableConv(OUT_CHANNELS, OUT_CHANNELS, **norm_params)
        self.dropout = nn.Dropout2d(drop_rate)
        self.final_conv = conv1x1(OUT_CHANNELS, num_classes)
        initialize(self)

    def forward(self, x):
        encoder_head = x[0]
        skip = x[3]  # downsampled 4x times

        x = self.aspp(encoder_head)
        x = self.conv0(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.proj_conv(skip)
        x = self.sep_conv1(torch.cat([skip, x], dim=1))
        x = self.sep_conv2(x)
        x = self.dropout(x)
        x = self.final_conv(x)
        return x


class Conv3x3NormAct(nn.Sequential):
    """Perform 3x3 conv norm act and optional 2x upsample"""

    def __init__(self, in_channels, out_channels, upsample=False, norm_layer=ABN, norm_act="relu"):
        super().__init__(
            conv3x3(in_channels, out_channels),
            norm_layer(out_channels, activation=norm_act),
            nn.Upsample(scale_factor=2) if upsample else nn.Identity(),
        )


class SegmentationUpsample(nn.Sequential):
    def __init__(self, in_channels, out_channels, n_upsamples=0, **bn_args):
        blocks = [Conv3x3NormAct(in_channels, out_channels, upsample=bool(n_upsamples), **bn_args)]
        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3NormAct(out_channels, out_channels, upsample=True, **bn_args))
        super().__init__(*blocks)
