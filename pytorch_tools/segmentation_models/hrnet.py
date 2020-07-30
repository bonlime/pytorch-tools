import logging
from copy import deepcopy
from functools import wraps

import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

from pytorch_tools.modules.residual import conv1x1
from pytorch_tools.modules.residual import conv3x3
from pytorch_tools.modules import ABN
from pytorch_tools.modules import InPlaceABN
from pytorch_tools.modules import bn_from_name
from pytorch_tools.modules.spatial_ocr_block import SpatialOCR
from pytorch_tools.modules.spatial_ocr_block import SpatialOCR_Gather

from pytorch_tools.utils.misc import initialize
from pytorch_tools.utils.misc import repeat_channels

from .encoders import get_encoder


def patch_bn_mom(module):
    """changes default bn momentum"""
    if isinstance(module, ABN) or isinstance(module, InPlaceABN):
        module.momentum = 0.01
    for m in module.children():
        patch_bn_mom(m)


def patch_inplace_abn(module):
    """changes weight from InplaceABN to be compatible with usual ABN"""
    if isinstance(module, ABN):
        module.weight = nn.Parameter(module.weight.abs() + 1e-5)
    for m in module.children():
        patch_inplace_abn(m)


class HRNet(nn.Module):
    """HRNet model for image segmentation
    NOTE: for this model input size should be divisible by 32!

    Args:
        encoder_name (str): name of classification model used as feature extractor to build segmentation model.
        encoder_weights (str): one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        num_classes (int): a number of classes for output (output shape - ``(batch, classes, h, w)``).
        pretrained (Union[str, None]): hrnet_w48 and hrnet_w48+OCR have pretrained weights. init models using functions rather than
            from class to load them. Available options are: None, `cityscape` 
        last_upsample (bool): Flag to enable upsampling predictions to the original image size. If set to `False` prediction
            would be 4x times smaller than input image. Default True.
        drop_rate (float): Probability of spatial dropout on last feature map
        OCR (bool): Flag to add object context block to the model. See `Ref` section for paper. 
        norm_layer (str): Normalization layer to use. One of 'abn', 'inplaceabn'. The inplace version lowers memory
            footprint. But increases backward time. Defaults to 'abn'.
        norm_act (str): Activation for normalizion layer. 'inplaceabn' doesn't support `ReLU` activation.
    
    Ref:
        Deep High-Resolution Representation Learning for Visual Recognition: https://arxiv.org/abs/1908.07919
        Object-Contextual Representations for Semantic Segmentation: https://arxiv.org/abs/1909.11065
    """

    def __init__(
        self,
        encoder_name="hrnet_w18",
        encoder_weights="imagenet",
        pretrained=None,  # not used
        num_classes=1,
        last_upsample=True,
        OCR=False,
        drop_rate=0,
        norm_layer="inplace_abn",  # use memory efficient by default
        norm_act="leaky_relu",
        **encoder_params,
    ):

        super().__init__()
        self.encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights,
            norm_layer=norm_layer,
            norm_act=norm_act,
            **encoder_params,
        )
        norm_layer = bn_from_name(norm_layer)
        final_channels = sum(self.encoder.out_shapes[:4])

        self.OCR = OCR
        if OCR:
            self.conv3x3 = nn.Sequential(
                conv3x3(final_channels, 512, bias=True), norm_layer(512, activation=norm_act),
            )
            self.ocr_gather_head = SpatialOCR_Gather()
            self.ocr_distri_head = SpatialOCR(
                in_channels=512, key_channels=256, out_channels=512, norm_layer=norm_layer, norm_act=norm_act
            )
            self.head = conv1x1(512, num_classes, bias=True)
            self.aux_head = nn.Sequential(  # in OCR first conv is 3x3
                conv3x3(final_channels, final_channels, bias=True),
                norm_layer(final_channels, activation=norm_act),
                conv1x1(final_channels, num_classes, bias=True),
            )
        else:
            self.head = nn.Sequential(
                conv1x1(final_channels, final_channels, bias=True),
                norm_layer(final_channels, activation=norm_act),
                conv1x1(final_channels, num_classes, bias=True),
            )

        up_kwargs = dict(mode="bilinear", align_corners=True)
        self.up_x2 = nn.Upsample(scale_factor=2, **up_kwargs)
        self.up_x4 = nn.Upsample(scale_factor=4, **up_kwargs)
        self.up_x8 = nn.Upsample(scale_factor=8, **up_kwargs)
        self.last_upsample = nn.Upsample(scale_factor=4, **up_kwargs) if last_upsample else nn.Identity()
        self.dropout = nn.Dropout2d(drop_rate)  # can't use inplace. it would raise a backprop error
        self.name = f"segm-{encoder_name}"
        # use lower momemntum
        patch_bn_mom(self)
        self._init_weights()

    def forward(self, x):
        """Sequentially pass `x` trough model`s `encoder` and `head` (return logits!)"""
        # retuns 5 feature maps, but 5th one is the same as 4th so drop it
        x = self.encoder(x)
        x = [x[3], self.up_x2(x[2]), self.up_x4(x[1]), self.up_x8(x[0])]
        x = torch.cat(x, 1)

        if self.OCR:
            out_aux = self.aux_head(x)
            x = self.conv3x3(x)
            context = self.ocr_gather_head(x, out_aux)
            x = self.ocr_distri_head(x, context)
            x = self.dropout(x)
            x = self.head(x)
            x = self.last_upsample(x)
            out_aux = self.last_upsample(out_aux)
            return out_aux, x
        else:
            x = self.dropout(x)
            x = self.head(x)
            x = self.last_upsample(x)
            return x

    def _init_weights(self):
        # it works better if we only init last bias not whole decoder part
        # set last layer bias for better convergence with sigmoid loss
        # -4.59 = -np.log((1 - 0.01) / 0.01)
        if self.OCR:
            nn.init.constant_(self.head.bias, -4.59)
            nn.init.constant_(self.aux_head[2].bias, -4.59)
        else:
            nn.init.constant_(self.head[2].bias, -4.59)


# fmt: off
SETTINGS = {
    "input_size": [3, 512, 1024],
    "input_range": [0, 1],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "num_classes": 19,
}
CFGS = {
    "hrnet_w48": {
        "default": {
            "params": {"encoder_name": "hrnet_w48", },
            "input_space": "RGB",
            "norm_layer": "inplace_abn",
            **SETTINGS,
        },
        "cityscape": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.3/hrnet_w48_cityscapes_cls19_1024x2048_ohem_trainvalset_remapped.pth"},
    },
    "hrnet_w48_ocr": {
        "default": {
            "params": {"encoder_name": "hrnet_w48", "OCR": True},
            "input_space": "BGR", # this weights were trained using cv2.imread 
            "norm_layer": "inplace_abn",
            **SETTINGS,
        },
        "cityscape": {"url": "https://github.com/bonlime/pytorch-tools/releases/download/v0.1.3/hrnet_w48_ocr_1_latest_remapped.pth"},
    },
}
# fmt: on
def _hrnet(arch, pretrained=None, **kwargs):
    cfgs = deepcopy(CFGS)
    cfg_settings = cfgs[arch]["default"]
    cfg_params = cfg_settings.pop("params")
    if pretrained:
        pretrained_settings = cfgs[arch][pretrained]
        pretrained_params = pretrained_settings.pop("params", {})
        cfg_settings.update(pretrained_settings)
        cfg_params.update(pretrained_params)
    common_args = set(cfg_params.keys()).intersection(set(kwargs.keys()))
    if common_args:
        logging.warning(
            f"Args {common_args} are going to be overwritten by default params for {pretrained} weights"
        )
    kwargs.update(cfg_params)
    model = HRNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(cfgs[arch][pretrained]["url"])
        kwargs_cls = kwargs.get("num_classes", None)
        if kwargs_cls and kwargs_cls != cfg_settings["num_classes"]:
            logging.warning(
                "Using model pretrained for {} classes with {} classes. Last layer is initialized randomly".format(
                    cfg_settings["num_classes"], kwargs_cls
                )
            )
            # if there is last_linear in state_dict, it's going to be overwritten
            if cfg_params.get("OCR", False):
                state_dict["aux_head.2.weight"] = model.state_dict()["aux_head.2.weight"]
                state_dict["aux_head.2.bias"] = model.state_dict()["aux_head.2.bias"]
                state_dict["head.weight"] = model.state_dict()["head.weight"]
                state_dict["head.bias"] = model.state_dict()["head.bias"]
            else:
                state_dict["head.2.weight"] = model.state_dict()["head.2.weight"]
                state_dict["head.2.bias"] = model.state_dict()["head.2.bias"]
        # support custom number of input channels
        if kwargs.get("in_channels", 3) != 3:
            old_weights = state_dict.get("encoder.conv1.weight")
            state_dict["encoder.conv1.weight"] = repeat_channels(old_weights, kwargs["in_channels"])
        model.load_state_dict(state_dict)
        # models were trained using inplaceabn. need to adjust for it. it works without
        # this patch but results are slightly worse
        patch_inplace_abn(model)
    setattr(model, "pretrained_settings", cfg_settings)
    return model


@wraps(HRNet)
def hrnet_w48(pretrained="cityscape", **kwargs):
    # set number of classes to 19 if not stated explicitly
    kwargs["num_classes"] = kwargs.get("num_classes", 19)
    return _hrnet("hrnet_w48", pretrained=pretrained, **kwargs)


@wraps(HRNet)
def hrnet_w48_ocr(pretrained="cityscape", **kwargs):
    kwargs["num_classes"] = kwargs.get("num_classes", 19)
    return _hrnet("hrnet_w48_ocr", pretrained=pretrained, **kwargs)
