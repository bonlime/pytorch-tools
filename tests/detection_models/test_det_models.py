import torch
import pytest
import numpy as np
from PIL import Image
from pytorch_tools.utils.preprocessing import get_preprocessing_fn
from pytorch_tools.utils.visualization import tensor_from_rgb_image

import pytorch_tools as pt
import pytorch_tools.detection_models as pt_det


# all weights were tested on 05.2020. for now only leave one model for faster tests
MODEL_NAMES = [
    "efficientdet_d0",
    "retinanet_r50_fpn",
    # "efficientdet_d1",
    # "efficientdet_d2",
    # "efficientdet_d3",
    # "efficientdet_d4",
    # "efficientdet_d5",
    # "efficientdet_d6",
    # "retinanet_r101_fpn",
]

# format "coco image class: PIL Image"
IMGS = {
    17: Image.open("tests/imgs/dog.jpg"),
}

INP = torch.ones(1, 3, 512, 512).cuda()


@torch.no_grad()
def _test_forward(model):
    return model(INP)


@pytest.mark.parametrize("arch", MODEL_NAMES)
def test_coco_pretrain(arch):
    # want TF same padding for better results
    kwargs = {}
    if "eff" in arch:
        kwargs["match_tf_same_padding"] = True
    m = pt_det.__dict__[arch](pretrained="coco", **kwargs).cuda()
    m.eval()
    # get size of the images used for pretraining
    inp_size = m.pretrained_settings["input_size"][-1]
    # get preprocessing fn according to pretrained settings
    preprocess_fn = get_preprocessing_fn(m.pretrained_settings)
    for im_cls, im in IMGS.items():
        im = np.array(im.resize((inp_size, inp_size)))
        im_t = tensor_from_rgb_image(preprocess_fn(im)).unsqueeze(0).float().cuda()
        boxes_scores_classes = m.predict(im_t)
        # check that most confident bbox is close to correct class. The reason for such strange test is
        # because in different models class mappings are shifted by +- 1
        assert (boxes_scores_classes[0, 0, 5] - im_cls) < 2


@pytest.mark.parametrize("arch", MODEL_NAMES[:2])
def test_pretrain_custom_num_classes(arch):
    m = pt_det.__dict__[arch](pretrained="coco", num_classes=80).eval().cuda()
    _test_forward(m)


@pytest.mark.parametrize("arch", MODEL_NAMES[:2])
def test_encoder_frozenabn(arch):
    m = pt_det.__dict__[arch](encoder_norm_layer="frozenabn").eval().cuda()
    _test_forward(m)
