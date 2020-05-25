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
    "efficientdet_b0",
    # "efficientdet_b1",
    # "efficientdet_b2",
    # "efficientdet_b3",
    # "efficientdet_b4",
    # "efficientdet_b5",
    # "efficientdet_b6",
]

# format "coco image class: PIL Image"
IMGS = {
    17: Image.open("tests/imgs/dog.jpg"),
}

INP = torch.ones(1, 3, 512, 512)


@torch.no_grad()
def _test_forward(model):
    return model(INP)


@pytest.mark.parametrize("arch", MODEL_NAMES)
def test_coco_pretrain(arch):
    m = pt_det.__dict__[arch](pretrained="coco").cuda()
    m.eval()
    # get size of the images used for pretraining
    inp_size = m.pretrained_settings["input_size"][-1]
    # get preprocessing fn according to pretrained settings
    preprocess_fn = get_preprocessing_fn(m.pretrained_settings)
    for im_cls, im in IMGS.items():
        im = np.array(im.resize((inp_size, inp_size)))
        im_t = tensor_from_rgb_image(preprocess_fn(im)).unsqueeze(0).float().cuda()
        boxes, scores, classes = m.predict(im_t)
        assert classes[0, 0] == im_cls  # check that most confident bbox is correct


@pytest.mark.parametrize("arch", MODEL_NAMES[:1])
def test_pretrain_custom_num_classes(arch):
    m = pt_det.__dict__[arch](pretrained="coco", num_classes=80).eval()
    _test_forward(m)
