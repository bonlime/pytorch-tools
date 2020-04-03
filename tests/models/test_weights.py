## test that imagenet pretrained weights are valid and able to classify correctly the cat and dog

import numpy as np
from PIL import Image
import pytest

from pytorch_tools.utils.preprocessing import get_preprocessing_fn
from pytorch_tools.utils.visualization import tensor_from_rgb_image
import pytorch_tools.models as models

MODEL_NAMES = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

# tests are made to be run from root project directory
# format "imagenet_image_class: PIL Image"
IMGS = {
    560: Image.open("tests/models/imgs/helmet.jpeg"),
    207: Image.open("tests/models/imgs/dog.jpg"),
}

# временная заглушка. TODO: убрать
MODEL_NAMES = [
    "resnet18",
    "resnet34",
    "densenet121",
    "densenet169",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "tresnetm",
    "tresnetl",
    "tresnetxl",
]


@pytest.mark.parametrize("arch", MODEL_NAMES)
def test_imagenet_pretrain(arch):
    m = models.__dict__[arch](pretrained="imagenet")
    m.eval()
    # get size of the images used for pretraining
    inp_size = m.pretrained_settings["input_size"][-1]
    # get preprocessing fn according to pretrained settings
    preprocess_fn = get_preprocessing_fn(m.pretrained_settings)
    for im_cls, im in IMGS.items():
        im = np.array(im.resize((inp_size, inp_size)))
        im = tensor_from_rgb_image(preprocess_fn(im))
        # add batch dim
        im = im.view(1, *im.shape).float()
        pred_cls = m(im).argmax()
        assert pred_cls == im_cls
