import numpy as np
import functools


def preprocess_input(x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs):
    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x


def get_preprocessing_fn(pretrain_settings):
    input_space = pretrain_settings.get("input_space")
    input_range = pretrain_settings.get("input_range")
    mean = pretrain_settings.get("mean")
    std = pretrain_settings.get("std")

    return functools.partial(
        preprocess_input, mean=mean, std=std, input_space=input_space, input_range=input_range
    )
