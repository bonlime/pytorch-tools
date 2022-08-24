import torch
import pytest

from pytorch_tools.fit_wrapper.utils import AverageMeter


def test_average_meter():
    values = torch.randn(100)
    meter = AverageMeter(avg_mom=0.9)
    my_avg_smooth = None
    for val in values:
        meter(val)
        if my_avg_smooth is None:
            my_avg_smooth = val
        else:
            my_avg_smooth = my_avg_smooth * 0.9 + val * 0.1
    assert torch.allclose(meter.avg, values.mean())
    assert torch.allclose(meter.avg_smooth, my_avg_smooth)
