import sys
import time
import torch
import pytest
import argparse
import numpy as np
from apex import amp
import torchvision as tv
import torch.backends.cudnn as cudnn

import pytorch_tools as pt
from pytorch_tools import models
import pytorch_tools.segmentation_models as pt_sm
import effdet
import timm


class AverageMeter:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


@pytest.mark.skip("Not meant for pytest")
def test_model(model, forward_only=False):
    optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)
    f_times = []
    fb_times = []
    with cudnn.flags(enabled=True, benchmark=True), torch.set_grad_enabled(not forward_only):
        start = torch.cuda.Event(enable_timing=True)
        f_end = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        def run_once():
            start.record()
            output = model(INP, TARGET) if TARGET is not None else model(INP)
            if isinstance(output, (list, tuple)):
                output = output[0]
            f_end.record()
            if forward_only:
                torch.cuda.synchronize()
                return start.elapsed_time(f_end), start.elapsed_time(f_end)
            loss = output.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(f_end), start.elapsed_time(end)

        # benchmark runs. usually much slower than following ones
        for _ in range(2):
            run_once()
        # during cudnn benchmarking a lot of memory is used. we need to reset
        # in order to get max mem alloc by the fastest algorithm
        torch.cuda.reset_max_memory_allocated(0)
        for _ in range(N_RUNS):
            f_meter = AverageMeter()
            fb_meter = AverageMeter()
            for _ in range(RUN_ITERS):
                f_t, fb_t = run_once()
                f_meter.update(f_t)
                fb_meter.update(fb_t)
            f_times.append(f_meter.avg)
            fb_times.append(fb_meter.avg)
        f_times = np.array(f_times)
        fb_times = np.array(fb_times)
    print(
        "Mean of {} runs {} iters each BS={}, SZ={}:\n\t {:.2f}+-{:.2f} msecs Forward. {:.2f}+-{:.2f} msecs Backward. Max memory: {:.2f}Mb. {:.2f} imgs/sec".format(
            N_RUNS,
            RUN_ITERS,
            BS,
            SZ,
            f_times.mean(),
            f_times.std(),
            (fb_times - f_times).mean(),
            (fb_times - f_times).std(),
            torch.cuda.max_memory_allocated(0) / 2 ** 20,
            BS * 1000 / fb_times.mean(),
        )
    )
    del optimizer
    del model


import torch.nn as nn


class DetectionTrainWrapper(nn.Module):
    def __init__(self, modell, size):
        super().__init__()
        self.loss = None
        self.model = modell
        anchors = pt.utils.box.generate_anchors_boxes(size)[0]
        # self.loss = pt.losses.DetectionLoss(anchors)
        self.loss = torch.jit.script(pt.losses.DetectionLoss(anchors))

    def forward(self, inp, target):
        # cls_out, box_out = self.model(inp)
        # return box_out
        loss = self.loss(self.model(inp), target)
        return loss
        # return box_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model Benchmarking")
    parser.add_argument(
        "--forward", "-f", action="store_true", help="Flag to only run forward. Disables grads"
    )
    parser.add_argument("--amp", action="store_true", help="Measure speed using apex mixed precision")
    parser.add_argument(
        "--torch_amp", action="store_true", help="Measure speed using torch native mixed precision"
    )
    parser.add_argument(
        "--bs", default=64, type=int, help="BS for benchmarking",
    )
    parser.add_argument(
        "--sz", default=224, type=int, help="Size of images for benchmarking",
    )
    args = parser.parse_args()
    # all models are first init to cpu memory to find errors earlier
    models_dict = {
        # "EffDet0 My": pt.detection_models.efficientdet_d0(match_tf_same_padding=False, pretrained=None),
        # "EffDet0 My Wrapped": DetectionTrainWrapper(
        #     modell=pt.detection_models.efficientdet_d0(match_tf_same_padding=False, pretrained=None),
        #     size=args.sz,
        # )
        "EffB0": pt.models.efficientnet_b0(),
        # "ResNet50": pt.models.resnet50(),
    }
    segm_models_dict = {
        # "Unet Resnet34": pt_sm.Unet("resnet34"),
    }

    print("Initialized models")
    BS = args.bs
    SZ = args.sz
    N_RUNS = 5
    RUN_ITERS = 10
    INP = torch.ones((BS, 3, SZ, SZ), requires_grad=not args.forward).cuda(0)
    TARGET = None
    # TARGET = torch.randint(0, 50, (BS, 20, 5)).cuda().float()

    for name, model in models_dict.items():
        print(f"{name} {count_parameters(model) / 1e6:.2f}M params")
        model = model.cuda(0)
        if args.amp:
            model = amp.initialize(model, verbosity=0, opt_level="O1")
            INP = INP.half()
        if args.torch_amp:
            model.forward = torch.cuda.amp.autocast()(model.forward)
        if args.forward:
            model.eval()

        # with torch.cuda.amp.autocast(enabled=args.torch_amp):
        test_model(model, forward_only=args.forward)

    # now test segmentation models
    INP = torch.ones((BS, 3, SZ, SZ), requires_grad=True).cuda(0)

    for name, model in segm_models_dict.items():
        enc_params = count_parameters(model.encoder) / 1e6
        total_params = count_parameters(model) / 1e6
        print(
            f"{name}. Encoder {enc_params:.2f}M. Decoder {total_params - enc_params:.2f}M. Total {total_params:.2f}M params"
        )
        model = model.cuda(0)
        if args.amp:
            model = amp.initialize(model, verbosity=0, opt_level="O1")
            INP = INP.half()
        if args.forward:
            model.eval()
        test_model(model, forward_only=args.forward)
