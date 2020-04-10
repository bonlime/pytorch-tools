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
            output = model(INP)
            f_end.record()
            if forward_only:
                torch.cuda.synchronize()
                return start.elapsed_time(f_end), start.elapsed_time(f_end)
            loss = criterion(output, TARGET)
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
        "Mean of {} runs {} iters each BS={}:\n\t {:.2f}+-{:.2f} msecs Forward. {:.2f}+-{:.2f} msecs Backward. Max memory: {:.2f}Mb. {:.2f} imgs/sec".format(
            N_RUNS,
            RUN_ITERS,
            BS,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model Benchmarking")
    parser.add_argument(
        "--forward", "-f", action="store_true", help="Flag to only run forward. Disables grads"
    )
    parser.add_argument(
        "--amp", action="store_true", help="Measure speed using apex mixed precision"
    )
    parser.add_argument(
        "--bs", default=64, type=int, help="BS for classification",
    )
    args = parser.parse_args()
    # all models are first init to cpu memory to find errors earlier
    models_dict = {
        # 'VGG16 ABN' : models.vgg16_bn(norm_layer='abn'),
        # 'VGG 16 InplaceABN:': models.vgg16_bn(norm_layer='inplaceabn'),
        # 'Resnet50 OS 32:' : models.resnet50(deep_stem=True, antialias=True, output_stride=32), #norm_layer='abn',
        # 'Resnet50 OS 16:': models.resnet50(deep_stem=True, antialias=True, output_stride=16),
        # "Resnet50:": models.resnet50(),
        # "Se-Resnet50:": models.resnet50(use_se=True),
        # "Resnet50 D:": models.resnet50(deep_stem=True, antialias=True),  # norm_layer='inplaceabn', ),
        # "Se-Resnet50 D": models.resnet50(deep_stem=True, antialias=True, use_se=True),
        # 'Resnet50 OS 8 InPlace:': models.resnet50(deep_stem=True, antialias=True, output_stride=8, norm_layer='inplaceabn', norm_act='leaky_relu'),
        'Resnet50': models.resnet50(),
        'Resnet50 tv': tv.models.resnet50(),
        # 'Resnet50 D Antialias:': models.resnet50(antialias=True),
        # 'Resnet50 Inplace': models.resnet50(norm_act="leaky_relu", norm_layer="inplaceabn")
        # 'Resnet50 ABN:' : models.se_resnet50(norm_layer='abn'),
        # 'Resnet50 InPlaceABN:': models.se_resnet50(norm_layer='inplaceabn', norm_act='leaky_relu'),
        # 'Densenet121 MemEff' : models.densenet121(memory_efficient=True),
        # 'Densenet121 NotMemEff' : models.densenet121(memory_efficient=False),
        # "Densenet121 ABN": models.densenet121(norm_layer="abn", norm_act="leaky_relu"),
        # "Densenet121 InPlaceABN": models.densenet121(norm_layer="inplaceabn", norm_act="leaky_relu"),
        # 'SE Resnext50_32x4 ABN:' : models.se_resnext50_32x4d(norm_layer='abn'),
        # 'SE Resnext50_32x4 InplaceABN:' : models.se_resnext50_32x4d(norm_layer='inplaceabn')
    }
    segm_models_dict = {
        # "Unet Resnet34": pt_sm.Unet("resnet34"),
        # "Unet SEResnet50": pt_sm.Unet("se_resnet50"),
        # "Linknet SEResnet50": pt_sm.Linknet("se_resnet50"),
        # "Segm BiFPN SEResnet50": pt_sm.SegmentationBiFPN("se_resnet50"),
        # "Segm FPN SEResnet50": pt_sm.SegmentationFPN("se_resnet50"),
        # "Deeplab SEResnet50": pt_sm.DeepLabV3("se_resnet50"),
        # "Unet SEResnet50": pt_sm.Unet("se_resnet50"),
    }

    print("Initialized models")
    BS = args.bs
    N_RUNS = 10
    RUN_ITERS = 10
    INP = torch.ones((BS, 3, 224, 224), requires_grad=not args.forward).cuda(0)
    TARGET = torch.ones(BS).long().cuda(0)
    criterion = torch.nn.CrossEntropyLoss().cuda(0)
    for name, model in models_dict.items():
        print(f"{name} {count_parameters(model) / 1e6:.2f}M params")
        model = model.cuda(0)
        if args.amp:
            model = amp.initialize(model, verbosity=0, opt_level="O1")
            INP = INP.half()
        if args.forward:
            model.eval()
        test_model(model, forward_only=args.forward)

    # now test segmentation models
    BS = 16
    INP = torch.ones((BS, 3, 224, 224), requires_grad=True).cuda(0)
    TARGET = torch.ones((BS, 1, 224, 224)).cuda(0)
    criterion = pt.losses.JaccardLoss().cuda(0)

    for name, model in segm_models_dict.items():
        enc_params = count_parameters(model.encoder) / 1e6
        total_params = count_parameters(model) / 1e6
        print(f"{name}. Encoder {enc_params:.2f}M. Decoder {total_params - enc_params:.2f}M. Total {total_params:.2f}M params")
        test_model(model, forward_only=args.forward)
