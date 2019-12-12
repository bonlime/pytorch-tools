import sys
import time
import torch
import pytest
import numpy as np
import torch.backends.cudnn as cudnn
from pytorch_tools import models
from pytorch_tools.utils.misc import AverageMeter

pytest.skip("Not meant for pytest", allow_module_level=True)


def test_model(model):
    model = model.cuda(0)
    optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)
    f_times = []
    fb_times = []
    with cudnn.flags(enabled=True, benchmark=True):
        start = torch.cuda.Event(enable_timing=True)
        f_end = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        def run_once():
            start.record()
            output = model(INP)
            loss = criterion(output, TARGET)
            f_end.record()
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
            f_meter = AverageMeter("F time")
            fb_meter = AverageMeter("FB time")
            for _ in range(RUN_ITERS):
                f_t, fb_t = run_once()
                f_meter.update(f_t)
                fb_meter.update(fb_t)
            f_times.append(f_meter.avg)
            fb_times.append(fb_meter.avg)
        f_times = np.array(f_times)
        fb_times = np.array(fb_times)
        print(
            "Mean of {} runs {} iters each BS={}:\n\t {:.2f}+-{:.2f} msecs Forward. {:.2f}+-{:.2f} msecs Backward. Max memory: {:.2f}Mb".format(
                N_RUNS,
                RUN_ITERS,
                BS,
                f_times.mean(),
                f_times.std(),
                (fb_times - f_times).mean(),
                (fb_times - f_times).std(),
                torch.cuda.max_memory_allocated(0) / 2 ** 20,
            )
        )
    del optimizer
    del model


# all models are first init to cpu memory to find errors earlier
models_dict = {
    # 'VGG16 ABN' : models.vgg16_bn(norm_layer='abn'),
    # 'VGG 16 InplaceABN:': models.vgg16_bn(norm_layer='inplaceabn'),
    # 'Resnet50 No Antialias:' : models.resnet50(antialias=False), #norm_layer='abn',
    # 'Resnet50 Antialias:': models.resnet50(antialias=True), #norm_layer='inplaceabn', ),
    # 'Resnet50 ABN:' : models.se_resnet50(norm_layer='abn'),
    # 'Resnet50 InPlaceABN:': models.se_resnet50(norm_layer='inplaceabn', norm_act='leaky_relu'),
    # 'Densenet121 MemEff' : models.densenet121(memory_efficient=True),
    # 'Densenet121 NotMemEff' : models.densenet121(memory_efficient=False),
    "Densenet121 ABN": models.densenet121(norm_layer="abn", norm_act="leaky_relu"),
    "Densenet121 InPlaceABN": models.densenet121(norm_layer="inplaceabn", norm_act="leaky_relu"),
    # 'SE Resnext50_32x4 ABN:' : models.se_resnext50_32x4d(norm_layer='abn'),
    # 'SE Resnext50_32x4 InplaceABN:' : models.se_resnext50_32x4d(norm_layer='inplaceabn')
}
print("Initialized models")
BS = 64
N_RUNS = 10
RUN_ITERS = 10
INP = torch.ones((BS, 3, 224, 224), requires_grad=True).cuda(0)
TARGET = torch.ones(BS).long().cuda(0)
criterion = torch.nn.CrossEntropyLoss().cuda(0)
for name, model in models_dict.items():
    print(name)
    test_model(model)
