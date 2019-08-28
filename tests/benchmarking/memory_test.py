import sys
import time
import torch
#import torch.nn as nn
import torch.backends.cudnn as cudnn
from pytorch_tools import models
import numpy as np

BS = 64
N_RUNS = 10
RUN_ITERS = 10
INP = torch.ones((BS, 3, 224, 224), requires_grad=True).cuda(0)
TARGET = torch.ones(BS).long().cuda(0)
criterion = torch.nn.CrossEntropyLoss().cuda(0)

def test_model(model):
    optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    gpu_results = []
    with cudnn.flags(enabled=True, benchmark=True):
        for i in range(N_RUNS):
            # during cudnn benchmarking a lot of memory is used. we need to reset
            # in order to get max mem alloc by the fastest algorithm
            if i == 1: 
                torch.cuda.reset_max_memory_allocated()
            start.record()
            for j in range(RUN_ITERS):
                output = model(INP)
                loss = criterion(output, TARGET)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            end.record()
            torch.cuda.synchronize()
            gpu_time = start.elapsed_time(end)
            gpu_results.append(gpu_time)
        # mean without first to drop benchmarking stage which is usually much slower
        print("Mean of {} runs {} iters each BS={}:\n\t {:.2f}+-{:.2f} msecs gpu. Max memory: {:.2f}Mb".format(
            N_RUNS, RUN_ITERS, BS,
            np.mean(gpu_results[1:])/RUN_ITERS, np.std(gpu_results[1:])/RUN_ITERS, 
            torch.cuda.max_memory_allocated() / 2**20
        ))
    del optimizer
    del model


print('VGG 16 ABN:')
test_model(models.vgg16_bn(norm_layer='abn').cuda(0))

print('VGG 16 InplaceABN:')
test_model(models.vgg16_bn(norm_layer='inplaceabn').cuda(0))

print('Resnet50 ABN:')
test_model(models.resnet50(norm_layer='abn').cuda(0))

print('Resnet50 InplaceABN:')
test_model(models.resnet50(norm_layer='inplaceabn').cuda(0))

print('SE Resnext50_32x4 ABN:')
test_model(models.se_resnext50_32x4d(norm_layer='abn').cuda(0))

print('SE Resnext50_32x4 InplaceABN:')
test_model(models.se_resnext50_32x4d(norm_layer='inplaceabn').cuda(0))
