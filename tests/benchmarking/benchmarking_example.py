"""Example of measuring memory consumption and speed in PyTorch"""
import torch
import time
from torch.autograd import Variable

# #### MEMORY ####
def consume_gpu_ram(n):
    return torch.ones((n, n)).cuda(0)


def consume_gpu_ram_256mb():
    return consume_gpu_ram(2 ** 13)


# should be 1024 peak, 0 used
z = [consume_gpu_ram_256mb() for i in range(4)]  # 1GB
del z
print("Peak memory: {}Mb".format(torch.cuda.max_memory_allocated(0) / 2 ** 10 / 2 ** 10))
print("Current memory: {}Mb".format(torch.cuda.memory_allocated(0) / 2 ** 10 / 2 ** 10))
torch.cuda.reset_max_memory_allocated()

# should be: 512 peaked, 256 used
c1 = consume_gpu_ram_256mb()
c2 = consume_gpu_ram_256mb()
del c1
print("Peak memory: {}Mb".format(torch.cuda.max_memory_allocated(0) / 2 ** 10 / 2 ** 10))
print("Current memory: {}Mb".format(torch.cuda.memory_allocated(0) / 2 ** 10 / 2 ** 10))
torch.backends.cudnn.benchmark = False
#### SPEED ####
x = torch.ones((8, 3, 32, 32), requires_grad=True).cuda(0)
conv = torch.nn.Conv2d(3, 64, 5).cuda(0)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
start_cpu = time.time()
y = torch.mean(conv(x))
y.backward()
end_cpu = time.time()
end.record()
torch.cuda.synchronize()
gpu_time = start.elapsed_time(end)
cpu_time = end_cpu - start_cpu
print(f"Gpu msecs: {gpu_time:.3f}. Cpu msecs: {cpu_time * 1e3:.3f}")
