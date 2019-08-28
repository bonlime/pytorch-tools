# Speed comparison of different models and their memory usage

VGG 16 ABN:
Mean of 10 runs 50 iters each BS=128:
         719.29+-1.45 msecs gpu. Max memory: 17732.86Mb
VGG 16 InplaceABN:
Mean of 10 runs 50 iters each BS=128:
         727.05+-0.73 msecs gpu. Max memory: 11857.39Mb
Resnet50 ABN:
Mean of 10 runs 50 iters each BS=128:
         447.21+-0.61 msecs gpu. Max memory: 10884.15Mb
Resnet50 InplaceABN:
Mean of 10 runs 50 iters each BS=128:
         479.64+-1.32 msecs gpu. Max memory: 8888.33Mb

SE Resnext50_32x4 ABN:
Mean of 10 runs 50 iters each BS=128:
         700.99+-0.67 msecs gpu. Max memory: 16826.77Mb
SE Resnext50_32x4 InplaceABN:
Mean of 10 runs 50 iters each BS=128:
         718.09+-2.18 msecs gpu. Max memory: 10575.54Mb

SE Resnext50_32x4 ABN:
Mean of 10 runs 10 iters each BS=64:
         394.58+-0.41 msecs gpu. Max memory: 8607.16Mb
SE Resnext50_32x4 InplaceABN:
Mean of 10 runs 10 iters each BS=64:
         410.02+-3.24 msecs gpu. Max memory: 5470.71Mb

## Methodology

Check `benchmarking_example.py` for an example of how to measure speed and memory usage. NOTE: Memory usage reported by pytorch doesn't include memory used by internal CUDA libraries which also has to be loaded to device. It means that more memory is actually used than reported by this tests. But the number of additional memory is always constant for the same device.
