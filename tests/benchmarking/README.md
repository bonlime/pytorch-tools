# Speed comparison of different models and their memory usage

**VGG 16 ABN:**  
Mean of 10 runs 50 iters each BS=128: 719.29+-1.45 msecs gpu. Max memory: 17732.86Mb  
**VGG 16 InplaceABN:**  
Mean of 10 runs 50 iters each BS=128: 727.05+-0.73 msecs gpu. Max memory: 11857.39Mb

**VGG 16 Naive PostAct:**  
Mean of 10 runs 10 iters each BS=128: 652.89+-0.98 msecs gpu. Max memory: 17585.80Mb  
**VGG 16 Chpnt PostAct:**  
Mean of 10 runs 10 iters each BS=128: 847.33+-10.08 msecs gpu. Max memory: 13423.22Mb

**VGG 16 Naive PreAct:**  
Mean of 10 runs 10 iters each BS=128: 641.53+-1.29 msecs gpu. Max memory: 17585.80Mb  
**VGG 16 Chpnt PreAct:**  
Mean of 10 runs 10 iters each BS=128: 841.74+-12.48 msecs gpu. Max memory: 13423.22Mb

**Resnet50 ABN:**  
Mean of 10 runs 50 iters each BS=128: 447.21+-0.61 msecs gpu. Max memory: 10884.15Mb  
**Resnet50 InplaceABN:**  
Mean of 10 runs 50 iters each BS=128: 479.64+-1.32 msecs gpu. Max memory: 8888.33Mb

**SE Resnext50_32x4 ABN:**  
Mean of 10 runs 50 iters each BS=128: 700.99+-0.67 msecs gpu. Max memory: 16826.77Mb  
**SE Resnext50_32x4 InplaceABN:**  
Mean of 10 runs 50 iters each BS=128: 718.09+-2.18 msecs gpu. Max memory: 10575.54Mb

**SE Resnext50_32x4 ABN:**  
Mean of 10 runs 10 iters each BS=64: 394.58+-0.41 msecs gpu. Max memory: 8607.16Mb  
**SE Resnext50_32x4 InplaceABN:**  
Mean of 10 runs 10 iters each BS=64: 410.02+-3.24 msecs gpu. Max memory: 5470.71Mb

**EffNet B0 rwightman 5.29M params**  
Mean of 10 runs 10 iters each BS=64: 27.15+-0.03 msecs Forward. 117.22+-3.62 msecs Backward. Max memory: 5491.36Mb  
**EffNet B1 rwightman 7.79M params**  
Mean of 10 runs 10 iters each BS=64: 39.05+-0.02 msecs Forward. 151.09+-3.33 msecs Backward. Max memory: 7680.92Mb  
**EffNet B2 rwightman 9.11M params**  
Mean of 10 runs 10 iters each BS=64: 41.30+-0.05 msecs Forward. 148.15+-2.87 msecs Backward. Max memory: 8064.01Mb  
**EffNet B3 rwightman 12.23M params**  
Mean of 10 runs 10 iters each BS=64: 53.73+-0.02 msecs Forward. 192.05+-2.13 msecs Backward. Max memory: 10675.62Mb  


**EffNet B1 rwightman CUDA Swish 7.79M params**  
Mean of 10 runs 10 iters each BS=64: 38.65+-0.01 msecs Forward. 145.76+-3.77 msecs Backward. Max memory: 7741.70Mb  
**EffNet B1 rwightman RELU 7.79M params**  
Mean of 10 runs 10 iters each BS=64: 38.50+-0.01 msecs Forward. 150.01+-4.07 msecs Backward. Max memory: 5767.71Mb  


**EffNet B0 lukemelas 5.29M params**  
Mean of 10 runs 10 iters each BS=64: 38.59+-0.01 msecs Forward. 141.19+-4.33 msecs Backward. Max memory: 5980.59Mb  
**EffNet B1 lukemelas 7.79M params**  
Mean of 10 runs 10 iters each BS=64: 54.97+-0.01 msecs Forward. 185.83+-5.13 msecs Backward. Max memory: 8254.66Mb  

**Resnet50 Baseline: 25.56M params**  
Mean of 10 runs 10 iters each BS=64: 59.65+-0.07 msecs Forward. 164.39+-2.58 msecs Backward. Max memory: 5935.15Mb  

**Resnet50 AMP** 25.56M params
Mean of 10 runs 10 iters each BS=256: 111.44+-0.03 msecs Forward. 357.80+-1.61 msecs Backward. Max memory: 11226.14Mb. 545.56 imgs/sec

**Resnet34 Leaky ReLU 21.82M params**  
Mean of 10 runs 10 iters each BS=64: 30.81+-0.18 msecs Forward. 103.95+-1.05 msecs Backward. Max memory: 2766.59Mb  
**Resnet34 ReLU 21.82M params**  
Mean of 10 runs 10 iters each BS=64: 30.76+-0.01 msecs Forward. 103.90+-1.10 msecs Backward. Max memory: 2932.52Mb  
**Resnet34 Mish 21.82M params**  
Mean of 10 runs 10 iters each BS=64: 35.07+-0.02 msecs Forward. 112.40+-4.26 msecs Backward. Max memory: 3739.84Mb  
**Resnet34 Mish Naive 21.82M params**  
Mean of 10 runs 10 iters each BS=64: 37.73+-0.01 msecs Forward. 125.56+-5.10 msecs Backward. Max memory: 5846.58Mb  


## Methodology

Check `benchmarking_example.py` for an example of how to measure speed and memory usage. NOTE: Memory usage reported by pytorch doesn't include memory used by internal CUDA libraries which also has to be loaded to device. It means that more memory is actually used than reported by this tests. But the number of additional memory is always constant for the same device.
