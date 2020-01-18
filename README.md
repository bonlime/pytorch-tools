# Pytorch-Tools

Tool box for PyTorch for fast prototyping.

# Overview  
* [FitWrapper](./pytorch_tools/fit_wrapper/README.md) - Keras like model trainer
* [Losses](./pytorch_tools/losses/README.md) - collection of different Loss functions.
* [Metrics](./pytorch_tools/metrics/README.md) - collection of metrics.
* [Models](./pytorch_tools/models/README.md) - classification model zoo.
* [Optimizers](./pytorch_tools/optim/README.md)
* [Segmentation Models](./pytorch_tools/segmentation_models/README.md) - segmentation models zoo
* [TTA wrapper](./pytorch_tools/tta_wrapper/README.md) - wrapper for easy test-time augmentation

# Installation
`pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git`  
`pip install git+https://github.com/bonlime/pytorch-tools.git@master`
