# Pytorch-Tools

Tool box for PyTorch for fast prototyping.

# Overview  
* [FitWrapper](./pytorch_tools/fit_wrapper/) - Keras like model trainer
* [Losses](./pytorch_tools/losses/) - collection of different Loss functions.
* [Metrics](./pytorch_tools/metrics/) - collection of metrics.
* [Models](./pytorch_tools/models/) - classification model zoo.
* [Optimizers](./pytorch_tools/optim/)
* [Segmentation Models](./pytorch_tools/segmentation_models/) - segmentation models zoo
* [TTA wrapper](./pytorch_tools/tta_wrapper/) - wrapper for easy test-time augmentation

# Installation
Requires GPU drivers and CUDA already installed.

`pip install git+https://github.com/bonlime/pytorch-tools.git@master`

It is also recommended to install NVIDIA Apex to allow usage of additional optimizers

`pip install ---upgrade -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git`  