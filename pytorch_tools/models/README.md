# Just Another PyTorch Model Zoo
All models here were either written from scratch or refactored from open-source implementations.  
All models here use `Activated Normalization` layers instead of traditional `Normalization` followed by `Activation`. It makes changing activation function and normalization layer easy and convenient. It also allows using [Inplace Activated Batch Norm](https://github.com/mapillary/inplace_abn) from the box, which is essential for reducing memory footprint in segmentation tasks.

## Pretrained models
All default weights from TorchVision repository are supported. There are also weights for modified Resnet family models trained on Imagenet 2012. It's hard to keep this README up to date with new weights, so check the code for all available weight for particular model.  
All models have `pretrained_settings` attribute with training size, mean, std and other useful information about the weights.

## Encoders  
All models from this repo could be used as feature extractors for both object detection and semantic segmentation. Passing `encoder=True` arg will overwrite `forward` method of the model to return features at 5 different resolutions starting from 1/32 to 1/2.

## Features
* Unified API. Create `resnet, efficientnet, hrnet` models using the same code
* Low memory footprint dy to heavy use of inplace operations. Could be reduced even more by using `norm_layer='inplaceabn'`
* Fast models. As of `04.20` Efficient net's in this repo are the fastest available on GitHub (afaik)
* Support for custom number of input channels in pretrained models. Try with `resnet34(pretrained='imagenet', in_channels=7)`
* All core functionality covered with tests


## Repositories used
* [Torch Vision Main Repo](https://github.com/pytorch/vision)  
* [Cadene pretrained models](https://github.com/Cadene/pretrained-models.pytorch/)
* [Ross Wightman models](https://github.com/rwightman/pytorch-image-models/)
* [Inplace ABN](https://github.com/mapillary/inplace_abn)
* [Efficient Densenet](https://github.com/gpleiss/efficient_densenet_pytorch)
* [Official Efficient Net](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)  
* [Original HRNet for Classification](https://github.com/HRNet/HRNet-Image-Classification)