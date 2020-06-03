# PyTorch Segmentation Models Zoo
All models here were either written from scratch or refactored from open-source implementations.  
All models here use `Activated Normalization` layers instead of traditional `Normalization` followed by `Activation`. It makes changing activation function and normalization layer easy and convenient. It also allows using [Inplace Activated Batch Norm](https://github.com/mapillary/inplace_abn) from the box, which is essential for reducing memory footprint in segmentation tasks.


## Encoders 
All [models](../models/) could be used as feature extractors (aka backbones) for segmentation architectures. Almost all combinations of backbones and segm.model are supported.


## Features
* Unified API. Create `Unet, SegmentaionFPN, HRnet` models using the same code
* Support for custom number of input channels in pretrained encoders
* All core functionality covered with tests


## Repositories used
* [Torch Vision Main Repo](https://github.com/pytorch/vision)  
* [Cadene pretrained models](https://github.com/Cadene/pretrained-models.pytorch/)
* [Ross Wightman models](https://github.com/rwightman/pytorch-image-models/)
* [Pytorch Toolbelt by @BloodAxe](https://github.com/BloodAxe/pytorch-toolbelt)  
* [Segmentation Models py @qubvel](https://github.com/qubvel/segmentation_models.pytorch)  
* [Original HRNet for Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)