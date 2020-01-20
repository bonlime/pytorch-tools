# Test Time Augmentation Wrapper
This module wraps existing PyTorch model, performs inference on multiple augmented images and them merges the predictions into one. 

Wrapper adds augmentation layers to your model like this:
```
            Input
              |           # input batch; shape B, H, W, C
         / / / \ \ \      # duplicate image for augmentation; shape N*B, H, W, C
        | | |   | | |     # apply augmentations (flips, rotation, shifts)
     your nn.Module model
        | | |   | | |     # reverse transformations (this part is skipped for classification)
         \ \ \ / / /      # merge predictions (mean, max, gmean)
              |           # output mask; shape B, H, W, C
            Output
```
# Example
```python
from pytorch_tools.tta_wrapper import TTA
# 2 x 3 x 3 = 18 augmentations per image!
tta_model = TTA(model, h_flip=True, h_shift=[5,-5], mul=[0.9, 1.1])
for batch in loader:
    prediction = tta_model(batch)
```
