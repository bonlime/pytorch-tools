# Collection of losses
All new losses should be inherited from `.base.Loss` class to ensure that they can be combined in a nice way.  
## Example of combining multiple losses
```
from pytorch_tools import losses
custom_loss = losses.BinaryDiceLoss() * 0.5 + losses.BinaryFocalLoss() * 5
```