# idea from https://github.com/qubvel/tta_wrapper reworked for PyTorch
import itertools
import torch
import torch.nn as nn
from pytorch_tools.tta_wrapper import functional as F


class Augmentation(object):

    transforms = {
        "h_flip": F.HFlip(),
        "v_flip": F.VFlip(),
        "rotation": F.Rotate(),
        "h_shift": F.HShift(),
        "v_shift": F.VShift(),
        # 'contrast': F.Contrast(),
        "add": F.Add(),
        "mul": F.Multiply(),
    }

    def __init__(self, **params):
        super().__init__()

        transforms = [Augmentation.transforms[k] for k in params.keys()]
        transform_params = [params[k] for k in params.keys()]

        # add identity parameters for all transforms and convert to list
        transform_params = [t.prepare(params) for t, params in zip(transforms, transform_params)]

        # get all combinations of transforms params
        transform_params = list(itertools.product(*transform_params))

        self.forward_aug = [t.forward for t in transforms]
        self.forward_params = transform_params

        # reverse transforms
        self.backward_aug = [t.backward for t in transforms[::-1]]
        # reverse params
        self.backward_params = [p[::-1] for p in transform_params]

        self.n_transforms = len(transform_params)

    def forward(self, x):
        self.bs = x.shape[0]
        transformed_batches = []
        for i, args in enumerate(self.forward_params):
            batch = x
            for f, arg in zip(self.forward_aug, args):
                batch = f(batch, arg)
            transformed_batches.append(batch)
        # returns shape B*Aug x C x H x W
        return torch.cat(transformed_batches, 0)

    def backward(self, x):
        # reshape to separate batches
        x = x.reshape([-1, self.bs, *x.shape[1:]])
        transformed_batches = []
        for i, args in enumerate(self.backward_params):
            batch = x[i]
            for f, arg in zip(self.backward_aug, args):
                batch = f(batch, arg)
            transformed_batches.append(batch)
        return torch.cat(transformed_batches, 0)


class TTA(nn.Module):
    """Module wrapper for convinient TTA. 
    Wrapper add augmentation layers to your model like this:

            Input
              |           # input batch; shape B, H, W, C
         / / / \ \ \      # duplicate image for augmentation; shape N*B, H, W, C
        | | |   | | |     # apply augmentations (flips, rotation, shifts)
     your nn.Module model
        | | |   | | |     # reverse transformations (this part is skipped for classification)
         \ \ \ / / /      # merge predictions (mean, max, gmean)
              |           # output mask; shape B, H, W, C
            Output
            
    Args:
        model (nn.Module): 
        segm (bool): Flag to revert augmentations before merging. Requires output of a model
            to be of the same size as input. Defaults to False.
        h_flip (bool): Horizontal flip.
        v_flip (bool): Vertical flip.
        h_shift (List[int]): list of horizontal shifts in pixels (e.g. [10, -10])
        v_shift (List[int]): list of vertical shifts in pixels (e.g. [10, -10])
        rotation (List[int]): list of angles (deg) for rotation should be divisible by 90 deg (e.g. [90, 180, 270])
        add (List[float]): list of floats to add to input images.
        mul (List[float]): list of float to multiply input. Ex: [0.9, 1.1]
        merge (str): Mode of merging augmented predictions. One of 'mean', 'gmean' and 'max'. 
            When using 'gmean' option make sure that predictions are less than 1 or number of augs isn't too large
            otherwise it could lead to an overflow.
        activation (str): Activation to apply to predictions before merging. One of {None, `sigmoid`, `softmax`}.  
    
    Returns:
        nn.Module
    """
    def __init__(
        self,
        model,
        segm=False,
        h_flip=False,
        v_flip=False,
        h_shift=None,
        v_shift=None,
        rotation=None,
        # contrast=None,
        add=None,
        mul=None,
        merge="mean",
        activation=None,
    ):

        super(TTA, self).__init__()
        self.tta = Augmentation(
            h_flip=h_flip,
            v_flip=v_flip,
            h_shift=h_shift,
            v_shift=v_shift,
            rotation=rotation,
            # contrast=contrast,
            add=add,
            mul=mul,
        )
        self.n_transforms = self.tta.n_transforms
        self.model = model
        self.segm = segm
        if merge == "mean":
            self.merge = F.mean
        elif merge == "gmean":
            self.merge = F.gmean
        elif merge == "max":
            self.merge = F.max
        else:
            raise ValueError(f"Merge type {merge} not implemented. Choose from: `mean`, `gmean`, `max`")
        self.activation = activation

    def forward(self, x):
        x = self.tta.forward(x)
        # x.shape = B*N_Transform x C x H x W
        x = self.model(x)
        # x.shape = `B*N_Transform x N_Classes (x H x W)`
        if self.segm:
            x = self.tta.backward(x)
        x = x.reshape([-1, self.tta.bs, *x.shape[1:]])
        # x.shape = `N_Transform x B x N_Classes (x H x W)`
        if self.activation == "sigmoid":
            x.sigmoid_()
        return self.merge(x)
