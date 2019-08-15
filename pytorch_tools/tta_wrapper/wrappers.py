import itertools
import torch
import torch.nn as nn
import pytorch_tools.tta_wrapper.functional as F

# forward на батче работает без проблем. получается
# B x C x H x W => B*Aug x C x H x W => B*Aug x N_Classes.=> B x Aug x
#
# После хотим получить B x N_Classes
# или для сегментации
# B x C x H x W => B*Aug x C x H x W => B*Aug x N_Classes x H x W.
# после хотим получить B x N_Classes x H x W
# решается хитрым reshape


class Augmentation(object):

    transforms = {
        'h_flip': F.HFlip(),
        'v_flip': F.VFlip(),
        # 'rotation': F.Rotate(),
        'h_shift': F.HShift(),
        'v_shift': F.VShift(),
        # 'contrast': F.Contrast(),
        'add': F.Add(),
        'mul': F.Multiply(),
    }

    def __init__(self, **params):
        super().__init__()

        transforms = [Augmentation.transforms[k] for k in params.keys()]
        transform_params = [params[k] for k in params.keys()]

        # add identity parameters for all transforms and convert to list
        transform_params = [t.prepare(params) for t, params in zip(
            transforms, transform_params)]

        # get all combinations of transforms params
        transform_params = list(itertools.product(*transform_params))

        self.forward_aug = [t.forward for t in transforms]
        self.forward_params = transform_params

        # reverse transforms
        self.backward_aug = [t.backward for t in transforms[::-1]]
        # reverse params
        self.backward_params = [p[::-1] for p in transform_params]

        self.n_transforms = len(transform_params)

    # @property
    def forward(self, x):
        # only stack first image in a batch for now
        image = x
        self.bs = x.shape[0]
        #images = torch.cat([x[0]] * self.n_transforms, axis=0)

        transformed_images = []
        for i, args in enumerate(self.forward_params):
            for f, arg in zip(self.forward_aug, args):
                image = f(image, arg)  # actually not image but a batch
            transformed_images.append(image)

        #x.reshape([bs, -1, *x.shape[1:]]).mean(1)
        # returns shape B*Aug x C x H x W
        return torch.cat(transformed_images, 0)

    # @property
    def backward(self, x):
        # x.reshape([-1, self.bs, *x.shape[1:]]).mean(0)
        x = x.reshape([-1, self.bs, *x.shape[1:]])
        transformed_images = []
        for i, args in enumerate(self.backward_params):
            image = x[i]
            for f, arg in zip(self.backward_aug, args):
                image = f(image, arg)
            transformed_images.append(image)
        return torch.cat(transformed_images, 0)


class TTA_Wrapper(nn.Module):

    def __init__(self,
                 model,
                 segm=False,
                 h_flip=False,
                 v_flip=False,
                 h_shift=None,
                 v_shift=None,
                 # rotation=None,
                 # contrast=None,
                 add=None,
                 mul=None,
                 merge='mean'):
        super(TTA_Wrapper, self).__init__()
        self.tta = Augmentation(h_flip=h_flip,
                                v_flip=v_flip,
                                h_shift=h_shift,
                                v_shift=v_shift,
                                # rotation=rotation,
                                # contrast=contrast,
                                add=add,
                                mul=mul
                                )
        self.model = model
        self.segm = segm
        if merge == 'mean':
            self.merge = F.mean
        elif merge == 'gmean':
            self.merge = F.gmean
        elif merge == 'max':
            self.merge = F.max
        else:
            raise ValueError(
                "Merge type {} not implemented. Valid options are: `mean`, `gmean`, `max`".format(merge))

    def forward(self, x):
        x = self.tta.forward(x)
        # x.shape = B*N_Transform x C x H x W
        x = self.model(x)
        # x.shape = `B*N_Transform x N_Classes` for classification or
        # x.shape = `B*N_Transform x N_Classes x H x W` for segmentation
        if self.segm:
            x = self.tta.backward(x)
        #print(x.shape)
        print(x)
        x = x.reshape([-1, self.tta.bs, *x.shape[1:]])
        return self.merge(x)