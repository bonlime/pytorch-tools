import torch


class DualTransform:

    identity_param = None

    def prepare(self, params):
        if isinstance(params, tuple):
            params = list(params)
        elif params is None:
            params = []
        elif not isinstance(params, list):
            params = [params]

        if not self.identity_param in params:
            params.append(self.identity_param)
        return params

    def forward(self, image, param):
        raise NotImplementedError

    def backward(self, image, param):
        raise NotImplementedError


class SingleTransform(DualTransform):

    def backward(self, image, param):
        return image


def hflip(flip=True):
    def wrapped(x):
        return x.flip(3) if flip else x
    return wrapped
    
def hshift(shift):
    def wrapped(x):
        return x.roll(shift, axis=1)
    return wrapped

class HFlip(DualTransform):

    identity_param = 0

    def prepare(self, params):
        if params == False:
            return [0]
        if params == True:
            return [1, 0]

    def forward(self, image, param):
        return image.flip(3) if param else image

    def backward(self, image, param):
        return self.forward(image, param)


class VFlip(DualTransform):

    identity_param = 0

    def prepare(self, params):
        if params == False:
            return [0]
        if params == True:
            return [1, 0]

    def forward(self, image, param):
        return image.flip(2) if param else image

    def backward(self, image, param):
        return self.forward(image, param)


# class Rotate(DualTransform):

#     identity_param = 0

#     def forward(self, image, angle):
#         k = angle // 90 if angle >= 0 else (angle + 360) // 90
#         return tf.image.rot90(image, k)

#     def backward(self, image, angle):
#         return self.forward(image, -angle)


class HShift(DualTransform):

    identity_param = 0

    def forward(self, batch, param):
        return batch.roll(param, dims=2)

    def backward(self, image, param):
        return batch.roll(-param, dims=2)


class VShift(DualTransform):

    identity_param = 0

    def forward(self, batch, param):
        return batch.roll(param, dims=3)

    def backward(self, batch, param):
        return batch.roll(-param, dims=3)


# class Contrast(SingleTransform):

#     identity_param = 1

#     def forward(self, image, param):
#         return tf.image.adjust_contrast(image, param)


class Add(SingleTransform):

    identity_param = 0

    def forward(self, batch, param):
        return batch + param


class Multiply(SingleTransform):

    identity_param = 1

    def forward(self, batch, param):
        return batch * param


def gmean(x):
    # x == N_aug x B x N_cls (x H x W)
    g_pow = 1 / x.shape[0]
    x = x.prod(0, False)
    return x.pow(g_pow)


def mean(x):
    # x == N_aug x B x N_cls (x H x W)
    return x.mean(0, False)


def max(x):
    # x == N_aug x B x N_cls (x H x W)
    return x.max(0, False).values
