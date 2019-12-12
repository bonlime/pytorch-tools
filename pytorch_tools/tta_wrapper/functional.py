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

    def forward(self, batch, param):
        raise NotImplementedError

    def backward(self, batch, param):
        raise NotImplementedError


class SingleTransform(DualTransform):
    def backward(self, batch, param):
        return batch


class HFlip(DualTransform):

    identity_param = 0

    def prepare(self, params):
        if params == False:
            return [0]
        if params == True:
            return [1, 0]

    def forward(self, batch, param):
        return batch.flip(2) if param else batch

    def backward(self, batch, param):
        return self.forward(batch, param)


class VFlip(DualTransform):

    identity_param = 0

    def prepare(self, params):
        if params == False:
            return [0]
        if params == True:
            return [1, 0]

    def forward(self, batch, param):
        return batch.flip(3) if param else batch

    def backward(self, batch, param):
        return self.forward(batch, param)


class Rotate(DualTransform):

    identity_param = 0

    def forward(self, batch, angle):
        # rotation is couterclockwise
        k = angle // 90
        return torch.rot90(batch, k, (2, 3))

    def backward(self, batch, angle):
        return self.forward(batch, -angle)


class HShift(DualTransform):

    identity_param = 0

    def forward(self, batch, param):
        return batch.roll(param, dims=3)

    def backward(self, batch, param):
        return batch.roll(-param, dims=3)


class VShift(DualTransform):

    identity_param = 0

    def forward(self, batch, param):
        return batch.roll(param, dims=2)

    def backward(self, batch, param):
        return batch.roll(-param, dims=2)


# class Contrast(SingleTransform):

#     identity_param = 1

#     def forward(self, batch, param):
#         return tf.image.adjust_contrast(batch, param)


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
