class Accuracy:
    def __init__(self, topk=1):
        self.name = "Acc@" + str(topk)
        self.topk = topk

    def __call__(self, output, target):
        """Args:
            output (Tensor): raw logits of shape (N, C)
            target (Long Tensor): true classes of shape (N,) or one-hot encoded of shape (N, C)"""
        if len(target.shape) == 2:
            target = target.argmax(1)
        _, pred = output.topk(self.topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[: self.topk].view(-1).float().sum(0, keepdim=True)
        return correct_k.mul(100.0 / output.size(0))


class BalancedAccuracy:
    """ BalancedAccuracy == mean of recalls for each class
        >>> y_true = [0, 1, 0, 0, 1, 0]
        >>> y_pred = [0, 1, 0, 0, 0, 1]
        >>> BalancedAccuracy()(y_true, y_pred)
        0.625
    """

    def __init__(self):
        self.name = "BalancedAcc"

    def __call__(self, output, target):
        """Args:
            output (Tensor): raw logits of shape (N, C) or already argmaxed of shape (N,)
            target (Long Tensor): true classes of shape (N,) or one-hot encoded of shape (N, C)"""
        if len(target.shape) == 2:
            target = target.argmax(1)
        if len(output.shape) == 2:
            output = output.argmax(1)
        correct = output.eq(target)
        result = 0
        for cls in target.unique():
            tp = (correct * target.eq(cls)).sum().float()
            tp_fn = target.eq(cls).sum()
            result += tp / tp_fn
        return result.mul(100.0 / target.unique().size(0))
