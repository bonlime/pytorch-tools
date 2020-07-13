class Accuracy:
    """
    Accuracy with support for binary and multiclass classification.
    Also supports images
    
    Args:
        y_pred (Tensor): raw logits of shape (N, C) or (N, C, H, W)
        y_true (Long Tensor): true classes of shape (N,) or one-hot encoded of shape (N, C)
            For images target should have shape (N, C, H, W) or (N, 1, H, W)
    """

    def __init__(self, topk=1):
        self.name = "Acc@" + str(topk)
        self.topk = topk

    def __call__(self, y_pred, y_true):
        if y_pred.size(1) == 1:
            # binary case
            y_pred = y_pred.gt(0).long()
            return y_pred.eq(y_true).float().mean() * 100.0
        else:
            # multiclass case
            _, y_pred = y_pred.topk(self.topk, 1, True, True)  # BS x C (x H x W)-> BS x topk (x H x W)
            if y_pred.dim() == 4:  # BS x topk x H x W -> BS * H * W x topk
                y_pred = y_pred.permute(0, 2, 3, 1).reshape(-1, self.topk)
            y_pred = y_pred.t()  # BS (* H * W) x topk -> topk x BS (* H * W)
            # revert one hot
            if y_true.dim() > 1 and y_true.size(1) != 1:
                y_true = y_true.argmax(1)  # BS x C (x H x W) -> BS (x H x W)
            y_true = y_true.flatten()  # maybe turn BS x 1 (x H x W) -> BS (* H * W)
            correct = y_pred.eq(y_true[None])
            return correct.flatten().sum().float() / y_true.size(0) * 100.0


class BalancedAccuracy:
    """
    BalancedAccuracy == mean of recalls for each class
    >>> y_true = [0, 1, 0, 0, 1, 0]
    >>> y_pred = [0, 1, 0, 0, 0, 1]
    >>> BalancedAccuracy()(y_true, y_pred)
    0.625

    Args:
        y_pred (Tensor): raw logits of shape (N, C) or already argmaxed of shape (N,)
        y_true (Long Tensor): true classes of shape (N,) or one-hot encoded of shape (N, C)
    """

    def __init__(self):
        self.name = "BalancedAcc"

    def __call__(self, y_pred, y_true):
        if len(y_true.shape) == 2:
            y_true = y_true.argmax(1)
        if len(y_pred.shape) == 2:
            y_pred = y_pred.argmax(1)
        correct = y_pred.eq(y_true)
        result = 0
        for cls in y_true.unique():
            tp = (correct * y_true.eq(cls)).sum().float()
            tp_fn = y_true.eq(cls).sum()
            result += tp / tp_fn
        return result.mul(100.0 / y_true.unique().size(0))
