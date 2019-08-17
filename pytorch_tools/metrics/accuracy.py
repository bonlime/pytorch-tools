class Accuracy:

    def __init__(self, topk=1):
        self.name = 'Acc@' + str(topk)
        self.topk = topk
    
    def __call__(self, output, target):
        _, pred = output.topk(self.topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:self.topk].view(-1).float().sum(0, keepdim=True)
        return  correct_k.mul(100.0 / output.size(0))

class BalancedAccuracy:
    """ BalancedAccuracy == mean of recalls for each class
        >>> y_true = [0, 1, 0, 0, 1, 0]
        >>> y_pred = [0, 1, 0, 0, 0, 1]
        >>> BalancedAccuracy()(y_true, y_pred)
        0.625
    """
    def __init__(self):
        self.name = 'BalancedAcc'

    def __call__(self, output, target):
        # if raw preds, then argmax them
        if output.shape != target.shape:
            output = output.argmax(1)
        correct = output.eq(target)
        result = 0
        for cls in target.unique():
            tp = (correct * target.eq(cls)).sum().float()
            tp_fn = target.eq(cls).sum()
            result += tp / tp_fn
        return result.mul(100. / target.unique().size(0))