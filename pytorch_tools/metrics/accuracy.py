

class Accuracy:

    def __init__(self, topk=1):
        self.name = 'Acc@' + topk
        self.topk = topk
    
    def __call__(self, output, target):
        # TODO maybe can write it even shorter
        _, pred = output.topk(self.topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:self.topk].view(-1).float().sum(0, keepdim=True)
        return  correct_k.mul(100.0 / output.size(0))


class BalancedAccuracy:
    """y_true = [0, 1, 0, 0, 1, 0]
        >>> y_pred = [0, 1, 0, 0, 0, 1]
        >>> BalancedAccuracy()(y_true, y_pred)
        0.625
    """
    def __init__(self):
        self.name = 'BalancedAcc'

    def __call__(self, output, target):
        raise NotImplementedError