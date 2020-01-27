from pytorch_tools.losses.dice_jaccard import DiceLoss, JaccardLoss


class DiceScore(DiceLoss):
    """Implementation of Dice score for image segmentation task"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Dice"

    def __call__(self, y_pred, y_true):
        loss = super().__call__(y_pred, y_true)
        return 1 - loss


class JaccardScore(JaccardLoss):
    """Implementation of Jaccard score for image segmentation task"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Jaccard"

    def __call__(self, y_pred, y_true):
        loss = super().__call__(y_pred, y_true)
        return 1 - loss
