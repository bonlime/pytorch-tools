from ..losses.functional import soft_dice_score
from ..losses.functional import soft_jaccard_score


class DiceScore:
    """Implementation of Dice loss for binary image segmentation task
    """

    def __init__(self, from_logits=True, smooth=1e-3):
        super(DiceScore, self).__init__()
        self.name = 'Dice'
        self.from_logits = from_logits
        self.smooth = smooth

    def __call__(self, output, target):
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        dice = soft_dice_score(output, target, from_logits=self.from_logits, smooth=self.smooth)
        return dice

class JaccardScore:
    """Implementation of Dice loss for binary image segmentation task
    """

    def __init__(self, from_logits=True, smooth=1e-3):
        super(JaccardScore, self).__init__()
        self.name = 'IoU'
        self.from_logits = from_logits
        self.smooth = smooth

    def __call__(self, output, target):
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        iou = soft_jaccard_score(output, target, from_logits=self.from_logits, smooth=self.smooth)
        return iou