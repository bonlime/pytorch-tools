import pytorch_tools as pt
from pytorch_tools.metrics import Accuracy
from pytorch_tools.metrics import BalancedAccuracy
from pytorch_tools.metrics import DiceScore, JaccardScore
#from ..utils.misc import to_numpy
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np 

def test_accuracy():
    output = torch.rand((16,4))
    output_np = output.numpy()
    target = torch.randint(0,4,(16,))
    target_np = target.numpy()
    expected = 100 * accuracy_score(target_np, np.argmax(output_np, 1))
    result = Accuracy()(output, target).flatten().numpy()
    assert np.allclose(expected, result)

def test_balanced_accuracy():
    output = torch.rand((16,4))
    output_np = output.numpy()
    target = torch.randint(0,4,(16,))
    target_np = target.numpy()
    expected = 100 * balanced_accuracy_score(target_np, np.argmax(output_np, 1))
    result = BalancedAccuracy()(output, target).flatten().numpy()
    assert np.allclose(expected, result)

IM_SIZE = 10
BS = 8
# only check that score == 1 - loss. see losses/test_losses for more tests

def test_dice_score():
    inp = torch.randn(BS, 1, IM_SIZE, IM_SIZE).float()
    target = torch.randint(0,2,(BS, 1, IM_SIZE, IM_SIZE)).float()

    dice_score = DiceScore(mode='binary', from_logits=False)(inp, target)
    dice_loss = pt.losses.DiceLoss(mode='binary', from_logits=False)(inp, target)
    assert dice_score == 1 - dice_loss


def test_jaccard_score():
    inp = torch.randn(BS, 1, IM_SIZE, IM_SIZE).float()
    target = torch.randint(0,2,(BS, 1, IM_SIZE, IM_SIZE)).float()

    jaccard_score = JaccardScore(mode='binary', from_logits=False)(inp, target)
    jaccard_loss = pt.losses.JaccardLoss(mode='binary', from_logits=False)(inp, target)
    assert jaccard_score == 1 - jaccard_loss