from __future__ import absolute_import
import torch.nn as nn

from .base import Loss
from .focal import FocalLoss
from .dice_jaccard import DiceLoss
from .dice_jaccard import JaccardLoss
from .lovasz import LovaszLoss
from .wing_loss import WingLoss
from .vgg_loss import ContentLoss, StyleLoss
from .smooth import CrossEntropyLoss
from .hinge import BinaryHinge
from .huber import SmoothL1Loss

from .detection import DetectionLoss

from .functional import focal_loss_with_logits
from .functional import soft_dice_score
from .functional import soft_jaccard_score
from .functional import wing_loss
from .functional import binary_hinge


class MSELoss(nn.MSELoss, Loss):
    pass


class L1Loss(nn.L1Loss, Loss):
    pass
