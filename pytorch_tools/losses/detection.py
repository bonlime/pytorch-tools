import torch
import torch.nn as nn
import pytorch_tools as pt
from functools import partial
from pytorch_tools.utils.box import generate_targets
from pytorch_tools.losses.base import Loss


class DetectionLoss(Loss):
    """Constructs Wrapper for Object Detection which combines Focal Loss and L1 loss"""

    def __init__(
        self,
        anchors=None,
        focal_gamma=2.0,
        focal_alpha=0.25,
        box_weight=50,
        matched_iou=0.5,
        unmatched_iou=0.4,
        script_cls_loss=True,
    ):
        super().__init__()
        self.cls_criterion = pt.losses.FocalLoss(reduction="none", gamma=focal_gamma, alpha=focal_alpha)
        if script_cls_loss:
            # reduces memory footprint and is faster
            self.cls_criterion = torch.jit.script(self.cls_criterion)

        # for some reasons Smooth loss gives NaNs during training if you start from pretrained model. while with L1
        # it trains perfectly fine. Also mmdet states that L1 works better so I leave L1 as default loss for now
        # self.box_criterion = pt.losses.SmoothL1Loss(delta=huber_delta, reduction="none")
        self.box_criterion = pt.losses.L1Loss(reduction="none")
        self.generate_targets = partial(
            generate_targets, matched_iou=matched_iou, unmatched_iou=unmatched_iou
        )

        # The loss normalizer normalizes the training loss according to the
        # number of positive samples. Using EMA could reduce the variance
        # of the normalizer thus improves the performance.
        # The hardcode 100 is not important as long as it is reasonable
        # and not too small according to Detectron2
        self.register_buffer("loss_normalizer", torch.tensor(100))
        # register anchors so that they are moved to cuda only once
        self.register_buffer("anchors", anchors)
        self.box_weight = box_weight

    def forward(self, outputs, target):
        """
        Args:
            outputs (Tuple[torch.Tensor]): cls_outputs, box_outputs
            target (torch.Tensor): shape [BS x N x 5]
        """
        cls_out, box_out = outputs
        box_t, cls_t, matches = self.generate_targets(
            anchors=self.anchors, batch_gt_boxes=target, num_classes=cls_out.size(2)
        )
        box_loss = self.box_criterion(box_out, box_t).masked_select(matches > 0).sum()
        cls_loss = self.cls_criterion(cls_out, cls_t).masked_select(matches >= 0).sum()
        # use foreground and background for classification and only foreground for regression
        num_fg = (matches > 0).sum()
        # using hardcoded value 0.9 for momentum. it works reasonably well and I doubt anyone would optimize it anyway
        self.loss_normalizer = self.loss_normalizer * 0.9 + num_fg * 0.1
        box_loss.div_(self.loss_normalizer + 1)
        cls_loss.div_(self.loss_normalizer + 1)

        # will leave it here for now
        if torch.isnan(cls_loss) or torch.isnan(box_loss):
            raise ValueError

        loss = cls_loss + self.box_weight * box_loss
        return loss
