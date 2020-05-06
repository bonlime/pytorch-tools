"""Various functions to help with bboxes"""
import torch
import numpy as np

def box2delta(boxes, anchors):
    """Convert boxes to deltas from anchors. Boxes are expected in 'ltrb' format
    Args:
        boxes (torch.Tensor): shape [N, 4]
        anchors (torch.Tensor): shape [N, 4]
    Returns:
        deltas (torch.Tensor): shape [N, 4]. offset_x, offset_y, scale_x, scale_y
    """

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
    boxes_wh = boxes[:, 2:] - boxes[:, :2] + 1
    boxes_ctr = boxes[:, :2] + 0.5 * boxes_wh
    offset_delta = (boxes_ctr - anchors_ctr) / anchors_wh
    scale_delta = torch.log(boxes_wh / anchors_wh)
    return torch.cat([offset_delta, scale_delta], 1)


def delta2box(deltas, anchors):
    """Convert anchors to boxes using deltas. Boxes are expected in 'ltrb' format
    Args:
        deltas (torch.Tensor): shape [N, 4]. 
        anchors (torch.Tensor): shape [N, 4]
    Returns:
        bboxes (torch.Tensor): bboxes obtained from anchors by regression [N, 4]
    """

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + ctr
    pred_wh = torch.exp(deltas[:, 2:]) * anchors_wh

    return torch.cat([pred_ctr - 0.5 * pred_wh, pred_ctr + 0.5 * pred_wh - 1], 1)

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in `ltrb`: (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


# based on https://github.com/NVIDIA/retinanet-examples/
# and on https://github.com/google/automl/ 
def generate_anchors_boxes(
    image_size, 
    num_scales=3,
    aspect_ratios=(1.0, 2.0, 0.5),
    pyramid_levels=[3, 4, 5, 6, 7],
    anchor_scale=4,
):
    """Generates multiscale anchor boxes
    Minimum object size which could be detected is anchor_scale * 2**pyramid_levels[0]. By default it's 32px
    Maximum object size which could be detected is anchor_scale * 2**pyramid_levels[-1]. By default it's 512px 
    
    Args: 
        image_size (int or (int, int)): shape of the image
        num_scales (int): integer number representing intermediate scales added on each level. For instances, 
            num_scales=3 adds three additional anchor scales [2^0, 2^0.33, 2^0.66] on each level.
        aspect_ratios (List[int]): Aspect ratios of anchor boxes
        pyramid_levels (List[int]): Levels from which features are taken. Needed to calculate stride
        anchor_scale (float): scale of size of the base anchor. Lower values allows detection of smaller objects.

    Returns:
        anchor_boxes (torch.Tensor): stacked anchor boxes on all feature levels. shape [N, 4]. 
            boxes are in 'ltrb' format
    """
    
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    scale_vals = [anchor_scale * 2 ** (i / num_scales) for i in range(num_scales)]
    # from lowest stride to largest. Anchors from models should be in the same order!
    strides = [2**i for i in pyramid_levels]
    
    # get offsets for anchor boxes for one pixel
    # can rewrite in pure Torch but using np is more convenient. This function usually should only be called once
    num_anchors = len(scale_vals) * len(aspect_ratios)
    ratio_vals_sq = np.sqrt(np.repeat(aspect_ratios, len(scale_vals)))
    scale_vals_tiled = np.tile(scale_vals, len(aspect_ratios))[:, np.newaxis]
    wh = np.stack([np.ones(num_anchors) * ratio_vals_sq, np.ones(num_anchors) / ratio_vals_sq], axis=1) 
    lt = - 0.5 * wh * scale_vals_tiled
    rb = 0.5 * wh * scale_vals_tiled
    base_offsets = torch.from_numpy(np.hstack([lt, rb])) # [num_anchors, 4]
    base_offsets = base_offsets.view(-1, 1, 1, 4) # [num_anchors, 1, 1, 4]

    # generate anchor boxes for all given strides
    all_anchors = []
    for stride in strides:
        x, y = torch.meshgrid([torch.arange(stride / 2, image_size[i], stride) for i in range(2)])
        xyxy = torch.stack((x, y, x, y), 2).unsqueeze(0) 
        anchors = (xyxy + base_offsets * stride).view(-1, 4).contiguous()
        all_anchors.append(anchors)
    all_anchors = torch.cat(all_anchors)
    # clip boxes to image. Not sure if we really need to clip them
    # all_anchors[:, 0::2] = all_anchors[:, 0::2].clamp(0, image_size[0])
    # all_anchors[:, 1::2] = all_anchors[:, 1::2].clamp(0, image_size[1])
    return all_anchors

def generate_targets(anchors, gt_boxes, num_classes, matched_iou=0.5, unmatched_iou=0.4):
    """Generate targets for regression and classification for SINGLE image
    
    Based on IoU between anchor and true bounding box there are three types of anchor boxes
    1) IoU >= matched_iou: Highest similarity. Matched/Positive. Mask value is 1
    2) matched_iou > IoU >= unmatched_iou: Medium similarity. Ignored. Mask value is -1
    3) unmatched_iou > IoU: Lowest similarity. Unmatched/Negative. Mask value is 0
    
    Args:
        anchors (torch.Tensor): all anchors on a single image. shape [N, 4]
        gt_boxes (torch.Tesor): all groud truth bounding boxes and classes on the image. shape [N, 5]
            classes are expected to be in the last column. 
            bboxes are in `ltrb` format!
        num_classes (int): number of classes. needed for one-hot encoding labels
        matched_iou (float):  
        unmatched_iou (float):
    
    Returns:
        box_target, cls_target, matches_mask
    
    """

    gt_boxes, gt_classes = gt_boxes.split(4, dim=1)
    overlap = box_iou(anchors, gt_boxes)
    
    # Keep best box per anchor
    overlap, indices = overlap.max(1)
    box_target = box2delta(gt_boxes[indices], anchors) # I've double checked that it's corrects
    # TODO: add test that anchor + box target gives gt_bbox
    
    # There are three types of anchors. 
    # matched (with objects), unmatched (with background), and in between (which should be ignored)
    IGNORED_VALUE = -1
    UNMATCHED_VALUE = 0
    matches_mask = torch.ones_like(overlap) * IGNORED_VALUE
    matches_mask[overlap < unmatched_iou] = UNMATCHED_VALUE # background
    matches_mask[overlap >= matched_iou] = 1

    # Generate one-hot-encoded target classes
    cls_target = torch.zeros(
        (anchors.size(0), num_classes + 1), device=gt_classes.device, dtype=gt_classes.dtype
    )
    gt_classes = gt_classes[indices].long()
    gt_classes[overlap < unmatched_iou] = num_classes  # background has no class
    cls_target.scatter_(1, gt_classes, 1)
    cls_target = cls_target[:, :num_classes] # remove background class from one-hot

    return cls_target, box_target, matches_mask