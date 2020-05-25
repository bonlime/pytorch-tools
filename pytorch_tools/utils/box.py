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

    anchors_wh = anchors[:, 2:] - anchors[:, :2] # + 1
    ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + ctr
    pred_wh = torch.exp(deltas[:, 2:]) * anchors_wh

    # TODO: make sure this +- 1 for centers aligns with Detectron implementation
    # return torch.cat([pred_ctr - 0.5 * pred_wh, pred_ctr + 0.5 * pred_wh - 1], 1)
    return torch.cat([pred_ctr - 0.5 * pred_wh, pred_ctr + 0.5 * pred_wh], 1)


def box_area(box):
    """Args:
    box (torch.Tensor): shape [N, 4] in 'ltrb' format
    """
    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

def clip_bboxes(bboxes, size):
    """Args:
        bboxes (torch.Tensor): in `ltrb` format. Shape [N, 4]
        size (Union[torch.Size, tuple]): (H, W)"""
    bboxes[:, 0::2] = bboxes[:, 0::2].clamp(0, size[1])
    bboxes[:, 1::2] = bboxes[:, 1::2].clamp(0, size[0])
    return bboxes

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
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

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
        num_anchors (int): number of anchors per location
    """
    
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    scale_vals = [anchor_scale * 2 ** (i / num_scales) for i in range(num_scales)]
    # from lowest stride to largest. Anchors from models should be in the same order!
    strides = [2**i for i in pyramid_levels]
    
    # get offsets for anchor boxes for one pixel
    # can rewrite in pure Torch but using np is more convenient. This function usually should only be called once
    num_anchors = len(scale_vals) * len(aspect_ratios)
    ratio_vals_sq = np.sqrt(np.tile(aspect_ratios, len(scale_vals)))
    scale_vals = np.repeat(scale_vals, len(aspect_ratios))[:, np.newaxis]
    wh = np.stack([np.ones(num_anchors) * ratio_vals_sq, np.ones(num_anchors) / ratio_vals_sq], axis=1) 
    lt = - 0.5 * wh * scale_vals
    rb = 0.5 * wh * scale_vals
    base_offsets = torch.from_numpy(np.hstack([lt, rb])) # [num_anchors, 4]
    base_offsets = base_offsets.view(-1, 1, 1, 4) # [num_anchors, 1, 1, 4]
    # generate anchor boxes for all given strides
    all_anchors = []
    for stride in strides:
        y, x = torch.meshgrid([torch.arange(stride / 2, image_size[i], stride) for i in range(2)])
        xyxy = torch.stack((x, y, x, y), 2).unsqueeze(0) 
        # permute to match TF EffDet anchors order after reshape
        anchors = (xyxy + base_offsets * stride).permute(1, 2, 0, 3).reshape(-1, 4)
        all_anchors.append(anchors)
    all_anchors = torch.cat(all_anchors)
    # clip boxes to image. Not sure if we really need to clip them
    # clip_bboxes(all_anchors, image_size)
    return all_anchors, num_anchors

def generate_targets(anchors, batch_gt_boxes, num_classes, matched_iou=0.5, unmatched_iou=0.4):
    """Generate targets for regression and classification
    
    Based on IoU between anchor and true bounding box there are three types of anchor boxes
    1) IoU >= matched_iou: Highest similarity. Matched/Positive. Mask value is 1
    2) matched_iou > IoU >= unmatched_iou: Medium similarity. Ignored. Mask value is -1
    3) unmatched_iou > IoU: Lowest similarity. Unmatched/Negative. Mask value is 0
    
    Args:
        anchors (torch.Tensor): all anchors on a single image. shape [N, 4]
        batch_gt_boxes (torch.Tesor): all groud truth bounding boxes and classes for the batch. shape [BS, N, 5]
            classes are expected to be in the last column. 
            bboxes are in `ltrb` format!
        num_classes (int): number of classes. needed for one-hot encoding labels
        matched_iou (float):  
        unmatched_iou (float):
    
    Returns:
        box_target, cls_target, matches_mask
    
    """
    def _generate_single_targets(gt_boxes):
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
    
    anchors = anchors.to(batch_gt_boxes) # change device & type if needed
    batch_results = ([], [], [])
    for single_gt_boxes in batch_gt_boxes:
        single_target_results = _generate_single_targets(single_gt_boxes)
        for batch_res, single_res in zip(batch_results, single_target_results):
            batch_res.append(single_res)
    b_cls_target, b_box_target, b_matches_mask = [torch.stack(targets) for targets in batch_results]
    return b_cls_target, b_box_target, b_matches_mask
    
# copied from torchvision
def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float)
    """
    Performs non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.
    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU > iou_threshold
    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = torch.ops.torchvision.nms(boxes_for_nms, scores, iou_threshold)
    return keep

def decode(batch_cls_head, batch_box_head, anchors, img_shape=None, threshold=0.05, top_n=1000, iou_threshold=0.5):
    """
    Decodes raw outputs of a model for easy visualization of bboxes
    
    Args: 
        batch_cls_head (torch.Tensor): shape [BS, *, NUM_CLASSES]
        batch_box_head (torch.Tensor): shape [BS, *, 4]
        anchors (torch.Tensor): shape [*, 4]
        img_shape (Tuple[int]): if given clips predicted bboxes to img height and width
        threshold (float): minimum score threshold to consider object detected
        top_n (int): maximum number of objects per image
        iou_threshold (float): iou_threshold for Non Maximum Supression
        
    Returns:
        out_bboxes (torch.Tensor): bboxes. Shape [BS, TOP_N] If img_shape is not given they are NOT CLIPPED (!)
        out_scores (torch.Tensor): Probability scores for each bbox. Shape [BS, TOP_N]
        out_classes (torch.Tensor): Predicted class for each bbox. Shape [BS, TOP_N]
    """
    
    batch_size = batch_cls_head.size(0)
    anchors = anchors.to(batch_cls_head)
    out_scores = torch.zeros((batch_size, top_n)).to(batch_cls_head)
    out_boxes = torch.zeros((batch_size, top_n, 4)).to(batch_cls_head)
    out_classes = torch.zeros((batch_size, top_n)).to(batch_cls_head)
    # it has to be raw logits but check anyway to avoid aplying sigmoid twice
    if batch_cls_head.min() < 0 or batch_cls_head.max() > 1:
        batch_cls_head = batch_cls_head.sigmoid()
    
    for batch in range(batch_size):
        # get regressed bboxes
        all_img_bboxes = delta2box(batch_box_head[batch], anchors)
        if img_shape: # maybe clip
            all_img_bboxes = clip_bboxes(all_img_bboxes, img_shape)
        # select at most `top_n` bboxes and from them select with score > threshold
        max_cls_score, max_cls_idx = batch_cls_head[batch].max(1)
        top_cls_score, top_cls_idx = max_cls_score.topk(top_n)
        top_cls_idx = top_cls_idx[top_cls_score > threshold]

        im_scores = max_cls_score[top_cls_idx]
        im_classes = max_cls_idx[top_cls_idx]
        im_bboxes = all_img_bboxes[top_cls_idx]
        
        # apply NMS
        nms_idx = batched_nms(im_bboxes, im_scores, im_classes, iou_threshold)
        im_scores = im_scores[nms_idx]
        im_classes = im_classes[nms_idx]
        im_bboxes = im_bboxes[nms_idx]
        
        out_scores[batch, :im_scores.size(0)] = im_scores
        out_classes[batch, :im_classes.size(0)] = im_classes
        out_boxes[batch, :im_bboxes.size(0)] = im_bboxes
        
    return out_boxes, out_scores, out_classes