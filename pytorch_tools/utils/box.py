"""
Various functions to help with bboxes for object detection
Everything is covered with tests to ensure correct output and scriptability (torch.jit.script)
Written by @bonlime
"""
import torch


def box2delta(boxes, anchors):
    # type: (Tensor, Tensor)->Tensor
    """Convert boxes to deltas from anchors. Boxes are expected in 'ltrb' format
    Args:
        boxes (torch.Tensor): shape [N, 4] or [BS, N, 4]
        anchors (torch.Tensor): shape [N, 4] or [BS, N, 4]
    Returns:
        deltas (torch.Tensor): shape [N, 4] or [BS, N, 4]
            offset_x, offset_y, scale_x, scale_y
    """
    # cast to fp32 to avoid numerical problems with log
    boxes, anchors = boxes.float(), anchors.float()
    anchors_wh = anchors[..., 2:] - anchors[..., :2]
    anchors_ctr = anchors[..., :2] + 0.5 * anchors_wh
    boxes_wh = boxes[..., 2:] - boxes[..., :2]
    boxes_ctr = boxes[..., :2] + 0.5 * boxes_wh
    offset_delta = (boxes_ctr - anchors_ctr) / anchors_wh
    scale_delta = torch.log(boxes_wh / anchors_wh)
    return torch.cat([offset_delta, scale_delta], -1)


def delta2box(deltas, anchors):
    # type: (Tensor, Tensor)->Tensor
    """Convert anchors to boxes using deltas. Boxes are expected in 'ltrb' format
    Args:
        deltas (torch.Tensor): shape [N, 4] or [BS, N, 4]
        anchors (torch.Tensor): shape [N, 4] or [BS, N, 4]
    Returns:
        bboxes (torch.Tensor): bboxes obtained from anchors by regression 
            Output shape is [N, 4] or [BS, N, 4] depending on input
    """
    # cast to fp32 to avoid numerical problems with exponent
    deltas, anchors = deltas.float(), anchors.float()
    anchors_wh = anchors[..., 2:] - anchors[..., :2]
    ctr = anchors[..., :2] + 0.5 * anchors_wh
    pred_ctr = deltas[..., :2] * anchors_wh + ctr

    # Value for clamping large dw and dh predictions. The heuristic is that we clamp
    # such that dw and dh are no larger than what would transform a 16px box into a
    # 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
    SCALE_CLAMP = 4.135  # ~= np.log(1000. / 16.)
    pred_wh = deltas[..., 2:].clamp(min=-SCALE_CLAMP, max=SCALE_CLAMP).exp() * anchors_wh
    return torch.cat([pred_ctr - 0.5 * pred_wh, pred_ctr + 0.5 * pred_wh], -1)


def box_area(box):
    # type: (Tensor) -> Tensor
    """Args:
    box (torch.Tensor): shape [N, 4] or [BS, N, 4] in 'ltrb' format
    """
    return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])


def clip_bboxes(bboxes, size):
    # type: (Tensor, Tuple[int, int]) -> Tensor
    """Args:
        bboxes (torch.Tensor): in `ltrb` format. Shape [N, 4]
        size (Union[torch.Size, tuple]): (H, W). Shape [2,]"""
    bboxes[:, 0::2] = bboxes[:, 0::2].clamp(0, size[1])
    bboxes[:, 1::2] = bboxes[:, 1::2].clamp(0, size[0])
    return bboxes


def clip_bboxes_batch(bboxes, size):
    # type: (Tensor, Tensor) -> Tensor
    """Args:
        This function could also accept not batched bboxes but it works
        slower than `clip_bboxes` in that case
        bboxes (torch.Tensor): in `ltrb` format. Shape [BS, N, 4]
        size (torch.Tensor): (H, W). Shape [BS, 2] """
    size = size.to(bboxes)
    h_size = size[..., 0].view(-1, 1, 1)  # .float()
    w_size = size[..., 1].view(-1, 1, 1)  # .float()
    h_bboxes = bboxes[..., 1::2]
    w_bboxes = bboxes[..., 0::2]
    zeros = torch.zeros_like(h_bboxes)
    bboxes[..., 1::2] = h_bboxes.where(h_bboxes > 0, zeros).where(h_bboxes < h_size, h_size)
    bboxes[..., 0::2] = w_bboxes.where(w_bboxes > 0, zeros).where(w_bboxes < w_size, w_size)
    # FIXME: I'm using where to support passing tensor. change to `clamp` when PR #32587 is resolved
    # bboxes[:, 0::2] = bboxes[:, 0::2].clamp(0, size[1].item())
    # bboxes[:, 1::2] = bboxes[:, 1::2].clamp(0, size[0].item())
    return bboxes


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def box_iou(boxes1, boxes2):
    # type: (Tensor, Tensor)->Tensor
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
    inter = wh[..., 0] * wh[..., 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


# copied from https://github.com/etienne87/torch_object_rnn/blob/master/core/utils/box.py
def batch_box_iou(box1, box2):
    # type: (Tensor, Tensor) -> Tensor
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
        box1: (tensor) bounding boxes, sized [N,4]. It's supposed to be anchors
        box2: (tensor) bounding boxes, sized [B,M,4].
    Return:
        iou (Tensor[B, N, M]): the BxNxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    # [B,N,M,2] = broadcast_max( (_,N,_,2), (B,_,M,2) )
    lt = torch.max(box1[None, :, None, :2], box2[:, None, :, :2])
    rb = torch.min(box1[None, :, None, 2:], box2[:, None, :, 2:])  # [B,N,M,2]

    wh = (rb - lt).clamp(min=0)  # [B,N,M,2]
    inter = wh[..., 0] * wh[..., 1]  # [B,N,M]

    area1 = box_area(box1)  # [N,]
    area2 = box_area(box2)  # [B,M,]
    iou = inter / (area1[None, :, None] + area2[:, None, :] - inter)  # [B,N,M]
    return iou


# based on https://github.com/NVIDIA/retinanet-examples/
# and on https://github.com/google/automl/
def generate_anchors_boxes(
    image_size, num_scales=3, aspect_ratios=(1.0, 2.0, 0.5), pyramid_levels=(3, 4, 5, 6, 7), anchor_scale=4,
):
    # type: (Tuple[int, int], int, List[float], List[int], int) -> Tuple[Tensor, int]
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
    strides = [2 ** i for i in pyramid_levels]

    # get offsets for anchor boxes for one pixel
    num_anchors = num_scales * len(aspect_ratios)
    ratio_vals_sq = torch.tensor(aspect_ratios).repeat(num_scales).sqrt()
    # view -> repeat -> view to simulate numpy.tile
    scale_vals = torch.tensor(scale_vals).view(-1, 1).repeat(1, len(aspect_ratios)).view(-1, 1)
    wh = torch.stack([torch.ones(num_anchors) * ratio_vals_sq, torch.ones(num_anchors) / ratio_vals_sq], 1)
    lt = -0.5 * wh * scale_vals
    rb = 0.5 * wh * scale_vals
    base_offsets = torch.stack([lt, rb], 1).float()  # [num_anchors, 4]
    base_offsets = base_offsets.view(-1, 1, 1, 4)  # [num_anchors, 1, 1, 4]
    # generate anchor boxes for all given strides
    all_anchors = []
    for stride in strides:
        y, x = torch.meshgrid([torch.arange(stride // 2, image_size[i], stride) for i in range(2)])
        xyxy = torch.stack((x, y, x, y), 2).unsqueeze(0)
        # permute to match TF EffDet anchors order after reshape
        anchors = (xyxy + base_offsets * stride).permute(1, 2, 0, 3).reshape(-1, 4)
        all_anchors.append(anchors)
    all_anchors = torch.cat(all_anchors)
    # clip boxes to image. Not sure if we really need to clip them
    # clip_bboxes(all_anchors, image_size)
    return all_anchors, num_anchors


def generate_targets(anchors, batch_gt_boxes, num_classes, matched_iou=0.5, unmatched_iou=0.4):
    # type: (Tensor, Tensor, int, float, float) -> Tuple[Tensor, Tensor, Tensor]
    """Generate targets for regression and classification
    
    Based on IoU between anchor and true bounding box there are three types of anchor boxes
    1) IoU >= matched_iou: Highest similarity. Matched/Positive. Mask value is 1
    2) matched_iou > IoU >= unmatched_iou: Medium similarity. Ignored. Mask value is -1
    3) unmatched_iou > IoU: Lowest similarity. Unmatched/Negative. Mask value is 0

    This function works on whole batch of images at once and is very efficient
    Args:
        anchors (torch.Tensor): all anchors on a single image. shape [N, 4]
        batch_gt_boxes (torch.Tensor): all ground truth bounding boxes and classes for the batch. shape [BS, N, 5]
            classes are expected to be in the last column.
            bboxes are in `ltrb` format!
        num_classes (int): number of classes. needed for one-hot encoding labels
        matched_iou (float): see above
        unmatched_iou (float): see above

    Returns:
        box_target, cls_target, matches_mask

    """
    anchors = anchors.to(batch_gt_boxes)  # change device & type if needed

    batch_gt_boxes, batch_gt_classes = batch_gt_boxes.split(4, dim=2)
    overlap = batch_box_iou(anchors, batch_gt_boxes)

    # Keep best box per anchor
    overlap, indices = overlap.max(-1)
    gathered_gt_boxes = batch_gt_boxes.gather(1, indices[..., None].expand(-1, -1, 4))
    box_target = box2delta(gathered_gt_boxes, anchors[None])

    # There are three types of anchors.
    # matched (with objects), unmatched (with background), and in between (which should be ignored)
    IGNORED_VALUE = -1
    matches_mask = torch.ones_like(overlap, dtype=torch.long) * IGNORED_VALUE
    UNMATCHED_VALUE = torch.tensor(0).to(matches_mask)
    MATCHED_VALUE = torch.tensor(1).to(matches_mask)
    matches_mask[overlap < unmatched_iou] = UNMATCHED_VALUE  # background
    matches_mask[overlap >= matched_iou] = MATCHED_VALUE  # foreground

    # Generate one-hot-encoded target classes
    bs, num_anchors = batch_gt_boxes.size(0), anchors.size(0)
    cls_target = torch.zeros(
        (bs, num_anchors, num_classes + 1), device=batch_gt_classes.device, dtype=batch_gt_classes.dtype
    )
    gathered_gt_classes = batch_gt_classes.gather(1, indices[..., None]).long()
    # set background to last class for scatter
    gathered_gt_classes[overlap < unmatched_iou] = torch.tensor(num_classes).to(gathered_gt_classes)
    cls_target.scatter_(2, gathered_gt_classes, 1)
    cls_target = cls_target[..., :num_classes]  # remove background class from one-hot

    return box_target, cls_target, matches_mask


# copied from torchvision
def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
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


# TODO: cover this with tests
# jit actually makes it slower for fp16 and results are different!
# FIXME: check it after 1.6 release. maybe they will fix JIT by that time
# @torch.jit.script
def decode(
    batch_cls_head,
    batch_box_head,
    anchors,
    img_shapes=None,
    score_threshold=0.0,
    max_detection_points=3000,  # 3840
    max_detection_per_image=100,
    iou_threshold=0.5,
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, int, int, float)->Tensor
    """
    Decodes raw outputs of a model for easy visualization of bboxes

    Args:
        batch_cls_head (torch.Tensor): shape [BS, *, NUM_CLASSES]
        batch_box_head (torch.Tensor): shape [BS, *, 4]
        anchors (torch.Tensor): shape [*, 4]
        img_shapes (torch.Tensor): if given clips predicted bboxes to img height and width. Shape [BS, 2] or [2,]
        score_threshold (float): minimum score threshold to consider object detected. Large values give slight speedup
            but decrease mean recall for small objects
        max_detection_points (int): Maximum number of bboxes to consider for NMS for one image.
            As of 07.20 in TenorRT 7 there is a hardcoded limit 3840 on this value. Make sure the default is lower than it
        max_detection_per_image (int): Maximum number of bboxes to return per image
        iou_threshold (float): iou_threshold for Non Maximum Supression

    Returns:
        torch.Tensor with bboxes, scores and classes
            shape [BS, MAX_DETECTION_PER_IMAGE, 6].
            bboxes in 'ltrb' format. If img_shape is not given they are NOT CLIPPED (!)
    """

    batch_size = batch_cls_head.size(0)
    num_classes = batch_cls_head.size(-1)

    # It's much faster to calculate topk once for full batch here rather than doing it inside loop
    # In TF The same bbox may belong to two different objects
    # select `max_detection_points` scores and corresponding bboxes
    scores_topk_all, cls_topk_indices_all = torch.topk(
        batch_cls_head.view(batch_size, -1), k=max_detection_points
    )
    indices_all = cls_topk_indices_all // num_classes
    classes_all = (cls_topk_indices_all % num_classes).float()  # turn to float for onnx export

    # applying sigmoid after topk is slightly faster than applying before
    scores_topk_all = scores_topk_all.sigmoid()  # logits -> scores

    # Gather corresponding bounding boxes & anchors, then regress and clip
    # offset for indices to match boxes after reshape
    offset = torch.arange(batch_size, device=indices_all.device).unsqueeze(1) * max_detection_points
    box_topk_all = batch_box_head.view(-1, 4)[(indices_all + offset).view(-1)].view(batch_size, -1, 4)
    # index anchors without offset because they are the same for all images in batch
    anchors_topk_all = anchors.to(box_topk_all)[indices_all.view(-1)].view(batch_size, -1, 4)
    regressed_boxes_all = delta2box(box_topk_all, anchors_topk_all)
    if img_shapes is not None:
        regressed_boxes_all = clip_bboxes_batch(regressed_boxes_all, img_shapes)

    # prepare output tensors
    device_dtype = {"device": regressed_boxes_all.device, "dtype": regressed_boxes_all.dtype}
    out_scores = torch.zeros((batch_size, max_detection_per_image), **device_dtype)
    out_boxes = torch.zeros((batch_size, max_detection_per_image, 4), **device_dtype)
    out_classes = torch.zeros((batch_size, max_detection_per_image), **device_dtype)

    for batch in range(batch_size):
        scores_topk = scores_topk_all[batch]
        # additionally filter prediction with low confidence
        valid_mask = scores_topk.ge(score_threshold)
        scores_topk = scores_topk[valid_mask]
        classes = classes_all[batch, valid_mask]
        regressed_boxes = regressed_boxes_all[batch, valid_mask]

        # apply NMS
        nms_idx = batched_nms(regressed_boxes, scores_topk, classes, iou_threshold)
        nms_idx = nms_idx[: min(len(nms_idx), max_detection_per_image)]

        # select suppressed bboxes
        im_scores = scores_topk[nms_idx]
        im_classes = classes[nms_idx]
        im_bboxes = regressed_boxes[nms_idx]
        im_classes += 1  # back to class idx with background class = 0

        out_scores[batch, : im_scores.size(0)] = im_scores
        out_classes[batch, : im_classes.size(0)] = im_classes
        out_boxes[batch, : im_bboxes.size(0)] = im_bboxes
        # no need to pad because it's already padded with 0's

    return torch.cat([out_boxes, out_scores.unsqueeze(-1), out_classes.unsqueeze(-1)], dim=2)
