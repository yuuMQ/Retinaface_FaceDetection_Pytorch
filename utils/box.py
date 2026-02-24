import numpy as np
import torch.nn as nn
import torch
from torchvision.ops import boxes


def xywh2xyxy(boxes):
    # [x, y, w, h] -> [x1, y1, x2, y2] | xy1 = top - left, xy2 = botton - right.
    y = boxes.clone() if isinstance(boxes, torch.Tensor) else np.copy(boxes)
    y[..., 0] = boxes[..., 0] - boxes[..., 2] / 2 # top left x
    y[..., 1] = boxes[..., 1] - boxes[..., 3] / 2 # top left y
    y[..., 2] = boxes[..., 0] + boxes[..., 2] / 2 # bottom right x
    y[..., 3] = boxes[..., 1] + boxes[..., 3] / 2 # bottom right y

    return y

def xyxy2xywh(boxes):
    # [x1, y1, x2, y2] -> [x, y, w, h]
    # w = x2 - x1 | h = y2 - y1 | (x = x1 + w / 2 | y = y1 + h / 2) Hoặc (x = (x1 + x2) / 2 | y = (y1 + y2) /2)
    y = boxes.clone() if isinstance(boxes, torch.Tensor) else np.copy(boxes)
    y[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2   # x_center
    y[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2  # y_center
    y[..., 2] = boxes[..., 2] - boxes[..., 0] # w
    y[..., 3] = boxes[..., 3] - boxes[..., 1] # h

    return y

def _box_inter_union(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # [N, M, 2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # [N, M, 2]

    wh = (right_bottom - left_top).clamp(min=0) # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

    union = area1[:, None] + area2 - inter

    return inter, union

def jaccard(boxes1, boxes2):
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou

def matrix_iou(a, b):
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)

    return area_i / np.maximum(area_a[:, np.newaxis], 1)

def match(overlap_threshold, gt_boxes, prior_boxes, variances, gt_labels, landmarks, loc_targets, conf_targets, landm_targets, batch_idx):
    # compute jaccard overlap between ground truth boxes and prior boxes
    overlaps = jaccard(gt_boxes, xywh2xyxy(prior_boxes))
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # Ignore ground truths with low overlap
    valid_grt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = best_prior_idx[valid_grt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        loc_targets[batch_idx] = 0
        conf_targets[batch_idx] = 0
        return

    # Find the best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    # Ensure every ground truth matches with its prior of max overlap
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matches = gt_boxes[best_truth_idx] # [num_priors, 4]
    conf = gt_labels[best_truth_idx] # [num_priors]
    conf[best_truth_overlap < overlap_threshold] = 0 # label as background
    loc = encode(matches, prior_boxes, variances)

    matches_landm = landmarks[best_truth_idx]
    landmarks = encode_landmarks(matches_landm, prior_boxes, variances)
    loc_targets[batch_idx] = loc # [num_priors, 4] encoded offsets to learn
    conf_targets[batch_idx] = conf # [num_priors] top class label for each prior

    landm_targets[batch_idx] = landmarks


def encode(matched, priors, variances):
    pass

def encode_landmarks(matched, priors, variances):
    pass

def decode(loc, priors, variances):
    pass

def decode_landmarks(predictions, priors, variances):
    pass

def log_sum_exp(x):
    pass

def nms(dets, threshold):
    pass