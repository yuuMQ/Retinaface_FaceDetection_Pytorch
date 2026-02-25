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

def matrix_iof(a, b):
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
    # calculate centers of ground truth boxes
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]

    # Normalize the centers with the size of the priors and variances
    g_cxcy /= (variances[0] * priors[:, 2:])

    # Calculate the sizes of the ground truth boxes
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1] # Transform the scale with log

    # Concatenate normalized centers and sizes to get the encoded boxes
    encoded_boxes = torch.cat([g_cxcy, g_wh], dim=1) # Concatenation along the last dimension

    return encoded_boxes # [num_priors, 4]

def encode_landmarks(matched, priors, variances):
    # Reshape matched landmarks into 5 points with 2 coordinates each (x, y)
    matched = matched.view(matched.size(0), 5, 2)

    # Extract priors' center coordinates (cx, cy) with width, height (w, h)
    priors_cx = priors[:, 0].view(-1, 1)
    priors_cy = priors[:, 1].view(-1, 1)
    priors_w = priors[:, 2].view(-1, 1)
    priors_h = priors[:, 3].view(-1, 1)

    # Compute the center offset between matched and prior landmarks
    g_cxcy = matched - torch.stack([priors_cx, priors_cy], dim=2)

    # Normalize by the variance-scaled width and height
    g_cxcy /= variances[0] * torch.stack([priors_w, priors_h], dim=2)

    # Flatten the landmark coordinates back to [num_priors, 10]
    g_cxcy = g_cxcy.view(g_cxcy.size(0), -1)

    return g_cxcy


def decode(loc, priors, variances):
    # Compute centers of predicted boxes
    cxcy = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]

    # Compute widths and heights of predicted boxes
    wh = priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])

    # Convert center, size to corner coordinates
    boxes = torch.empty_like(loc)
    boxes[:, :2] = cxcy - wh / 2 # x_min, y_min
    boxes[:, 2:] = cxcy + wh / 2 # x_max, y_max

    return boxes

def decode_landmarks(predictions, priors, variances):
    # Reshape predictions to [num_priors, 5, 2] to handle each pair (x, y) in a batch
    predictions = predictions.view(predictions.size(0), 5, 2)

    # Perform the same operation on all landmark pairs at once
    landmarks = priors[:, :2].unsqueeze(1) + predictions * variances[0] * priors[:, 2:].unsqueeze(1)

    # Flatten back to [num_priors, 10]
    landmarks = landmarks.view(landmarks.size(0), -1)

    return landmarks


def log_sum_exp(x):
    return torch.logsumexp(x, dim=1, keepdim=True)

def nms(dets, threshold):
    # dets: array of detections with each row: [x1, y1, x2, y2, score]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep