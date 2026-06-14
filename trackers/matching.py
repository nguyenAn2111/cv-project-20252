"""Matching utilities for project-local ByteTrack."""

import numpy as np

try:
    import lap
except ImportError:  # pragma: no cover
    lap = None


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if lap is not None:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = np.asarray([[ix, mx] for ix, mx in enumerate(x) if mx >= 0], dtype=int)
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
        return matches, unmatched_a, unmatched_b

    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_a = set(range(cost_matrix.shape[0]))
    unmatched_b = set(range(cost_matrix.shape[1]))
    for row, col in zip(rows, cols):
        if cost_matrix[row, col] <= thresh:
            matches.append([row, col])
            unmatched_a.discard(row)
            unmatched_b.discard(col)
    return np.asarray(matches, dtype=int), np.asarray(sorted(unmatched_a)), np.asarray(sorted(unmatched_b))


def bbox_iou_matrix(boxes_a, boxes_b):
    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)
    ious = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return ious

    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])

    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area_a[:, None] + area_b[None, :] - inter
    valid = union > 0
    ious[valid] = inter[valid] / union[valid]
    return ious


def iou_distance(atracks, btracks):
    if (atracks and isinstance(atracks[0], np.ndarray)) or (btracks and isinstance(btracks[0], np.ndarray)):
        boxes_a = atracks
        boxes_b = btracks
    else:
        boxes_a = [track.xyxy for track in atracks]
        boxes_b = [track.xyxy for track in btracks]
    return 1.0 - bbox_iou_matrix(boxes_a, boxes_b)


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix

    iou_sim = 1.0 - cost_matrix
    det_scores = np.asarray([det.score for det in detections], dtype=np.float32)
    fused_sim = iou_sim * det_scores[None, :]
    return 1.0 - fused_sim
