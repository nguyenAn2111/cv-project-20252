"""Level 2 ByteTrack variant with project-local association logic.

The tracker keeps the local ByteTrack lifecycle and replaces the first-stage
association cost with a weighted cost that also accounts for center
displacement, bbox shape changes, detection confidence, and class consistency.
"""

from __future__ import annotations

import numpy as np

from .track import BYTETracker
from . import matching


def _xyxy(items):
    return np.asarray([item.xyxy for item in items], dtype=np.float32)


def _center_distance(tracks, detections):
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    track_boxes = _xyxy(tracks)
    det_boxes = _xyxy(detections)

    track_centers = np.column_stack(
        (
            (track_boxes[:, 0] + track_boxes[:, 2]) / 2.0,
            (track_boxes[:, 1] + track_boxes[:, 3]) / 2.0,
        )
    )
    det_centers = np.column_stack(
        (
            (det_boxes[:, 0] + det_boxes[:, 2]) / 2.0,
            (det_boxes[:, 1] + det_boxes[:, 3]) / 2.0,
        )
    )

    track_sizes = np.column_stack(
        (
            np.maximum(1.0, track_boxes[:, 2] - track_boxes[:, 0]),
            np.maximum(1.0, track_boxes[:, 3] - track_boxes[:, 1]),
        )
    )
    det_sizes = np.column_stack(
        (
            np.maximum(1.0, det_boxes[:, 2] - det_boxes[:, 0]),
            np.maximum(1.0, det_boxes[:, 3] - det_boxes[:, 1]),
        )
    )

    deltas = track_centers[:, None, :] - det_centers[None, :, :]
    normalizers = (track_sizes[:, None, :] + det_sizes[None, :, :]) / 2.0
    normalized = deltas / np.maximum(1.0, normalizers)
    return np.clip(np.linalg.norm(normalized, axis=2) / 2.0, 0.0, 1.0)


def _shape_distance(tracks, detections):
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    track_boxes = _xyxy(tracks)
    det_boxes = _xyxy(detections)

    track_w = np.maximum(1.0, track_boxes[:, 2] - track_boxes[:, 0])
    track_h = np.maximum(1.0, track_boxes[:, 3] - track_boxes[:, 1])
    det_w = np.maximum(1.0, det_boxes[:, 2] - det_boxes[:, 0])
    det_h = np.maximum(1.0, det_boxes[:, 3] - det_boxes[:, 1])

    track_area = track_w * track_h
    det_area = det_w * det_h
    track_ratio = track_w / track_h
    det_ratio = det_w / det_h

    area_cost = np.abs(np.log(track_area[:, None] / det_area[None, :]))
    ratio_cost = np.abs(np.log(track_ratio[:, None] / det_ratio[None, :]))
    return np.clip((area_cost + ratio_cost) / 4.0, 0.0, 1.0)


def _class_penalty(tracks, detections, penalty):
    if len(tracks) == 0 or len(detections) == 0 or penalty <= 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    track_cls = np.asarray([track.cls for track in tracks])
    det_cls = np.asarray([det.cls for det in detections])
    return (track_cls[:, None] != det_cls[None, :]).astype(np.float32) * penalty


class Level2BYTETracker(BYTETracker):
    """ByteTrack variant tuned for vehicle MOT experiments."""

    def get_dists(self, tracks, detections):
        iou_cost = matching.iou_distance(tracks, detections)
        if iou_cost.size == 0:
            return iou_cost

        center_weight = float(getattr(self.args, "center_weight", 0.12))
        shape_weight = float(getattr(self.args, "shape_weight", 0.06))
        score_weight = float(getattr(self.args, "score_weight", 0.08))
        class_penalty = float(getattr(self.args, "class_mismatch_penalty", 0.20))

        center_weight = max(0.0, min(0.45, center_weight))
        shape_weight = max(0.0, min(0.30, shape_weight))
        score_weight = max(0.0, min(0.30, score_weight))
        iou_weight = max(0.0, 1.0 - center_weight - shape_weight - score_weight)

        cost = (
            iou_weight * iou_cost
            + center_weight * _center_distance(tracks, detections)
            + shape_weight * _shape_distance(tracks, detections)
        )

        if score_weight > 0:
            det_scores = np.asarray([det.score for det in detections], dtype=np.float32)
            score_cost = 1.0 - det_scores[None, :]
            cost += score_weight * score_cost

        cost += _class_penalty(tracks, detections, class_penalty)
        return np.clip(cost, 0.0, 1.0)
