"""
Multi-Object Tracking Evaluation Script
=======================================

This script evaluates multi-object tracking results by comparing predicted
tracking outputs against ground-truth annotations in MOTChallenge-style format.

Both the ground-truth file and the prediction file are expected to contain
bounding boxes in the following format:

```
frame_id, object_id, x, y, width, height, confidence, class, visibility, ...
```

Only the first six fields are required and used during evaluation:

```
frame_id, object_id, x, y, width, height
```

The evaluation process consists of the following steps:

```
1. Read ground-truth and prediction files.
2. Extract frame IDs, object IDs, and bounding boxes.
3. For each frame, compute the IoU between ground-truth boxes and predicted
   boxes.
4. Convert IoU values into a distance matrix using:

       distance = 1 - IoU

5. Reject invalid matches whose IoU is lower than 0.5.
6. Update a MOT accumulator with matched and unmatched objects.
7. Compute standard MOT metrics using the motmetrics library.
```

The reported metrics include:

```
- MOTA: Multi-Object Tracking Accuracy
- MOTP: Multi-Object Tracking Precision
- IDF1: Identity F1 Score
- Precision and Recall
- False Positives
- False Negatives
- ID Switches
- Fragmentations
- Mostly Tracked and Mostly Lost targets
```

The final evaluation summary is printed to the terminal and can optionally be
saved as a CSV file for reporting and comparison between different tracking
pipelines.

Typical usage:

```
python scripts/evaluate_mot.py \
    --gt path/to/ground_truth.txt \
    --pred path/to/prediction.txt \
    --out_csv outputs/baseline/metrics.csv
```
DangTran
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import motmetrics as mm


def read_mot_file(path):
    """
    Đọc file MOT format:
    frame, id, x, y, w, h, conf, class, visibility, ...
    Chỉ lấy 6 trường đầu.
    """
    path = Path(path)
    records = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) < 6:
                continue

            frame_id = int(float(parts[0]))
            obj_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            if w <= 0 or h <= 0:
                continue

            records.append(
                {
                    "frame": frame_id,
                    "id": obj_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )

    return pd.DataFrame(records)


def iou_matrix(gt_boxes, pred_boxes):
    """
    Tạo ma trận distance = 1 - IoU. 
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.empty((len(gt_boxes), len(pred_boxes)))

    distances = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=float)

    for i, gt in enumerate(gt_boxes):
        gx, gy, gw, gh = gt
        gx2 = gx + gw
        gy2 = gy + gh

        for j, pr in enumerate(pred_boxes):
            px, py, pw, ph = pr
            px2 = px + pw
            py2 = py + ph

            inter_x1 = max(gx, px)
            inter_y1 = max(gy, py)
            inter_x2 = min(gx2, px2)
            inter_y2 = min(gy2, py2)

            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)

            inter_area = inter_w * inter_h
            gt_area = gw * gh
            pred_area = pw * ph

            union_area = gt_area + pred_area - inter_area

            if union_area <= 0:
                iou = 0.0
            else:
                iou = inter_area / union_area

            # motmetrics dùng distance, distance nhỏ hơn là tốt hơn
            # Nếu IoU < 0.5 thì không cho match
            if iou < 0.5:
                distances[i, j] = np.nan
            else:
                distances[i, j] = 1.0 - iou

    return distances


def evaluate_mot(gt_path, pred_path, output_csv=None, name="baseline"):
    gt_df = read_mot_file(gt_path)
    pred_df = read_mot_file(pred_path)

    print("Ground truth boxes:", len(gt_df))
    print("Prediction boxes:", len(pred_df))

    acc = mm.MOTAccumulator(auto_id=True)

    all_frames = sorted(set(gt_df["frame"].unique()) | set(pred_df["frame"].unique()))

    for frame_id in all_frames:
        gt_frame = gt_df[gt_df["frame"] == frame_id]
        pred_frame = pred_df[pred_df["frame"] == frame_id]

        gt_ids = gt_frame["id"].tolist()
        pred_ids = pred_frame["id"].tolist()

        gt_boxes = gt_frame[["x", "y", "w", "h"]].values
        pred_boxes = pred_frame[["x", "y", "w", "h"]].values

        distances = iou_matrix(gt_boxes, pred_boxes)

        acc.update(gt_ids, pred_ids, distances)

    mh = mm.metrics.create()

    metrics = [
        "num_frames",
        "mota",
        "motp",
        "idf1",
        "precision",
        "recall",
        "num_objects",
        "num_predictions",
        "num_false_positives",
        "num_misses",
        "num_switches",
        "num_fragmentations",
        "mostly_tracked",
        "mostly_lost",
    ]

    summary = mh.compute(acc, metrics=metrics, name=name)

    print("\n===== MOT Evaluation Result =====")
    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    ))

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_csv)
        print(f"\nMetric CSV was saved at: {output_csv}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--name", default="baseline")

    args = parser.parse_args()

    evaluate_mot(
        gt_path=args.gt,
        pred_path=args.pred,
        output_csv=args.out_csv,
        name=args.name,
    )
