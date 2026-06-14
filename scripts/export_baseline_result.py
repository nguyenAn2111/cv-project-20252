"""
Baseline Tracking Result Exporter
=================================

This script runs a YOLO-based object detector combined with the ByteTrack
multi-object tracking algorithm on an input video and exports the tracking
results in MOTChallenge-style text format.

For each video frame, the script performs object detection, associates detected
bounding boxes across consecutive frames using ByteTrack, assigns a unique track
ID to each object, and writes the final tracking results to a text file.

The exported file follows the MOT format:

```
frame_id, track_id, x, y, width, height, confidence, class, visibility
```

where:
- frame_id: zero-based frame index of the video.
- track_id: unique identity assigned by the tracker.
- x, y: top-left coordinate of the bounding box.
- width, height: size of the bounding box.
- confidence: detection confidence score returned by YOLO.
- class, visibility: unused fields, filled with -1 for compatibility.

This script is mainly used to generate baseline prediction results from the
original YOLO + ByteTrack pipeline. The output can later be compared with
ground-truth annotations using standard MOT evaluation metrics such as MOTA,
IDF1, precision, recall, false positives, false negatives, and ID switches.

Typical usage:

```
python scripts/export_baseline_result.py \
    --model yolo11n.pt \
    --video path/to/input_video.mp4 \
    --tracker bytetrack.yaml \
    --out outputs/baseline/predictions.txt \
    --conf 0.25 \
    --iou 0.5
```

Optional:
Use --vehicle_coco to keep only common COCO vehicle classes:
car, motorcycle, bus, and truck.
DangTran
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm


def export_baseline_mot(
    model_path: str,
    video_path: str,
    tracker_cfg: str,
    output_txt: str,
    conf: float = 0.25,
    iou: float = 0.5,
    classes=None,
):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_txt = Path(output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    frame_id = 0  # Zenodo VNTraffic bắt đầu từ frame 0

    for _ in tqdm(range(total_frames), desc="Running YOLO + ByteTrack"):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            source=frame,
            tracker=tracker_cfg,
            conf=conf,
            iou=iou,
            classes=classes,
            persist=True,
            verbose=False,
        )

        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()

            for box, track_id, score in zip(boxes_xyxy, track_ids, scores):
                x1, y1, x2, y2 = box

                w = x2 - x1
                h = y2 - y1

                if w <= 0 or h <= 0:
                    continue

                # MOT format:
                # frame, id, x, y, w, h, conf, class, visibility
                line = (
                    f"{frame_id},{track_id},"
                    f"{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                    f"{score:.4f},-1,-1,-1"
                )
                lines.append(line)

        frame_id += 1

    cap.release()

    with open(output_txt, "w") as f:
        f.write("\n".join(lines))

    print(f"\nPrediction MOT result at: {output_txt}")
    print(f"The number of prediction lines: {len(lines)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--vehicle_coco",
        action="store_true",
        help="Only COCO classes: car, motorcycle, bus, truck"
    )

    args = parser.parse_args()

    classes = [2, 3, 5, 7] if args.vehicle_coco else None

    export_baseline_mot(
        model_path=args.model,
        video_path=args.video,
        tracker_cfg=args.tracker,
        output_txt=args.out,
        conf=args.conf,
        iou=args.iou,
        classes=classes,
    )