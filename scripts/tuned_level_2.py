"""
Run level 2 custom tracking end-to-end.

Level 2 uses YOLO only as a detector and runs a project-local ByteTrack variant
from trackers/custom_byte_tracker.py. This makes the tracker implementation
separable from the Ultralytics baseline and suitable for direct comparison.
"""

from pathlib import Path
from types import SimpleNamespace
import argparse
import sys

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO

try:
    from .evaluate_mot import evaluate_mot
    from .render_tracking_video import (
        build_label,
        draw_frame_id,
        draw_track,
        ensure_can_write,
        open_video_writer,
        stable_color,
    )
    from .tuned_level_1 import COCO_VEHICLE_CLASSES, parse_class_ids
except ImportError:
    from evaluate_mot import evaluate_mot
    from render_tracking_video import (
        build_label,
        draw_frame_id,
        draw_track,
        ensure_can_write,
        open_video_writer,
        stable_color,
    )
    from tuned_level_1 import COCO_VEHICLE_CLASSES, parse_class_ids

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trackers import Level2BYTETracker

DEFAULT_MODEL = PROJECT_ROOT / "yolo11n.pt"
DEFAULT_VIDEO = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "Vehicle_Tracking"
    / "VNTraffic"
    / "VNTraffic_Original-video.mp4"
)
DEFAULT_GT = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "Vehicle_Tracking"
    / "VNTraffic"
    / "VNTraffic_GroundTruth.txt"
)
DEFAULT_TRACKER = PROJECT_ROOT / "configs" / "custom_tracker.yaml"
DEFAULT_PRED = PROJECT_ROOT / "outputs" / "level2" / "vntraffic_level2_yolo11n_custom_bytetrack.txt"
DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "level2" / "tables" / "vntraffic_level2_metrics.csv"
DEFAULT_VIDEO_OUT = PROJECT_ROOT / "outputs" / "level2" / "vntraffic_level2_yolo11n_custom_bytetrack.mp4"


def load_tracker_args(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data.setdefault("track_high_thresh", 0.22)
    data.setdefault("track_low_thresh", 0.06)
    data.setdefault("new_track_thresh", 0.30)
    data.setdefault("track_buffer", 55)
    data.setdefault("match_thresh", 0.78)
    data.setdefault("fuse_score", True)
    return SimpleNamespace(**data)


def resolve_classes(classes=None, all_classes=False):
    if all_classes:
        return None
    if classes is not None:
        return parse_class_ids(classes)
    return COCO_VEHICLE_CLASSES


def filter_boxes(boxes, min_box_area=0.0, max_aspect_ratio=None):
    if boxes is None or len(boxes) == 0:
        return boxes

    xyxy = boxes.xyxy
    widths = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0])
    heights = np.maximum(0.0, xyxy[:, 3] - xyxy[:, 1])
    areas = widths * heights

    keep = areas >= float(min_box_area)

    if max_aspect_ratio is not None and max_aspect_ratio > 0:
        ratios = np.maximum(widths / np.maximum(1.0, heights), heights / np.maximum(1.0, widths))
        keep = keep & (ratios <= float(max_aspect_ratio))

    return boxes[keep]


def mot_line(frame_id, track):
    x1, y1, x2, y2, track_id, score, cls_id, _idx = track
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None

    return (
        f"{frame_id},{int(track_id)},"
        f"{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},"
        f"{float(score):.4f},{int(cls_id)},-1,-1"
    )


def export_level_2_mot(
    model_path,
    video_path,
    tracker_cfg,
    output_txt,
    output_video=None,
    conf=0.08,
    iou=0.5,
    classes=None,
    show_class=False,
    show_conf=False,
    max_frames=None,
    overwrite=False,
):
    tracker_args = load_tracker_args(tracker_cfg)
    tracker = Level2BYTETracker(args=tracker_args)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, int(max_frames))

    output_txt = Path(output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    if output_video is not None:
        output_video = ensure_can_write(output_video, overwrite)
        writer = open_video_writer(output_video, fps, width, height)

    min_box_area = float(getattr(tracker_args, "min_box_area", 0.0))
    max_aspect_ratio = getattr(tracker_args, "max_aspect_ratio", None)
    mot_lines = []
    frame_id = 0
    progress = tqdm(total=total_frames, desc="Running YOLO + Level 2 tracker")

    try:
        while frame_id < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            result = model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                classes=classes,
                verbose=False,
            )[0]
            boxes = result.boxes.cpu().numpy()
            boxes = filter_boxes(
                boxes,
                min_box_area=min_box_area,
                max_aspect_ratio=max_aspect_ratio,
            )
            tracks = tracker.update(boxes, frame, None)

            for track in tracks:
                line = mot_line(frame_id, track)
                if line:
                    mot_lines.append(line)

                if writer is not None:
                    x1, y1, x2, y2, track_id, score, cls_id, _idx = track
                    color = stable_color(track_id)
                    label = build_label(
                        track_id=track_id,
                        cls_id=cls_id,
                        conf=score,
                        names=result.names,
                        show_class=show_class,
                        show_conf=show_conf,
                    )
                    draw_track(frame, (x1, y1, x2, y2), track_id, label, color)

            if writer is not None:
                draw_frame_id(frame, frame_id)
                writer.write(frame)

            frame_id += 1
            progress.update(1)
    finally:
        progress.close()
        cap.release()
        if writer is not None:
            writer.release()

    output_txt.write_text("\n".join(mot_lines), encoding="utf-8")

    print(f"\nLevel 2 MOT prediction saved to: {output_txt}")
    print(f"Prediction lines: {len(mot_lines)}")
    if output_video is not None:
        print(f"Level 2 visualization saved to: {output_video}")
    print(f"Processed frames: {frame_id}")


def run_level_2(
    model=str(DEFAULT_MODEL),
    video=str(DEFAULT_VIDEO),
    gt=str(DEFAULT_GT),
    tracker=str(DEFAULT_TRACKER),
    pred=str(DEFAULT_PRED),
    metrics=str(DEFAULT_METRICS),
    vis=str(DEFAULT_VIDEO_OUT),
    conf=0.08,
    iou=0.5,
    name="level2",
    classes=None,
    all_classes=False,
    show_class=False,
    show_conf=False,
    max_frames=None,
    overwrite=False,
    skip_export=False,
    skip_eval=False,
    skip_render=False,
):
    selected_classes = resolve_classes(classes=classes, all_classes=all_classes)

    if not skip_export:
        print("===== Level 2: export MOT prediction =====")
        export_level_2_mot(
            model_path=model,
            video_path=video,
            tracker_cfg=tracker,
            output_txt=pred,
            output_video=None if skip_render else vis,
            conf=conf,
            iou=iou,
            classes=selected_classes,
            show_class=show_class,
            show_conf=show_conf,
            max_frames=max_frames,
            overwrite=overwrite,
        )
    else:
        print(f"===== Level 2: reuse MOT prediction at {pred} =====")

    if not skip_eval:
        print("\n===== Level 2: evaluate MOT metrics =====")
        evaluate_mot(
            gt_path=gt,
            pred_path=pred,
            output_csv=metrics,
            name=name,
        )
    else:
        print("\n===== Level 2: skip MOT evaluation =====")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run level 2 custom ByteTrack export, evaluation, and visualization."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO model path")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Input video path")
    parser.add_argument("--gt", default=str(DEFAULT_GT), help="MOT ground-truth txt path")
    parser.add_argument("--tracker", default=str(DEFAULT_TRACKER), help="Level 2 tracker YAML path")
    parser.add_argument("--pred", default=str(DEFAULT_PRED), help="Output MOT prediction txt")
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS), help="Output metrics CSV")
    parser.add_argument("--vis", default=str(DEFAULT_VIDEO_OUT), help="Output visualization video")
    parser.add_argument("--conf", type=float, default=0.08, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU threshold")
    parser.add_argument("--name", default="level2", help="Experiment name written to metrics table")
    parser.add_argument("--skip_export", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class IDs, accepts space or comma separated values, e.g. 2 3 5 7",
    )
    parser.add_argument("--all_classes", action="store_true")
    parser.add_argument("--show_class", action="store_true")
    parser.add_argument("--show_conf", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_level_2(
        model=args.model,
        video=args.video,
        gt=args.gt,
        tracker=args.tracker,
        pred=args.pred,
        metrics=args.metrics,
        vis=args.vis,
        conf=args.conf,
        iou=args.iou,
        name=args.name,
        classes=args.classes,
        all_classes=args.all_classes,
        show_class=args.show_class,
        show_conf=args.show_conf,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
        skip_export=args.skip_export,
        skip_eval=args.skip_eval,
        skip_render=args.skip_render,
    )


if __name__ == "__main__":
    main()
