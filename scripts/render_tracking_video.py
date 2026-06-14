"""
Render YOLO + ByteTrack tracking results to a video.

The script keeps the existing quantitative baseline untouched and adds a visual
demo layer:

- input: raw video
- input: YOLO model and ByteTrack YAML
- output: video with bbox, stable track color, and track ID
- optional label details: class name and confidence score
- overlay: current frame ID at the top-left corner
- optional output: MOT-format prediction txt for later evaluation

Example:
    python scripts/render_tracking_video.py \
        --model yolo11n.pt \
        --video dataset/raw/Vehicle_Tracking/VNTraffic/VNTraffic_Original-video.mp4 \
        --tracker configs/bytetrack_baseline.yaml \
        --out outputs/visualizations/vntraffic_baseline.mp4 \
        --save_mot outputs/visualizations/vntraffic_baseline.txt \
        --vehicle_coco
"""

from pathlib import Path
import argparse
import os
import tempfile

import cv2
from tqdm import tqdm

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-cache"),
)

from ultralytics import YOLO


COCO_VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


def parse_class_ids(value):
    if value is None:
        return None

    class_ids = []
    for item in value:
        for part in item.split(","):
            part = part.strip()
            if part:
                class_ids.append(int(part))

    return class_ids or None


def stable_color(track_id):
    """Return a deterministic BGR color for each track ID."""
    track_id = int(track_id)
    r = (37 * track_id + 89) % 255
    g = (17 * track_id + 149) % 255
    b = (29 * track_id + 211) % 255

    # Avoid very dark colors on traffic footage.
    return (
        int(max(80, b)),
        int(max(80, g)),
        int(max(80, r)),
    )


def ensure_can_write(path, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Use --overwrite or choose another path."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def open_video_writer(output_path, fps, width, height):
    suffix = output_path.suffix.lower()
    if suffix == ".avi":
        codec = "XVID"
    else:
        codec = "mp4v"

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    return writer


def build_label(track_id, cls_id=None, conf=None, names=None, show_class=False, show_conf=False):
    parts = [f"ID {int(track_id)}"]

    if show_conf and conf is not None:
        parts.append(f"{float(conf):.2f}")

    if show_class and cls_id is not None:
        cls_id = int(cls_id)
        if names and cls_id in names:
            parts.append(str(names[cls_id]))
        else:
            parts.append(f"cls {cls_id}")

    return " ".join(parts)


def draw_frame_id(frame, frame_id):
    height, width = frame.shape[:2]
    label = f"Frame {int(frame_id)}"
    font_scale = max(0.65, min(width, height) / 1400)
    thickness = max(1, round(min(width, height) / 700))
    pad_x = 9
    pad_y = 7
    margin = 12

    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )

    bg_x1 = margin
    bg_y1 = margin
    bg_x2 = min(width - 1, bg_x1 + text_w + pad_x * 2)
    bg_y2 = min(height - 1, bg_y1 + text_h + baseline + pad_y * 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(
        frame,
        label,
        (bg_x1 + pad_x, bg_y2 - baseline - pad_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_track(frame, box, track_id, label, color):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width - 1, round(x2))))
    y2 = int(max(0, min(height - 1, round(y2))))

    thickness = max(2, round(min(width, height) / 500))
    font_scale = max(0.55, min(width, height) / 1600)
    font_thickness = max(1, thickness - 1)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness,
    )

    pad_x = 5
    pad_y = 4
    label_h = text_h + baseline + pad_y * 2

    text_bg_x1 = x1
    text_bg_y1 = max(0, y1 - label_h)
    text_bg_x2 = min(width - 1, x1 + text_w + pad_x * 2)
    text_bg_y2 = y1

    if text_bg_y2 - text_bg_y1 < label_h:
        text_bg_y1 = y1
        text_bg_y2 = min(height - 1, y1 + label_h)

    cv2.rectangle(
        frame,
        (text_bg_x1, text_bg_y1),
        (text_bg_x2, text_bg_y2),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (text_bg_x1 + pad_x, text_bg_y2 - baseline - pad_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA,
    )


def mot_line(frame_id, track_id, box, score, cls_id=None):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    cls_value = -1 if cls_id is None else int(cls_id)
    return (
        f"{frame_id},{int(track_id)},"
        f"{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
        f"{float(score):.4f},{cls_value},-1,-1"
    )


def render_tracking_video(
    model_path,
    video_path,
    tracker_cfg,
    output_video,
    save_mot=None,
    conf=0.25,
    iou=0.5,
    classes=None,
    show_class=False,
    show_conf=False,
    max_frames=None,
    overwrite=False,
):
    output_video = ensure_can_write(output_video, overwrite)
    save_mot = ensure_can_write(save_mot, overwrite) if save_mot else None

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
        total_frames = min(total_frames, max_frames)

    writer = open_video_writer(output_video, fps, width, height)
    mot_lines = []

    frame_id = 0
    progress = tqdm(total=total_frames, desc="Rendering tracking video")

    try:
        while frame_id < total_frames:
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
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, track_id, score, cls_id in zip(boxes, track_ids, scores, cls_ids):
                    color = stable_color(track_id)
                    label = build_label(
                        track_id=track_id,
                        cls_id=cls_id,
                        conf=score,
                        names=result.names,
                        show_class=show_class,
                        show_conf=show_conf,
                    )
                    draw_track(frame, box, track_id, label, color)

                    if save_mot:
                        line = mot_line(frame_id, track_id, box, score, cls_id)
                        if line:
                            mot_lines.append(line)

            draw_frame_id(frame, frame_id)
            writer.write(frame)
            frame_id += 1
            progress.update(1)
    finally:
        progress.close()
        cap.release()
        writer.release()

    if save_mot:
        save_mot.write_text("\n".join(mot_lines), encoding="utf-8")

    print(f"\nVideo saved to: {output_video}")
    if save_mot:
        print(f"MOT prediction saved to: {save_mot}")
        print(f"Prediction lines: {len(mot_lines)}")
    print(f"Rendered frames: {frame_id}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render YOLO + ByteTrack tracking visualization video."
    )
    parser.add_argument("--model", required=True, help="Path to YOLO model, e.g. yolo11n.pt")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="ByteTrack YAML path")
    parser.add_argument("--out", required=True, help="Output video path (.mp4 or .avi)")
    parser.add_argument("--save_mot", default=None, help="Optional MOT prediction txt path")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU threshold")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class IDs, accepts space or comma separated values, e.g. 2 3 5 7",
    )
    parser.add_argument(
        "--vehicle_coco",
        action="store_true",
        help=(
            "Keep only COCO vehicle classes: car, motorcycle, bus, truck. "
            "Do not use this with a custom one-class vehicle model."
        ),
    )
    parser.add_argument("--show_class", action="store_true", help="Show class name in track label")
    parser.add_argument("--show_conf", action="store_true", help="Show confidence score in track label")
    parser.add_argument("--max_frames", type=int, default=None, help="Render only first N frames")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.vehicle_coco:
        classes = COCO_VEHICLE_CLASSES
    else:
        classes = parse_class_ids(args.classes)

    render_tracking_video(
        model_path=args.model,
        video_path=args.video,
        tracker_cfg=args.tracker,
        output_video=args.out,
        save_mot=args.save_mot,
        conf=args.conf,
        iou=args.iou,
        classes=classes,
        show_class=args.show_class,
        show_conf=args.show_conf,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
    )
