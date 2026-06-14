"""
Run tuned tracking level 1 end-to-end.

This script reuses the existing project utilities:
- export_baseline_result.py for YOLO + ByteTrack MOT prediction export
- evaluate_mot.py for MOT metrics
- render_tracking_video.py for visualization video rendering

It keeps the original baseline untouched and writes tuned outputs under
outputs/tuned by default.

Usage:
    Run with the project defaults:

        python scripts/tuned_level_1.py --overwrite

    The default input video is:

        dataset/raw/Vehicle_Tracking/VNTraffic/VNTraffic_Original-video.mp4

    The default YOLO model is:

        yolo11n.pt

    Because the default model is COCO-pretrained YOLO11n and the VNTraffic
    ground truth contains vehicles, the script keeps COCO vehicle classes by
    default: car, motorcycle, bus, and truck. For a custom one-class vehicle
    model, pass --classes 0. To disable class filtering, pass --all_classes.

    The default tuned tracker config is:

        configs/bytetrack_custom.yaml

    The default YOLO confidence threshold is 0.1. This is intentionally lower
    than the baseline 0.25 so ByteTrack can receive weak detections and use
    them for second-stage association instead of losing vehicles too early.

    The script creates three tuned outputs:

        outputs/tuned/vntraffic_tuned_yolo11n_bytetrack.txt
        outputs/tuned/tables/vntraffic_tuned_metrics.csv
        outputs/tuned/vntraffic_tuned_yolo11n_bytetrack.mp4

    Run with a custom model, video, and output paths:

        python scripts/tuned_level_1.py \
            --model runs/detect/train/weights/best.pt \
            --video dataset/raw/Vehicle_Tracking/VNTraffic/VNTraffic_Original-video.mp4 \
            --gt dataset/raw/Vehicle_Tracking/VNTraffic/VNTraffic_GroundTruth.txt \
            --tracker configs/bytetrack_custom.yaml \
            --classes 0 \
            --pred outputs/tuned/custom_prediction.txt \
            --metrics outputs/tuned/tables/custom_metrics.csv \
            --vis outputs/tuned/custom_visualization.mp4 \
            --overwrite

    Useful quick-tuning command without rendering video:

        python scripts/tuned_level_1.py --skip_render
"""

from pathlib import Path
import argparse

try:
    from .evaluate_mot import evaluate_mot
    from .export_baseline_result import export_baseline_mot
    from .render_tracking_video import render_tracking_video
except ImportError:
    from evaluate_mot import evaluate_mot
    from export_baseline_result import export_baseline_mot
    from render_tracking_video import render_tracking_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
DEFAULT_TRACKER = PROJECT_ROOT / "configs" / "bytetrack_custom.yaml"
DEFAULT_PRED = PROJECT_ROOT / "outputs" / "tuned" / "vntraffic_tuned_yolo11n_bytetrack.txt"
DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "tuned" / "tables" / "vntraffic_tuned_metrics.csv"
DEFAULT_VIDEO_OUT = PROJECT_ROOT / "outputs" / "tuned" / "vntraffic_tuned_yolo11n_bytetrack.mp4"
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


def resolve_classes(classes=None, all_classes=False):
    if all_classes:
        return None

    if classes is not None:
        return parse_class_ids(classes)

    return COCO_VEHICLE_CLASSES


def run_level_1(
    model=str(DEFAULT_MODEL),
    video=str(DEFAULT_VIDEO),
    gt=str(DEFAULT_GT),
    tracker=str(DEFAULT_TRACKER),
    pred=str(DEFAULT_PRED),
    metrics=str(DEFAULT_METRICS),
    vis=str(DEFAULT_VIDEO_OUT),
    conf=0.1,
    iou=0.5,
    name="tuned_level_1",
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
        print("===== Tuned Level 1: export MOT prediction =====")
        export_baseline_mot(
            model_path=model,
            video_path=video,
            tracker_cfg=tracker,
            output_txt=pred,
            conf=conf,
            iou=iou,
            classes=selected_classes,
        )
    else:
        print(f"===== Tuned Level 1: reuse MOT prediction at {pred} =====")

    if not skip_eval:
        print("\n===== Tuned Level 1: evaluate MOT metrics =====")
        evaluate_mot(
            gt_path=gt,
            pred_path=pred,
            output_csv=metrics,
            name=name,
        )
    else:
        print("\n===== Tuned Level 1: skip MOT evaluation =====")

    if not skip_render:
        print("\n===== Tuned Level 1: render visualization video =====")
        render_tracking_video(
            model_path=model,
            video_path=video,
            tracker_cfg=tracker,
            output_video=vis,
            save_mot=None,
            conf=conf,
            iou=iou,
            classes=selected_classes,
            show_class=show_class,
            show_conf=show_conf,
            max_frames=max_frames,
            overwrite=overwrite,
        )
    else:
        print("\n===== Tuned Level 1: skip visualization rendering =====")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tuned level 1 YOLO + ByteTrack export, evaluation, and visualization."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO model path")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Input video path")
    parser.add_argument("--gt", default=str(DEFAULT_GT), help="MOT ground-truth txt path")
    parser.add_argument("--tracker", default=str(DEFAULT_TRACKER), help="Tuned ByteTrack YAML path")
    parser.add_argument("--pred", default=str(DEFAULT_PRED), help="Output MOT prediction txt")
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS), help="Output metrics CSV")
    parser.add_argument("--vis", default=str(DEFAULT_VIDEO_OUT), help="Output visualization video")
    parser.add_argument("--conf", type=float, default=0.1, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU threshold")
    parser.add_argument("--name", default="tuned_level_1", help="Experiment name written to the metrics table")
    parser.add_argument(
        "--skip_export",
        action="store_true",
        help="Reuse an existing MOT prediction txt instead of exporting a new one",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip MOT metric evaluation",
    )
    parser.add_argument(
        "--skip_render",
        action="store_true",
        help="Skip visualization rendering for quick tuning runs",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class IDs, accepts space or comma separated values, e.g. 2 3 5 7",
    )
    parser.add_argument(
        "--vehicle_coco",
        action="store_true",
        help="Keep only COCO vehicle classes: car, motorcycle, bus, truck. This is the default preset.",
    )
    parser.add_argument(
        "--all_classes",
        action="store_true",
        help="Disable the default COCO vehicle class filter",
    )
    parser.add_argument("--show_class", action="store_true", help="Show class name in video labels")
    parser.add_argument("--show_conf", action="store_true", help="Show confidence score in video labels")
    parser.add_argument("--max_frames", type=int, default=None, help="Render only first N frames")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing visualization video",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_level_1(
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
