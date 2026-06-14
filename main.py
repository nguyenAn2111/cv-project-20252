from pathlib import Path
import argparse
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-cache"),
)

from scripts.evaluate_mot import evaluate_mot
from scripts.export_baseline_result import export_baseline_mot
from scripts.render_tracking_video import render_tracking_video
from scripts.tuned_level_1 import COCO_VEHICLE_CLASSES, parse_class_ids, run_level_1
from scripts.tuned_level_2 import export_level_2_mot, run_level_2


PROJECT_ROOT = Path(__file__).resolve().parent

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
BASELINE_TRACKER = PROJECT_ROOT / "configs" / "bytetrack_baseline.yaml"
LEVEL_1_TRACKER = PROJECT_ROOT / "configs" / "bytetrack_custom.yaml"
LEVEL_2_TRACKER = PROJECT_ROOT / "configs" / "custom_tracker.yaml"


def resolve_classes(args):
    if args.all_classes:
        return None
    if args.classes is not None:
        return parse_class_ids(args.classes)
    return COCO_VEHICLE_CLASSES


def level_output_paths(level):
    output_dir = PROJECT_ROOT / "outputs" / level
    tracker_name = "custom_bytetrack" if level == "level2" else "bytetrack"
    return {
        "pred": output_dir / f"vntraffic_{level}_yolo11n_{tracker_name}.txt",
        "metrics": output_dir / "tables" / f"vntraffic_{level}_metrics.csv",
        "video": output_dir / f"vntraffic_{level}_yolo11n_{tracker_name}.mp4",
    }


def get_outputs(args):
    outputs = level_output_paths(args.level)
    if args.pred is not None:
        outputs["pred"] = Path(args.pred)
    if args.metrics is not None:
        outputs["metrics"] = Path(args.metrics)
    if args.output_video is not None:
        outputs["video"] = Path(args.output_video)
    return outputs


def tracker_for_level(level):
    if level == "baseline":
        return BASELINE_TRACKER
    if level == "level1":
        return LEVEL_1_TRACKER
    if level == "level2":
        return LEVEL_2_TRACKER
    raise ValueError(f"Unsupported level: {level}")


def ensure_ground_truth(args):
    gt_path = Path(args.gt)
    if not gt_path.is_file():
        raise FileNotFoundError(
            f"Ground truth file not found: {gt_path}. "
            "Use --no_gt for videos without ground truth."
        )


def run_baseline_with_gt(args, outputs):
    classes = resolve_classes(args)

    export_baseline_mot(
        model_path=args.model,
        video_path=args.video,
        tracker_cfg=str(BASELINE_TRACKER),
        output_txt=str(outputs["pred"]),
        conf=args.conf,
        iou=args.iou,
        classes=classes,
    )

    evaluate_mot(
        gt_path=args.gt,
        pred_path=str(outputs["pred"]),
        output_csv=str(outputs["metrics"]),
        name="baseline",
    )

    render_tracking_video(
        model_path=args.model,
        video_path=args.video,
        tracker_cfg=str(BASELINE_TRACKER),
        output_video=str(outputs["video"]),
        save_mot=None,
        conf=args.conf,
        iou=args.iou,
        classes=classes,
        show_class=args.show_class,
        show_conf=args.show_conf,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
    )


def run_tracking_only(args, outputs):
    classes = resolve_classes(args)
    tracker = tracker_for_level(args.level)

    if args.level in {"baseline", "level1"}:
        render_tracking_video(
            model_path=args.model,
            video_path=args.video,
            tracker_cfg=str(tracker),
            output_video=str(outputs["video"]),
            save_mot=None,
            conf=args.conf,
            iou=args.iou,
            classes=classes,
            show_class=args.show_class,
            show_conf=args.show_conf,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        )
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pred = Path(tmp_dir) / "level2_prediction.txt"
        export_level_2_mot(
            model_path=args.model,
            video_path=args.video,
            tracker_cfg=str(tracker),
            output_txt=str(tmp_pred),
            output_video=str(outputs["video"]),
            conf=args.conf,
            iou=args.iou,
            classes=classes,
            show_class=args.show_class,
            show_conf=args.show_conf,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run video tracking with baseline, level1, or level2. "
            "Use --with_gt for metrics, or --no_gt for tracking video only."
        )
    )
    parser.add_argument("--level", choices=["baseline", "level1", "level2"], default="level1")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--gt", default=str(DEFAULT_GT))
    gt_group = parser.add_mutually_exclusive_group()
    gt_group.add_argument(
        "--with_gt",
        dest="has_gt",
        action="store_true",
        default=True,
        help="Evaluate against ground truth and write metrics. This is the default.",
    )
    gt_group.add_argument(
        "--no_gt",
        dest="has_gt",
        action="store_false",
        help="Run tracking only and write only the visualization video.",
    )
    parser.add_argument("--output_video", default=None, help="Custom output tracking video path")
    parser.add_argument("--pred", default=None, help="Custom MOT prediction txt path for --with_gt runs")
    parser.add_argument("--metrics", default=None, help="Custom metrics CSV path for --with_gt runs")
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="YOLO confidence threshold. Defaults to 0.25 for baseline, 0.1 for level1, and 0.08 for level2.",
    )
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Class IDs to keep, e.g. --classes 0 or --classes 2,3,5,7",
    )
    parser.add_argument("--all_classes", action="store_true")
    parser.add_argument("--show_class", action="store_true")
    parser.add_argument("--show_conf", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.conf is None:
        if args.level == "baseline":
            args.conf = 0.25
        elif args.level == "level2":
            args.conf = 0.08
        else:
            args.conf = 0.1

    outputs = get_outputs(args)

    if not args.has_gt:
        run_tracking_only(args, outputs)
        return

    ensure_ground_truth(args)

    if args.level == "baseline":
        run_baseline_with_gt(args, outputs)
        return

    if args.level == "level1":
        run_level_1(
            model=args.model,
            video=args.video,
            gt=args.gt,
            tracker=str(LEVEL_1_TRACKER),
            pred=str(outputs["pred"]),
            metrics=str(outputs["metrics"]),
            vis=str(outputs["video"]),
            conf=args.conf,
            iou=args.iou,
            name="level1",
            classes=args.classes,
            all_classes=args.all_classes,
            show_class=args.show_class,
            show_conf=args.show_conf,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        )
        return

    if args.level == "level2":
        run_level_2(
            model=args.model,
            video=args.video,
            gt=args.gt,
            tracker=str(LEVEL_2_TRACKER),
            pred=str(outputs["pred"]),
            metrics=str(outputs["metrics"]),
            vis=str(outputs["video"]),
            conf=args.conf,
            iou=args.iou,
            name="level2",
            classes=args.classes,
            all_classes=args.all_classes,
            show_class=args.show_class,
            show_conf=args.show_conf,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        )
        return

    raise ValueError(f"Unsupported level: {args.level}")


if __name__ == "__main__":
    main()
