"""
Extract video frames for YOLO image-label dataset preparation.

Default behavior samples frames from the source video at a fixed interval and
writes them to the YOLO image directory:

    dataset/processed/images/train
    dataset/processed/images/val

This is different from MOT-optimized extraction. MOT extraction usually keeps
only frames listed in the ground-truth file. For YOLO training, sampled frames
are often more useful because they let you control data volume, reduce near-
duplicate consecutive frames, and create a clean train/val temporal split.

Examples:
    Extract every 5th frame to train:

        python setup-data/extract_frame.py --split train --frame_step 5

    Extract a validation segment from the end of the video:

        python setup-data/extract_frame.py --split val --start_frame 400 --frame_step 2

    Keep the old MOT behavior:

        python setup-data/extract_frame.py --mode gt --split train
"""

from pathlib import Path
import argparse

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "dataset" / "raw" / "Vehicle_Tracking" / "VNTraffic"
VIDEO_PATH = RAW_DIR / "VNTraffic_Original-video.mp4"
GT_PATH = RAW_DIR / "VNTraffic_GroundTruth.txt"

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "processed" / "images"
DATASET_PREFIX = "zenodo_vntraffic"


def load_frame_ids(gt_file):
    """Read unique frame IDs from a MOT ground-truth file."""
    frame_ids = set()

    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            cols = line.split(",")
            if len(cols) < 6:
                continue

            frame_ids.add(int(float(cols[0])))

    return sorted(frame_ids)


def build_sampled_frame_ids(total_frames, start_frame=0, end_frame=None, frame_step=5):
    if frame_step <= 0:
        raise ValueError("--frame_step must be greater than 0")

    start_frame = max(0, int(start_frame))
    end_frame = total_frames - 1 if end_frame is None else min(total_frames - 1, int(end_frame))

    if start_frame > end_frame:
        raise ValueError(
            f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}, total_frames={total_frames}"
        )

    return list(range(start_frame, end_frame + 1, frame_step))


def extract_frames(video_path, frame_ids, output_dir, prefix, image_ext=".jpg", overwrite=False, jpg_quality=95):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id_set = set(frame_ids)
    saved_count = 0
    skipped_existing = 0
    missing_ids = set(frame_ids)

    print(f"Video path: {video_path}")
    print(f"Total video frames reported by OpenCV: {total_video_frames}")
    print(f"Frames requested: {len(frame_ids)}")
    print(f"Output directory: {output_dir}")

    current_frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame_id in frame_id_set:
            image_name = f"{prefix}_{current_frame_id:06d}{image_ext}"
            image_path = output_dir / image_name

            if image_path.exists() and not overwrite:
                skipped_existing += 1
                missing_ids.discard(current_frame_id)
                current_frame_id += 1
                continue

            params = []
            if image_ext.lower() in {".jpg", ".jpeg"}:
                params = [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)]

            success = cv2.imwrite(str(image_path), frame, params)
            if success:
                saved_count += 1
                missing_ids.discard(current_frame_id)
            else:
                print(f"[WARNING] Failed to save: {image_path}")

        current_frame_id += 1

    cap.release()

    print("\nDone.")
    print(f"Saved frames: {saved_count}")
    print(f"Skipped existing frames: {skipped_existing}")

    if missing_ids:
        print(f"[WARNING] Missing {len(missing_ids)} requested frames. First missing IDs: {sorted(missing_ids)[:10]}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract video frames for YOLO dataset preparation."
    )
    parser.add_argument("--video", default=str(VIDEO_PATH), help="Input video path")
    parser.add_argument("--gt", default=str(GT_PATH), help="MOT ground-truth path, used only with --mode gt")
    parser.add_argument(
        "--mode",
        choices=["sampled", "gt"],
        default="sampled",
        help="sampled extracts by frame interval; gt extracts frames listed in MOT ground truth",
    )
    parser.add_argument("--split", choices=["train", "val"], default="train", help="YOLO image split")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT), help="Root output directory for images")
    parser.add_argument("--prefix", default=DATASET_PREFIX, help="Output image filename prefix")
    parser.add_argument("--start_frame", type=int, default=0, help="First frame to consider in sampled mode")
    parser.add_argument("--end_frame", type=int, default=None, help="Last frame to consider in sampled mode")
    parser.add_argument("--frame_step", type=int, default=5, help="Extract every Nth frame in sampled mode")
    parser.add_argument("--image_ext", default=".jpg", choices=[".jpg", ".jpeg", ".png"], help="Output image format")
    parser.add_argument("--jpg_quality", type=int, default=95, help="JPEG quality when writing jpg/jpeg")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted frames")
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output_root) / args.split

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if args.mode == "gt":
        frame_ids = load_frame_ids(Path(args.gt))
        frame_ids = [frame_id for frame_id in frame_ids if 0 <= frame_id < total_frames]
        print("Extraction mode: MOT ground-truth frame IDs")
    else:
        frame_ids = build_sampled_frame_ids(
            total_frames=total_frames,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            frame_step=args.frame_step,
        )
        print("Extraction mode: sampled frame interval")

    extract_frames(
        video_path=video_path,
        frame_ids=frame_ids,
        output_dir=output_dir,
        prefix=args.prefix,
        image_ext=args.image_ext,
        overwrite=args.overwrite,
        jpg_quality=args.jpg_quality,
    )


if __name__ == "__main__":
    main()
