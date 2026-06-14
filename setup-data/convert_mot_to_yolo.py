"""
MOT-to-YOLO Annotation Conversion Utility.

This script converts VNTraffic ground truth annotations from MOT format to YOLO
object detection format. The original MOT annotation file contains tracking
information in the following structure:

    frame_id, object_id, x, y, width, height, confidence, -1, -1, -1

In this format, `x` and `y` represent the top-left corner of the bounding box,
while YOLO requires normalized bounding box coordinates in the form:

    class_id x_center y_center width height

During conversion, the script groups all objects that belong to the same frame
and writes them into a single YOLO label file. For example, all annotations with
frame_id = 0 are written to:

    zenodo_vntraffic_000000.txt

which corresponds to the image:

    zenodo_vntraffic_000000.jpg

Since YOLO detection training does not use object IDs, the `object_id` field is
discarded. For a single-class vehicle detection task, all objects are assigned
class_id = 0. The original object IDs should only be used later for MOT tracking
evaluation, such as MOTA, IDF1, FP, FN, and ID switches.

Input:
    - MOT ground truth file: VNTraffic_GroundTruth.txt
    - Extracted image frames, e.g. zenodo_vntraffic_000000.jpg

Output:
    - YOLO label files in TXT format, stored in the YOLO dataset label directory,
      e.g. dataset/processed/train/labels/

Note:
    The image filename prefix and label filename prefix must be identical so that
    YOLO can correctly match each image with its corresponding annotation file.
DangTran
"""

from pathlib import Path
import cv2


# =========================
# Project paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "dataset" / "raw" / "Vehicle_Tracking" / "VNTraffic"

GT_PATH = RAW_DIR / "VNTraffic_GroundTruth.txt"

IMAGES_DIR = PROJECT_ROOT / "dataset" / "processed" / "images" / "train"
LABELS_DIR = PROJECT_ROOT / "dataset" / "processed" / "labels" / "train"

DATASET_PREFIX = "zenodo_vntraffic"

# If your dataset only contains vehicles/cars, use one class:
# 0 = vehicle
CLASS_ID = 0


def convert_mot_to_yolo(
    gt_path: Path,
    images_dir: Path,
    labels_dir: Path,
    prefix: str,
    class_id: int = 0,
):
    """
    Convert MOT ground truth format to YOLO detection format.

    MOT:
    frame_id, object_id, x, y, w, h, confidence, -1, -1, -1

    YOLO:
    class_id x_center_norm y_center_norm width_norm height_norm
    """

    labels_dir.mkdir(parents=True, exist_ok=True)

    frame_to_labels = {}

    with open(gt_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            cols = line.split(",")

            if len(cols) < 6:
                print(f"[WARNING] Skip invalid line {line_idx}: {line}")
                continue

            frame_id = int(float(cols[0]))
            object_id = int(float(cols[1]))

            x = float(cols[2])
            y = float(cols[3])
            w = float(cols[4])
            h = float(cols[5])

            image_name = f"{prefix}_{frame_id:06d}.jpg"
            image_path = images_dir / image_name

            if not image_path.exists():
                print(f"[WARNING] Missing image for frame {frame_id}: {image_path}")
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"[WARNING] Cannot read image: {image_path}")
                continue

            img_h, img_w = image.shape[:2]

            # MOT bbox uses top-left x, top-left y, width, height
            # YOLO needs normalized center x, center y, width, height
            x_center = (x + w / 2.0) / img_w
            y_center = (y + h / 2.0) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Clip values to [0, 1] to avoid small annotation overflow
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            yolo_line = (
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{w_norm:.6f} "
                f"{h_norm:.6f}"
            )

            if frame_id not in frame_to_labels:
                frame_to_labels[frame_id] = []

            frame_to_labels[frame_id].append(yolo_line)

    for frame_id, labels in frame_to_labels.items():
        label_name = f"{prefix}_{frame_id:06d}.txt"
        label_path = labels_dir / label_name

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(labels))

    print("\nDone.")
    print(f"Created {len(frame_to_labels)} YOLO label files.")
    print(f"Labels saved to: {labels_dir}")


if __name__ == "__main__":
    convert_mot_to_yolo(
        gt_path=GT_PATH,
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        prefix=DATASET_PREFIX,
        class_id=CLASS_ID,
    )