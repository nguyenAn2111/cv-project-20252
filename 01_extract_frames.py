import cv2
import os

ROOT_DIR = "/home/annq/HUST/2025.2/Computer-Vision/project/Vehicle_Tracking/VNTraffic"
VIDEO_PATH = os.path.join(ROOT_DIR, "VNTraffic_GroundTruth-video.mp4")
GT_PATH = os.path.join(ROOT_DIR, "VNTraffic_GroundTruth.txt")
OUTPUT_DIR = os.path.join(ROOT_DIR, "frames")


def load_frame_ids(gt_file):
    frame_ids = set()

    with open(gt_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            cols = line.split(",")

            frame_id = int(cols[0])
            frame_ids.add(frame_id)

    return sorted(frame_ids)


def extract_frames(video_path, frame_ids, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    current_frame = 0
    target_idx = 0

    total_needed = len(frame_ids)

    while cap.isOpened() and target_idx < total_needed:

        ret, frame = cap.read()

        if not ret:
            break

        target_frame = frame_ids[target_idx]

        if current_frame == target_frame:

            filename = os.path.join(
                output_dir,
                f"{current_frame:06d}.jpg"
            )

            cv2.imwrite(filename, frame)

            print(f"Saved: {filename}")

            target_idx += 1

        current_frame += 1

    cap.release()

    print(f"\nDone. Extracted {target_idx} frames.")


if __name__ == "__main__":

    frame_ids = load_frame_ids(GT_PATH)

    print(f"Found {len(frame_ids)} unique frame IDs")

    extract_frames(
        VIDEO_PATH,
        frame_ids,
        OUTPUT_DIR
    )