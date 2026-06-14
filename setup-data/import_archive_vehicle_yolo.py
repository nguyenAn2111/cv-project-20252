"""
Import archive/daytime + archive/nighttime YOLO labels into one-class vehicle data.

The downloaded dataset already uses YOLO txt labels, but it contains multiple
vehicle-related class IDs. This importer remaps every object to class 0
(`vehicle`) and creates a balanced train/valid split.

Output layout:
    dataset/processed/images/train
    dataset/processed/images/valid
    dataset/processed/labels/train
    dataset/processed/labels/valid
"""

from pathlib import Path
import argparse
import random
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = PROJECT_ROOT / "archive"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "processed"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def collect_pairs(archive_root):
    domains = {
        "daytime": archive_root / "daytime-dataset" / "daytime",
        "nighttime": archive_root / "nighttime-dataset" / "nighttime",
    }
    pairs = []
    missing_labels = []

    for domain, folder in domains.items():
        if not folder.exists():
            raise FileNotFoundError(f"Missing source folder: {folder}")

        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            label_path = image_path.with_suffix(".txt")
            if not label_path.exists():
                missing_labels.append(image_path)
                continue

            pairs.append((domain, image_path, label_path))

    if missing_labels:
        print(f"[WARNING] Skipped {len(missing_labels)} images without labels.")
        for path in missing_labels[:10]:
            print(f"  missing label for: {path}")

    return pairs


def split_pairs_by_domain(pairs, valid_ratio, seed):
    rng = random.Random(seed)
    by_domain = {}
    for domain, image_path, label_path in pairs:
        by_domain.setdefault(domain, []).append((domain, image_path, label_path))

    train = []
    valid = []
    for domain, domain_pairs in by_domain.items():
        rng.shuffle(domain_pairs)
        valid_count = max(1, round(len(domain_pairs) * valid_ratio))
        valid.extend(domain_pairs[:valid_count])
        train.extend(domain_pairs[valid_count:])
        print(
            f"{domain}: total={len(domain_pairs)}, "
            f"train={len(domain_pairs) - valid_count}, valid={valid_count}"
        )

    rng.shuffle(train)
    rng.shuffle(valid)
    return {"train": train, "valid": valid}


def remap_label_to_vehicle(src_label, dst_label):
    lines_out = []
    invalid = 0

    with open(src_label, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if len(parts) != 5:
                invalid += 1
                continue

            try:
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                invalid += 1
                continue

            if any(value < 0.0 or value > 1.0 for value in coords):
                invalid += 1
                continue

            lines_out.append("0 " + " ".join(f"{value:.8f}" for value in coords))

    dst_label.write_text("\n".join(lines_out), encoding="utf-8")
    return len(lines_out), invalid


def link_or_copy(src, dst, mode):
    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return

    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def prepare_output_dirs(output_root):
    dirs = {}
    for split in ("train", "valid"):
        dirs[(split, "images")] = output_root / "images" / split
        dirs[(split, "labels")] = output_root / "labels" / split
        dirs[(split, "images")].mkdir(parents=True, exist_ok=True)
        dirs[(split, "labels")].mkdir(parents=True, exist_ok=True)
    return dirs


def import_dataset(archive_root, output_root, valid_ratio, seed, mode, dry_run=False):
    pairs = collect_pairs(archive_root)
    splits = split_pairs_by_domain(pairs, valid_ratio=valid_ratio, seed=seed)

    print(f"\nTotal image-label pairs: {len(pairs)}")
    print(f"Train pairs: {len(splits['train'])}")
    print(f"Valid pairs: {len(splits['valid'])}")

    if dry_run:
        print("\nDry run only. No files were written.")
        return

    dirs = prepare_output_dirs(output_root)
    total_boxes = 0
    total_invalid = 0

    for split, split_pairs in splits.items():
        for domain, image_path, label_path in split_pairs:
            stem = f"kaggle_{domain}_{image_path.stem}"
            dst_image = dirs[(split, "images")] / f"{stem}{image_path.suffix.lower()}"
            dst_label = dirs[(split, "labels")] / f"{stem}.txt"

            if dst_image.exists():
                dst_image.unlink()
            if dst_label.exists():
                dst_label.unlink()

            link_or_copy(image_path, dst_image, mode=mode)
            boxes, invalid = remap_label_to_vehicle(label_path, dst_label)
            total_boxes += boxes
            total_invalid += invalid

    print("\nDone.")
    print(f"Output root: {output_root}")
    print(f"Imported boxes remapped to class 0: {total_boxes}")
    print(f"Skipped invalid label lines: {total_invalid}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import archive vehicle data into one-class YOLO train/valid folders."
    )
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE), help="Archive root directory")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="YOLO dataset output root")
    parser.add_argument("--valid_ratio", type=float, default=0.2, help="Validation ratio per domain")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    parser.add_argument(
        "--mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="How to place image files in dataset/processed",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only print split statistics")
    return parser.parse_args()


def main():
    args = parse_args()

    if not 0.0 < args.valid_ratio < 1.0:
        raise ValueError("--valid_ratio must be between 0 and 1")

    import_dataset(
        archive_root=Path(args.archive),
        output_root=Path(args.output),
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        mode=args.mode,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
