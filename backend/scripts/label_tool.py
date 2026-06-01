"""BrailleVision labeling helper.

This tool extracts detected Braille cells from real images so you can label them manually,
then builds a training dataset from those labels.

Usage:
  python scripts/label_tool.py extract --input-dir ../real_braille_photos --output-dir ../label_data
  python scripts/label_tool.py build --samples ../label_data/samples.csv --labels ../label_data/labels.csv --output ../label_data/real_dataset.npz
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.preprocessing import preprocess_frame
from core.dot_detector import detect_dots
from core.segmentation import segment_braille_dots
from core.ml_model import _CLASS_MAP

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_image_files(source_dir: Path) -> list[Path]:
    files = []
    for root, _, filenames in os.walk(source_dir):
        for filename in sorted(filenames):
            extension = Path(filename).suffix.lower()
            if extension in SUPPORTED_EXTENSIONS:
                files.append(Path(root) / filename)
    return files


def save_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def extract_samples(input_dir: Path, output_dir: Path, save_crops: bool, correct_perspective: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = output_dir / "cells"
    annotated_dir = output_dir / "annotated"
    cells_dir.mkdir(exist_ok=True)
    annotated_dir.mkdir(exist_ok=True)

    rows = []
    images = find_image_files(input_dir)
    if not images:
        raise FileNotFoundError(f"No supported images found in {input_dir}")

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        prep = preprocess_frame(image, correct_perspective=correct_perspective)
        dots = detect_dots(prep.cleaned)
        detection = segment_braille_dots(dots, prep.working)
        annotated_path = annotated_dir / f"{image_path.stem}_annotated{image_path.suffix}"
        cv2.imwrite(str(annotated_path), detection.annotated_image)

        for idx, cell in enumerate(detection.cells):
            cell_image_path = ""
            x, y, w, h = cell.bbox
            if save_crops and w > 0 and h > 0:
                crop = prep.working[y : y + h, x : x + w]
                cell_image_path = str(cells_dir / f"{image_path.stem}_cell{idx}.jpg")
                cv2.imwrite(cell_image_path, crop)

            rows.append(
                {
                    "source_image": str(image_path.name),
                    "cell_index": str(idx),
                    "bbox": json.dumps([int(x), int(y), int(w), int(h)]),
                    "pattern": cell.pattern,
                    "char": cell.char,
                    "confidence": f"{cell.confidence:.3f}",
                    "intensities": json.dumps([float(v) for v in cell.intensities]) if cell.intensities else "",
                    "crop_image": cell_image_path,
                }
            )

        print(f"Extracted {len(detection.cells)} cells from {image_path.name}")

    samples_csv = output_dir / "samples.csv"
    save_csv(samples_csv, rows, [
        "source_image",
        "cell_index",
        "bbox",
        "pattern",
        "char",
        "confidence",
        "intensities",
        "crop_image",
    ])
    print(f"Saved extracted sample metadata to {samples_csv}")


def build_dataset(samples_path: Path, labels_path: Path, output_path: Path) -> None:
    samples = load_csv(samples_path)
    labels = load_csv(labels_path)

    label_map = {}
    for row in labels:
        key = (row["source_image"], row["cell_index"])
        label_map[key] = row["label_pattern"].strip()

    features = []
    targets = []
    missing = 0
    for row in samples:
        key = (row["source_image"], row["cell_index"])
        if key not in label_map:
            missing += 1
            continue

        label_pattern = label_map[key]
        if label_pattern not in _CLASS_MAP:
            raise ValueError(f"Unknown label pattern: {label_pattern} for {key}")

        intensities = []
        if row["intensities"]:
            try:
                intensities = json.loads(row["intensities"])
            except json.JSONDecodeError:
                intensities = []

        if len(intensities) == 6:
            feature = np.array(intensities, dtype=np.float32)
        else:
            feature = np.array([float(bit) for bit in row["pattern"]], dtype=np.float32)

        features.append(feature)
        targets.append(int(_CLASS_MAP[label_pattern]))

    if missing:
        print(f"Warning: skipped {missing} sample rows without labels.")

    if not features:
        raise ValueError("No matching labeled samples found.")

    x = np.stack(features, axis=0)
    y = np.array(targets, dtype=np.int32)
    np.savez_compressed(str(output_path), x=x, y=y)
    print(f"Saved dataset with {len(x)} samples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Labeling helper for BrailleVision")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract detected Braille cells from images")
    extract_parser.add_argument("--input-dir", required=True, help="Folder containing real Braille images")
    extract_parser.add_argument("--output-dir", required=True, help="Folder to save extracted cells and metadata")
    extract_parser.add_argument("--no-crops", action="store_true", help="Do not save cropped cell preview images")
    extract_parser.add_argument("--correct-perspective", action="store_true", help="Enable perspective correction during preprocessing")

    build_parser = subparsers.add_parser("build", help="Build a trainable .npz dataset from extracted samples and labels")
    build_parser.add_argument("--samples", required=True, help="Path to samples.csv produced by extract")
    build_parser.add_argument("--labels", required=True, help="Path to labels.csv with manual labels")
    build_parser.add_argument("--output", required=True, help="Path to save the dataset .npz")

    args = parser.parse_args()

    if args.command == "extract":
        extract_samples(
            Path(args.input_dir),
            Path(args.output_dir),
            save_crops=not args.no_crops,
            correct_perspective=args.correct_perspective,
        )
    elif args.command == "build":
        build_dataset(Path(args.samples), Path(args.labels), Path(args.output))


if __name__ == "__main__":
    main()
