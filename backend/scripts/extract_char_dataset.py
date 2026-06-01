"""Extract 6-intensity features from 28x28 Braille character images.

This script reads a folder of 28x28 Braille character images (like the angelina/character dataset)
and converts each to a 6-element intensity vector matching the real pipeline format.

Usage:
    python scripts/extract_char_dataset.py --input-dir ../braille_char_dataset --output ../label_data/char_dataset.npz
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.ml_model import _CLASS_MAP

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# Load Braille character to pattern mapping
_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "braille_cells.json")
with open(_DATA_PATH, "r", encoding="utf-8") as f:
    _BRAILLE_DATA = json.load(f)

# Create reverse mapping: character → pattern
_CHAR_TO_PATTERN = {}
for pattern, char in _BRAILLE_DATA["letters"].items():
    _CHAR_TO_PATTERN[char] = pattern


def extract_braille_char_from_filename(filename: str) -> str:
    """Extract character from filename.
    Handles formats like: 'a1.JPG0dim.jpg' -> 'a', 'a_0_original.png' -> 'a'
    """
    # Remove extension and get stem
    stem = filename.split(".")[0]
    
    # Try to get first alphabetic character
    for char in stem:
        if char.isalpha():
            return char.lower()
    
    return None


def detect_dots_in_28x28(image: np.ndarray) -> list[tuple[int, int, float]]:
    """Simple blob detection in a 28x28 Braille character image.
    Returns list of (x, y, intensity).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Invert if needed (assume black dots on white background)
    if np.mean(gray) > 128:
        gray = 255 - gray

    # Normalize intensity
    gray = gray.astype(np.float32) / 255.0

    # Simple blob detection: find local peaks
    blurred = cv2.GaussianBlur((gray * 255).astype(np.uint8), (3, 3), 0)
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 2 or area > 100:  # reasonable blob size in 28x28
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        intensity = float(np.mean(gray[max(0, cy - 2) : min(28, cy + 3), max(0, cx - 2) : min(28, cx + 3)]))
        dots.append((cx, cy, intensity))

    return dots


def fit_dots_to_6_slots(dots: list[tuple[int, int, float]]) -> np.ndarray:
    """Map detected dots to the 6 Braille cell positions.
    28x28 image space: assume dots are in a rough 2x3 grid.
    Slots: left column (0,1,2) at x ~8, right column (3,4,5) at x ~18
           top row at y ~6, middle at y ~14, bottom at y ~22 (approx).
    """
    intensities = np.zeros(6, dtype=np.float32)

    if not dots:
        return intensities

    # Define slot centers in 28x28 space
    slot_positions = [
        (8, 6),    # slot 0: top-left
        (8, 14),   # slot 1: mid-left
        (8, 22),   # slot 2: bot-left
        (18, 6),   # slot 3: top-right
        (18, 14),  # slot 4: mid-right
        (18, 22),  # slot 5: bot-right
    ]

    # Assign each detected dot to the nearest slot
    match_radius = 6
    used_slots = set()

    for dx, dy, intensity in sorted(dots, key=lambda d: -d[2]):  # process brightest first
        best_slot = -1
        best_dist = match_radius + 1

        for slot_idx, (sx, sy) in enumerate(slot_positions):
            if slot_idx in used_slots:
                continue
            dist = np.sqrt((dx - sx) ** 2 + (dy - sy) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_slot = slot_idx

        if best_slot >= 0:
            intensities[best_slot] = max(intensities[best_slot], intensity)
            used_slots.add(best_slot)

    return intensities


def process_image_file(image_path: Path) -> tuple[np.ndarray, str, bool]:
    """Read image, extract features, return (6-intensity vector, character label, success).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None, False

    character = extract_braille_char_from_filename(image_path.stem)
    if not character:
        return None, None, False

    # Resize to 28x28 if needed
    if image.shape[:2] != (28, 28):
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    dots = detect_dots_in_28x28(image)
    intensities = fit_dots_to_6_slots(dots)

    return intensities, character, True


def extract_char_dataset(input_dir: Path, output_path: Path) -> None:
    """Extract all 28x28 character images and build a .npz dataset."""
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_data = []
    y_data = []
    char_counts = {}

    image_files = sorted([f for f in input_dir.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS])

    if not image_files:
        raise FileNotFoundError(f"No image files found in {input_dir}")

    print(f"Found {len(image_files)} image files. Processing...")
    print(f"First few files: {[f.name for f in image_files[:5]]}")

    skipped_count = 0
    for idx, image_path in enumerate(image_files):
        intensities, character, success = process_image_file(image_path)

        if not success:
            skipped_count += 1
            if skipped_count <= 3:
                print(f"  [DEBUG] Skipped {image_path.name}: couldn't read or extract character")
            continue

        # Convert character to Braille pattern
        if character not in _CHAR_TO_PATTERN:
            skipped_count += 1
            if skipped_count <= 3:
                print(f"  [DEBUG] Skipped {image_path.name}: character '{character}' not in Braille mapping")
            continue

        pattern = _CHAR_TO_PATTERN[character]
        
        # Convert pattern to class ID
        if pattern not in _CLASS_MAP:
            skipped_count += 1
            if skipped_count <= 3:
                print(f"  [DEBUG] Skipped {image_path.name}: pattern '{pattern}' not in class map")
            continue

        x_data.append(intensities)
        class_id = _CLASS_MAP[pattern]
        y_data.append(class_id)
        char_counts[character] = char_counts.get(character, 0) + 1

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(image_files)} (extracted {len(x_data)}, skipped {skipped_count})")

    if not x_data:
        print(f"\n[ERROR] No valid labeled images extracted.")
        print(f"Total files found: {len(image_files)}")
        print(f"Total skipped: {skipped_count}")
        print(f"\nAvailable characters in Braille mapping: {sorted(_CHAR_TO_PATTERN.keys())}")
        raise ValueError("No valid labeled images extracted from dataset.")

    x = np.stack(x_data, axis=0).astype(np.float32)
    y = np.array(y_data, dtype=np.int32)

    np.savez_compressed(str(output_path), x=x, y=y)

    print(f"\n✅ Extracted {len(x)} samples.")
    print("Character distribution:")
    for char in sorted(char_counts.keys()):
        print(f"  {char}: {char_counts[char]}")
    print(f"\nSaved dataset to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract features from 28x28 Braille character images.")
    parser.add_argument("--input-dir", required=True, help="Folder containing 28x28 Braille character images.")
    parser.add_argument("--output", required=True, help="Path to save the output .npz dataset.")
    args = parser.parse_args()

    extract_char_dataset(Path(args.input_dir), Path(args.output))


if __name__ == "__main__":
    main()
