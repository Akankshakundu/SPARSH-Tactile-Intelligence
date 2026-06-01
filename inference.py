"""
Standalone Braille OCR inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(SCRIPT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)

from core.recognition import run_recognition


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(source: str) -> list[str]:
    if os.path.isfile(source):
        return [source]
    if os.path.isdir(source):
        return [
            os.path.join(source, name)
            for name in sorted(os.listdir(source))
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXTS
        ]
    print(f"[ERROR] Source not found: {source}")
    sys.exit(1)


def run_inference(
    source: str,
    weights: str | None = None,
    output_dir: str | None = None,
    save_annotated: bool = True,
    correct_perspective: bool = True,
    verbose: bool = True,
) -> list[dict]:
    if weights and verbose:
        print(f"[INFO] Ignoring weights at {weights}; the active OCR path is deterministic and pattern-based.")

    image_paths = collect_images(source)
    if not image_paths:
        print(f"[ERROR] No supported images found in: {source}")
        sys.exit(1)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results: list[dict] = []
    if verbose:
        print(f"\n[INFO] Running inference on {len(image_paths)} image(s)...\n")

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"[SKIP] Cannot read image: {image_path}")
            continue

        start = time.perf_counter()
        result = run_recognition(
            image,
            include_annotated=save_annotated,
            correct_perspective=correct_perspective,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        text = result.text if result.text.strip() else "[No Braille detected]"

        if verbose:
            print("=" * 60)
            print(f"  Image   : {os.path.basename(image_path)}")
            print(f"  Text    : {text}")
            print(f"  Cells   : {result.cell_count}  |  Dots: {result.dot_count}")
            print(f"  Conf    : {int(result.confidence * 100)}%")
            print(f"  Time    : {elapsed_ms:.1f} ms")
            if result.error:
                print(f"  Error   : {result.error}")

        if save_annotated and result.annotated_image_b64 and output_dir:
            import base64

            annotated = base64.b64decode(result.annotated_image_b64)
            ann_img = cv2.imdecode(np.frombuffer(annotated, dtype=np.uint8), cv2.IMREAD_COLOR)
            stem = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(os.path.join(output_dir, f"{stem}_annotated.jpg"), ann_img)

        results.append(
            {
                "image": image_path,
                "text": text,
                "cell_count": result.cell_count,
                "dot_count": result.dot_count,
                "confidence": result.confidence,
                "processing_time_ms": round(elapsed_ms, 2),
                "lines": result.lines,
                "patterns_by_line": result.patterns_by_line,
                "debug": result.debug,
            }
        )

    if output_dir:
        json_path = os.path.join(output_dir, "results.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\n[INFO] Results saved to: {json_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Braille OCR inference on images.")
    parser.add_argument("--source", required=True, help="Image file or directory.")
    parser.add_argument("--weights", default=None, help="Deprecated compatibility flag; ignored.")
    parser.add_argument("--output-dir", default="sample_outputs", help="Directory for annotated images and results.json.")
    parser.add_argument("--no-annotated", action="store_true", help="Skip saving annotated images.")
    parser.add_argument("--no-perspective", action="store_true", help="Disable perspective correction.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-image console output.")

    args = parser.parse_args()
    run_inference(
        source=args.source,
        weights=args.weights,
        output_dir=args.output_dir,
        save_annotated=not args.no_annotated,
        correct_perspective=not args.no_perspective,
        verbose=not args.quiet,
    )
