"""
BrailleVision — Standalone Inference Script
============================================
Run Braille recognition on a single image or a folder of images from the command line.
No server required — runs the full pipeline locally.

Usage:
    python inference.py --source sample_inputs/test_braille.jpg
    python inference.py --source sample_inputs/
    python inference.py --source sample_inputs/test_braille.jpg --no-annotated
    python inference.py --source sample_inputs/test_braille.jpg --output-dir sample_outputs/

Example (matches hackathon judge command):
    python inference.py --source sample_inputs/test_braille.jpg --weights backend/models/ml_model.npz
"""

import argparse
import os
import sys
import time
import json

# ── Resolve backend path so imports work from project root ──────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(SCRIPT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)

import cv2
import numpy as np

from core.ml_model import initialize_ml_model, load_trained_model
from core.recognition import run_recognition


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(source: str) -> list[str]:
    """Return a list of image file paths from a file or directory."""
    if os.path.isfile(source):
        return [source]
    if os.path.isdir(source):
        paths = []
        for fname in sorted(os.listdir(source)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                paths.append(os.path.join(source, fname))
        return paths
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
    """
    Run Braille recognition on one image or a directory of images.

    Returns a list of result dicts, one per image.
    """
    # ── Load model ──────────────────────────────────────────────────────────
    if weights and os.path.exists(weights):
        loaded = load_trained_model(weights)
        if loaded:
            print(f"[MODEL] Loaded weights from: {weights}")
        else:
            print(f"[MODEL] Could not load {weights} — falling back to synthetic model.")
            initialize_ml_model()
    else:
        initialize_ml_model()

    # ── Collect images ───────────────────────────────────────────────────────
    image_paths = collect_images(source)
    if not image_paths:
        print(f"[ERROR] No supported images found in: {source}")
        sys.exit(1)

    print(f"\n[INFO] Running inference on {len(image_paths)} image(s)...\n")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[SKIP] Cannot read image: {img_path}")
            continue

        t0 = time.perf_counter()
        result = run_recognition(
            img,
            include_annotated=save_annotated,
            correct_perspective=correct_perspective,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        fname = os.path.basename(img_path)
        decoded_text = result.text if result.text.strip() else "[No Braille detected]"

        # ── Console output ───────────────────────────────────────────────────
        if verbose:
            print(f"{'─'*60}")
            print(f"  Image   : {fname}")
            print(f"  Text    : {decoded_text}")
            print(f"  Cells   : {result.cell_count}  |  Dots: {result.dot_count}")
            print(f"  Conf    : {int(result.confidence * 100)}%")
            print(f"  Time    : {elapsed_ms:.1f} ms")
            if result.cells:
                chips = "  ".join(
                    f"{c['char']}({c['confidence_pct']}%)"
                    for c in result.cells
                    if c["char"] != " "
                )
                print(f"  Cells   : {chips}")
            if result.error:
                print(f"  [WARN]  : {result.error}")

        # ── Save annotated image ─────────────────────────────────────────────
        if save_annotated and result.annotated_image_b64 and output_dir:
            import base64
            ann_bytes = base64.b64decode(result.annotated_image_b64)
            arr = np.frombuffer(ann_bytes, dtype=np.uint8)
            ann_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            stem = os.path.splitext(fname)[0]
            out_path = os.path.join(output_dir, f"{stem}_annotated.jpg")
            cv2.imwrite(out_path, ann_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if verbose:
                print(f"  Saved   : {out_path}")

        record = {
            "image": img_path,
            "text": decoded_text,
            "cell_count": result.cell_count,
            "dot_count": result.dot_count,
            "confidence": result.confidence,
            "processing_time_ms": round(elapsed_ms, 2),
            "lines": result.lines,
            "patterns_by_line": result.patterns_by_line,
        }
        results.append(record)

    # ── Summary ──────────────────────────────────────────────────────────────
    if verbose and len(results) > 1:
        print(f"\n{'═'*60}")
        print(f"  SUMMARY: {len(results)} images processed")
        avg_conf = sum(r["confidence"] for r in results) / len(results)
        avg_ms = sum(r["processing_time_ms"] for r in results) / len(results)
        print(f"  Avg confidence : {int(avg_conf * 100)}%")
        print(f"  Avg latency    : {avg_ms:.1f} ms")
        print(f"{'═'*60}\n")

    # ── Save JSON results ────────────────────────────────────────────────────
    if output_dir:
        json_path = os.path.join(output_dir, "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\n[INFO] Results saved to: {json_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BrailleVision — Offline Braille inference on images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --source sample_inputs/test_braille.jpg
  python inference.py --source sample_inputs/ --output-dir sample_outputs/
  python inference.py --source sample_inputs/test_braille.jpg --weights backend/models/ml_model.npz
        """,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to a single image file or a directory of images.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to a trained model .npz file (optional; defaults to backend/models/ml_model.npz).",
    )
    parser.add_argument(
        "--output-dir",
        default="sample_outputs",
        help="Directory to save annotated images and results.json (default: sample_outputs/).",
    )
    parser.add_argument(
        "--no-annotated",
        action="store_true",
        help="Skip saving annotated overlay images.",
    )
    parser.add_argument(
        "--no-perspective",
        action="store_true",
        help="Disable perspective/tilt correction.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-image console output.",
    )

    args = parser.parse_args()

    run_inference(
        source=args.source,
        weights=args.weights,
        output_dir=args.output_dir,
        save_annotated=not args.no_annotated,
        correct_perspective=not args.no_perspective,
        verbose=not args.quiet,
    )
