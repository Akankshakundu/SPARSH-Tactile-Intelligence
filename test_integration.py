#!/usr/bin/env python3
"""
Phase 2 Integration Tests for Braille OCR Pipeline.
Tests core components: preprocessing, dot detection, segmentation, recognition.
"""

import sys
import os
import logging
from pathlib import Path

# Add backend to path
SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import cv2
import numpy as np
from core.recognition import run_recognition
from core.preprocessing import preprocess_frame
from core.dot_detector import detect_dots
from core.segmentation import segment_braille_dots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)


def test_image_file(image_path: str) -> dict:
    """Test full pipeline on a single image."""
    LOGGER.info(f"\n{'='*70}")
    LOGGER.info(f"Testing: {image_path}")
    LOGGER.info(f"{'='*70}")
    
    if not os.path.exists(image_path):
        LOGGER.error(f"Image not found: {image_path}")
        return {"status": "skipped", "reason": "not_found"}
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            LOGGER.error(f"Cannot read image: {image_path}")
            return {"status": "error", "reason": "cannot_read"}
        
        LOGGER.info(f"Image size: {img.shape}")
        
        # Run full pipeline
        result = run_recognition(img, include_annotated=True)
        
        LOGGER.info(f"Result:")
        LOGGER.info(f"  Success: {result.success}")
        LOGGER.info(f"  Text: {result.text!r}")
        LOGGER.info(f"  Cells: {result.cell_count}")
        LOGGER.info(f"  Dots: {result.dot_count}")
        LOGGER.info(f"  Confidence: {result.confidence * 100:.1f}%")
        LOGGER.info(f"  Time: {result.processing_time_ms:.1f}ms")
        
        if result.error:
            LOGGER.error(f"  Error: {result.error}")
            return {
                "status": "error",
                "text": result.text,
                "cells": result.cell_count,
                "error": result.error,
            }
        
        if result.debug:
            if "segmentation" in result.debug:
                seg = result.debug["segmentation"]
                LOGGER.info(f"  Debug:")
                LOGGER.info(f"    Lines: {seg.get('line_count', 'N/A')}")
                LOGGER.info(f"    Dots detected: {seg.get('dot_count', 'N/A')}")
        
        return {
            "status": "ok",
            "text": result.text,
            "cells": result.cell_count,
            "dots": result.dot_count,
            "confidence": result.confidence,
            "processing_time_ms": result.processing_time_ms,
        }
        
    except Exception as e:
        LOGGER.exception(f"Unexpected error:")
        return {
            "status": "exception",
            "error": str(e),
        }


def find_sample_images(directory: str) -> list[str]:
    """Find all image files in directory."""
    sample_dir = Path(directory)
    if not sample_dir.exists():
        LOGGER.warning(f"Sample directory not found: {directory}")
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    images = []
    for ext in image_extensions:
        images.extend(sorted(sample_dir.glob(f"*{ext}")))
        images.extend(sorted(sample_dir.glob(f"*{ext.upper()}")))
    
    return [str(img) for img in images]


def main():
    """Run integration tests."""
    LOGGER.info("Braille OCR Pipeline - Phase 2 Integration Tests")
    LOGGER.info("=" * 70)
    
    # Determine sample directory
    base_dir = Path(__file__).parent
    sample_dir = base_dir / "sample_inputs"
    
    LOGGER.info(f"Looking for samples in: {sample_dir}")
    
    images = find_sample_images(str(sample_dir))
    if not images:
        LOGGER.error("No sample images found!")
        return 1
    
    LOGGER.info(f"Found {len(images)} sample image(s)")
    
    # Test each image
    results = {}
    for image_path in images:
        filename = Path(image_path).name
        results[filename] = test_image_file(image_path)
    
    # Summary
    LOGGER.info(f"\n{'='*70}")
    LOGGER.info("TEST SUMMARY")
    LOGGER.info(f"{'='*70}")
    
    ok_count = sum(1 for r in results.values() if r.get("status") == "ok")
    error_count = sum(1 for r in results.values() if r.get("status") in ("error", "exception"))
    skipped_count = sum(1 for r in results.values() if r.get("status") == "skipped")
    
    LOGGER.info(f"Passed:  {ok_count}/{len(results)}")
    LOGGER.info(f"Failed:  {error_count}/{len(results)}")
    LOGGER.info(f"Skipped: {skipped_count}/{len(results)}")
    
    for filename, result in results.items():
        status = result.get("status", "unknown").upper()
        if status == "OK":
            text_preview = result.get("text", "")[:40]
            LOGGER.info(f"  ✓ {filename}: {result.get('cells', 0)} cells, '{text_preview}'")
        elif status == "SKIPPED":
            LOGGER.info(f"  ⊘ {filename}: {result.get('reason', 'unknown')}")
        else:
            LOGGER.error(f"  ✗ {filename}: {result.get('error', 'unknown error')}")
    
    # Return exit code
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
