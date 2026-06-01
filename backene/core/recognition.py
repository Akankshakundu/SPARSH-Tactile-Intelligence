"""
Pipeline orchestration for Braille OCR.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .braille_mapper import decode_lines_with_metadata
from .dot_detector import detect_dots
from .pipeline_types import DetectionResult
from .preprocessing import encode_image_to_base64, preprocess_frame
from .segmentation import segment_braille_dots


LOGGER = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    success: bool
    text: str
    lines: list[str]
    patterns_by_line: list[list[str]]
    cells: list[dict[str, Any]] = field(default_factory=list)
    dot_count: int = 0
    cell_count: int = 0
    processing_time_ms: float = 0.0
    annotated_image_b64: Optional[str] = None
    error: Optional[str] = None
    confidence: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)


def _cells_to_payload(cells, line_cell_chars: list[list[str]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    line_indexes = {line_idx: 0 for line_idx in range(len(line_cell_chars))}
    for index, cell in enumerate(cells):
        char = cell.char
        if cell.row < len(line_cell_chars):
            offset = line_indexes[cell.row]
            if offset < len(line_cell_chars[cell.row]):
                char = line_cell_chars[cell.row][offset]
            line_indexes[cell.row] += 1
        payload.append(
            {
                "index": index,
                "pattern": cell.pattern,
                "char": char,
                "confidence": round(float(cell.confidence), 3),
                "confidence_pct": int(round(float(cell.confidence) * 100)),
                "bbox": list(cell.bbox),
                "line": cell.row,
                "matrix": cell.matrix,
            }
        )
    return payload


def _estimate_confidence(cells) -> float:
    scored = [cell.confidence for cell in cells if cell.pattern != "000000"]
    if not scored:
        return 0.0
    return round(float(sum(scored) / len(scored)), 3)


def analyze_braille_image(image: np.ndarray, correct_perspective: bool = True) -> tuple[DetectionResult, dict[str, Any]]:
    preprocess = preprocess_frame(image, correct_perspective=correct_perspective)
    dots = detect_dots(preprocess.cleaned)
    detection = segment_braille_dots(dots, preprocess.working)
    debug = {
        "preprocessing": {
            "selected_variant": preprocess.selected_variant,
            "variants": preprocess.variant_scores,
        },
        "segmentation": detection.debug,
    }
    return detection, debug


def run_recognition(
    image: np.ndarray,
    include_annotated: bool = True,
    correct_perspective: bool = True,
) -> RecognitionResult:
    start = time.perf_counter()

    try:
        detection, debug = analyze_braille_image(image, correct_perspective=correct_perspective)
        decoded_lines, line_cell_chars = decode_lines_with_metadata(detection.patterns_by_line)
        full_text = "\n".join(line for line in decoded_lines if line.strip()).strip()

        if not full_text and any(pattern != "000000" for line in detection.patterns_by_line for pattern in line):
            full_text = " ".join("".join(chars).strip() for chars in line_cell_chars).strip()

        annotated_b64 = None
        if include_annotated and detection.annotated_image is not None:
            annotated_b64 = encode_image_to_base64(detection.annotated_image)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        confidence = _estimate_confidence(detection.cells)

        LOGGER.info(
            "Recognition complete: %d dots, %d cells, text=%r, confidence=%.2f",
            sum(len(cell.dot_centers) for cell in detection.cells),
            len(detection.cells),
            full_text,
            confidence,
        )

        return RecognitionResult(
            success=True,
            text=full_text,
            lines=decoded_lines,
            patterns_by_line=detection.patterns_by_line,
            cells=_cells_to_payload(detection.cells, line_cell_chars),
            dot_count=sum(len(cell.dot_centers) for cell in detection.cells),
            cell_count=len(detection.cells),
            processing_time_ms=round(elapsed_ms, 2),
            annotated_image_b64=annotated_b64,
            confidence=confidence,
            debug=debug,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        LOGGER.exception("Braille recognition failed")
        return RecognitionResult(
            success=False,
            text="",
            lines=[],
            patterns_by_line=[],
            cells=[],
            processing_time_ms=round(elapsed_ms, 2),
            error=str(exc),
            confidence=0.0,
        )
