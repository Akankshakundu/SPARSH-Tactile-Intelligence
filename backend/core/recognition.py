"""
Recognition pipeline — preprocessing + dot detection + braille mapping.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from .preprocessing import preprocess_frame, encode_image_to_base64
from .dot_detector import detect_braille
from .braille_mapper import decode_lines, decode_from_cell_chars, pattern_to_char


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


def _cells_to_payload(cells) -> list[dict[str, Any]]:
    payload = []
    for i, c in enumerate(cells):
        ch = c.char
        if ch in ("?", "·") and c.pattern != "000000":
            ch = pattern_to_char(c.pattern)
        payload.append(
            {
                "index": i,
                "pattern": c.pattern,
                "char": ch if c.pattern != "000000" else " ",
                "confidence": round(float(c.confidence), 3),
                "confidence_pct": int(round(float(c.confidence) * 100)),
                "bbox": list(c.bbox),
                "line": c.row,
            }
        )
    return payload


def run_recognition(
    image: np.ndarray,
    include_annotated: bool = True,
    correct_perspective: bool = True,
) -> RecognitionResult:
    t_start = time.perf_counter()

    try:
        prep = preprocess_frame(image, correct_perspective=correct_perspective)
        # Use upscaled working frame so dot coordinates match the annotated overlay
        detection = detect_braille(prep.cleaned, prep.working)

        # Primary decode: use ML-classified per-cell characters
        full_text = decode_from_cell_chars(detection.cells)

        # Fallback: pattern-based decode if char pipeline is empty but cells exist
        if not full_text.strip() and detection.patterns_by_line:
            decoded_lines = decode_lines(detection.patterns_by_line)
            full_text = "\n".join(decoded_lines).strip()

        decoded_lines = [full_text] if full_text else []
        if detection.patterns_by_line and len(detection.patterns_by_line) > 1:
            decoded_lines = [
                decode_from_cell_chars(line_cells)
                for line_cells in detection.lines
            ]
            full_text = "\n".join(l for l in decoded_lines if l.strip()).strip() or full_text

        annotated_b64 = None
        if include_annotated and detection.annotated_image is not None:
            annotated_b64 = encode_image_to_base64(detection.annotated_image)

        t_end = time.perf_counter()
        ms = (t_end - t_start) * 1000

        dot_count = sum(len(c.dot_centers) for c in detection.cells)

        return RecognitionResult(
            success=True,
            text=full_text,
            lines=decoded_lines if decoded_lines else ([full_text] if full_text else []),
            patterns_by_line=detection.patterns_by_line,
            cells=_cells_to_payload(detection.cells),
            dot_count=dot_count,
            cell_count=len(detection.cells),
            processing_time_ms=round(ms, 2),
            annotated_image_b64=annotated_b64,
            confidence=_estimate_confidence(detection.cells),
        )

    except Exception as e:
        t_end = time.perf_counter()
        ms = (t_end - t_start) * 1000
        return RecognitionResult(
            success=False,
            text="",
            lines=[],
            patterns_by_line=[],
            cells=[],
            dot_count=0,
            cell_count=0,
            processing_time_ms=round(ms, 2),
            error=str(e),
            confidence=0.0,
        )


def _estimate_confidence(cells) -> float:
    """Mean per-cell classifier confidence (ignoring blank spaces)."""
    if not cells:
        return 0.0
    scored = [c.confidence for c in cells if c.pattern != "000000"]
    if not scored:
        return 0.0
    return round(float(sum(scored) / len(scored)), 3)
