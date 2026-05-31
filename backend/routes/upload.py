"""
Image upload route — accepts a single image, runs Braille recognition,
returns decoded text + annotated image + optional TTS audio.
"""

import io
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from core.recognition import run_recognition
from core.tts_engine import get_tts_engine
from core.braille_mapper import get_all_patterns
from core.preprocessing import decode_base64_to_image
from core.history_db import save_upload_record, get_all_records, clear_all_history

router = APIRouter()


class CellPrediction(BaseModel):
    index: int
    pattern: str
    char: str
    confidence: float
    confidence_pct: int
    bbox: list[int]
    line: int = 0


class RecognitionResponse(BaseModel):
    success: bool
    text: str
    lines: list[str]
    patterns_by_line: list[list[str]]
    cells: list[CellPrediction] = []
    dot_count: int
    cell_count: int
    processing_time_ms: float
    confidence: float
    annotated_image_b64: Optional[str] = None
    audio_b64: Optional[str] = None
    error: Optional[str] = None


async def _read_image_from_upload(file: UploadFile) -> np.ndarray:
    """Read an uploaded file into an OpenCV BGR numpy array."""
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    arr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Ensure it is a valid JPG/PNG.")
    return img


@router.post("/api/upload", response_model=RecognitionResponse, tags=["Recognition"])
async def upload_image(
    file: UploadFile = File(..., description="Braille image (JPG or PNG)"),
    include_annotated: bool = Query(True, description="Include annotated image in response"),
    include_audio: bool = Query(False, description="Include TTS audio (base64 MP3) in response"),
    correct_perspective: bool = Query(True, description="Attempt perspective/tilt correction"),
):
    """
    Upload a Braille image and receive decoded English text.

    - **file**: JPG or PNG image of physical Braille
    - **include_annotated**: Returns annotated image with detected dots overlaid
    - **include_audio**: Returns TTS audio bytes (base64 encoded MP3)
    - **correct_perspective**: Corrects camera tilt automatically
    """
    img = await _read_image_from_upload(file)

    result = run_recognition(
        img,
        include_annotated=include_annotated,
        correct_perspective=correct_perspective
    )

    audio_b64 = None
    if include_audio and result.success and result.text.strip():
        tts = get_tts_engine()
        audio_b64 = tts.synthesize_to_base64(result.text)

    # Save to local offline history index
    if result.success:
        ann_img = None
        if result.annotated_image_b64:
            try:
                ann_img = decode_base64_to_image(result.annotated_image_b64)
            except Exception:
                pass
        
        save_upload_record(
            original_img=img,
            annotated_img=ann_img,
            text=result.text,
            confidence=result.confidence,
            cell_count=result.cell_count,
            dot_count=result.dot_count,
            processing_time_ms=result.processing_time_ms
        )

    return RecognitionResponse(
        success=result.success,
        text=result.text,
        lines=result.lines,
        patterns_by_line=result.patterns_by_line,
        cells=result.cells,
        dot_count=result.dot_count,
        cell_count=result.cell_count,
        processing_time_ms=result.processing_time_ms,
        confidence=result.confidence,
        annotated_image_b64=result.annotated_image_b64 if include_annotated else None,
        audio_b64=audio_b64,
        error=result.error,
    )


@router.get("/api/history", tags=["History"])
async def get_history():
    """Retrieve all logged historical recognition records."""
    return get_all_records()


@router.delete("/api/history", tags=["History"])
async def clear_history():
    """Clear all saved history records and image files."""
    success = clear_all_history()
    return {"success": success, "message": "History database successfully wiped." if success else "Failed to clear history."}


@router.get("/api/braille/reference", tags=["Reference"])
async def braille_reference():
    """
    Returns the full Braille dot-pattern to character lookup table.
    Useful for debugging and frontend reference charts.
    """
    return {
        "description": "Braille Grade 1 pattern reference. Each key is a 6-bit string (dot1..dot6), value is the character.",
        "patterns": get_all_patterns()
    }
