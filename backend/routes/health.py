"""
Health check and diagnostics route.
"""

from fastapi import APIRouter
from pydantic import BaseModel
import cv2
import sys
import platform

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    python_version: str
    platform: str
    opencv_version: str
    message: str


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns backend health status and environment info."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        python_version=sys.version,
        platform=platform.platform(),
        opencv_version=cv2.__version__,
        message="BrailleVision backend is running."
    )


@router.get("/", tags=["System"])
async def root():
    return {
        "project": "BrailleVision",
        "description": "Real-time physical Braille recognition API",
        "endpoints": {
            "health": "/health",
            "upload_image": "POST /api/upload",
            "websocket_stream": "WS /ws/stream",
            "braille_reference": "GET /api/braille/reference"
        }
    }
