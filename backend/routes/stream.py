"""
WebSocket streaming route — real-time Braille recognition from live camera feed.
Client sends base64-encoded frames, server responds with decoded text instantly.
"""

import json
import time
import base64
import asyncio
import numpy as np
import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.preprocessing import decode_base64_to_image
from core.recognition import run_recognition
from core.tts_engine import get_tts_engine

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        print(f"[WS] Client connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        print(f"[WS] Client disconnected. Total: {len(self.active)}")

    async def send_json(self, ws: WebSocket, data: dict):
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            pass


manager = ConnectionManager()

# Track last sent text per connection to avoid repeating same TTS
_last_text_per_ws: dict[int, str] = {}


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Real-time Braille recognition WebSocket.

    Protocol:
    ─────────────────────────────────────
    CLIENT → SERVER (JSON):
    {
      "type": "frame",
      "image": "<base64-encoded JPEG/PNG>",
      "include_annotated": true,       // optional
      "include_audio": false,          // optional
      "correct_perspective": true      // optional
    }

    OR:
    { "type": "ping" }

    SERVER → CLIENT (JSON):
    {
      "type": "result",
      "success": true,
      "text": "hello world",
      "lines": ["hello world"],
      "cell_count": 11,
      "dot_count": 33,
      "confidence": 0.91,
      "processing_time_ms": 45.2,
      "annotated_image_b64": "...",   // if requested
      "audio_b64": "...",             // if requested and text changed
      "timestamp": 1717000000.123
    }

    OR on error:
    { "type": "error", "message": "..." }

    OR:
    { "type": "pong" }
    ─────────────────────────────────────
    """
    await manager.connect(websocket)
    ws_id = id(websocket)
    _last_text_per_ws[ws_id] = ""

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": "Invalid JSON received."
                })
                continue

            msg_type = msg.get("type", "frame")

            # ── Ping / keep-alive ──
            if msg_type == "ping":
                await manager.send_json(websocket, {"type": "pong"})
                continue

            # ── Frame processing ──
            if msg_type == "frame":
                b64_image = msg.get("image")
                if not b64_image:
                    await manager.send_json(websocket, {
                        "type": "error",
                        "message": "No 'image' field in frame message."
                    })
                    continue

                include_annotated = msg.get("include_annotated", True)
                include_audio = msg.get("include_audio", False)
                correct_perspective = msg.get("correct_perspective", False)  # off by default for speed

                # Decode image
                try:
                    img = _decode_frame(b64_image)
                except Exception as e:
                    await manager.send_json(websocket, {
                        "type": "error",
                        "message": f"Could not decode image: {str(e)}"
                    })
                    continue

                # Run recognition (CPU-bound — run in executor to not block event loop)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    _run_sync_recognition,
                    img, include_annotated, correct_perspective
                )

                # TTS audio — only generate if text changed
                audio_b64 = None
                if include_audio and result.success and result.text.strip():
                    if result.text != _last_text_per_ws.get(ws_id, ""):
                        tts = get_tts_engine()
                        audio_b64 = await loop.run_in_executor(
                            None,
                            tts.synthesize_to_base64,
                            result.text
                        )

                _last_text_per_ws[ws_id] = result.text

                response = {
                    "type": "result",
                    "success": result.success,
                    "text": result.text,
                    "lines": result.lines,
                    "patterns_by_line": result.patterns_by_line,
                    "cells": result.cells,
                    "cell_count": result.cell_count,
                    "dot_count": result.dot_count,
                    "confidence": result.confidence,
                    "processing_time_ms": result.processing_time_ms,
                    "timestamp": time.time(),
                }

                if include_annotated and result.annotated_image_b64:
                    response["annotated_image_b64"] = result.annotated_image_b64

                if audio_b64:
                    response["audio_b64"] = audio_b64

                if result.error:
                    response["error"] = result.error

                await manager.send_json(websocket, response)

            else:
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        _last_text_per_ws.pop(ws_id, None)
    except Exception as e:
        print(f"[WS] Unexpected error: {e}")
        manager.disconnect(websocket)
        _last_text_per_ws.pop(ws_id, None)


def _decode_frame(b64_string: str) -> np.ndarray:
    """Decode base64 image string to OpenCV BGR array."""
    # Strip data URI prefix if present (e.g. "data:image/jpeg;base64,...")
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    data = base64.b64decode(b64_string)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode returned None — invalid image data.")
    return img


def _run_sync_recognition(img: np.ndarray, include_annotated: bool, correct_perspective: bool):
    """Synchronous recognition wrapper for executor."""
    return run_recognition(img, include_annotated=include_annotated, correct_perspective=correct_perspective)
