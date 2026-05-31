"""
BrailleVision — FastAPI Backend Entry Point
Real-time physical Braille recognition system.
"""

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ─── Make sure imports resolve from backend/ directory ───
sys.path.insert(0, os.path.dirname(__file__))

from routes.health import router as health_router
from routes.upload import router as upload_router
from routes.stream import router as stream_router

# ─── App definition ───────────────────────────────────────
app = FastAPI(
    title="BrailleVision API",
    description=(
        "Real-time physical Braille recognition system. "
        "Converts camera images of embossed/handwritten Braille into English text and speech."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS — allow frontend (any origin during development) ─
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────
app.include_router(health_router)
app.include_router(upload_router)
app.include_router(stream_router)

# ─── Serve frontend static files and uploads ──────────────
from core.history_db import UPLOADS_DIR
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    @app.get("/app", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# ─── Startup / shutdown events ────────────────────────────
@app.on_event("startup")
async def on_startup():
    from core.ml_model import initialize_ml_model
    initialize_ml_model()
    
    print("=" * 55)
    print("  SPARSH Tactile Reader Started")
    print("  API docs : http://localhost:8000/docs")
    print("  Frontend : http://localhost:8000/app")
    print("  WebSocket: ws://localhost:8000/ws/stream")
    print("=" * 55)


@app.on_event("shutdown")
async def on_shutdown():
    print("[BrailleVision] Shutting down.")


# ─── Run directly ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[os.path.dirname(__file__)]
    )
