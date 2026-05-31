from routes.health import router as health_router
from routes.upload import router as upload_router
from routes.stream import router as stream_router

__all__ = ["health_router", "upload_router", "stream_router"]
