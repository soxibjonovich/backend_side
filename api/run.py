from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.openapi.docs import get_swagger_ui_html
from contextlib import asynccontextmanager

from api.models import APIStatusResponse
from api.config import Config
from api.routers.stt import stt_router
from api.routers.live import websocket_router
from api.routers import logic as transcription_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - replaces deprecated startup/shutdown events.
    Loads model on startup, cleans up on shutdown.
    """
    # Startup
    print("🚀 Loading Whisper model...")
    transcription_service.load_model()
    print("✅ Model loaded and ready!")

    yield

    # Shutdown
    print("🛑 Cleaning up model resources...")
    transcription_service.cleanup_model()
    print("✅ Cleanup completed!")


app = FastAPI(
    title="Uzbek Whisper API",
    version="2.0.0",
    description="Whisper-based STT API with batch and real-time transcription",
    lifespan=lifespan,
)


# Include routers
app.include_router(stt_router)  # Batch API: /api/*
app.include_router(websocket_router)  # Live STT: /live/*


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    """Custom Swagger UI with favicon"""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Uzbek Whisper API",
        swagger_favicon_url="/favicon.ico",
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve favicon"""
    return FileResponse("api/static/perplexity-color.png")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    model_status = transcription_service.get_model_status()

    return {
        "service": "Uzbek Whisper API",
        "version": "2.0.0",
        "status": Config.STATUS.value,
        "model_loaded": model_status["loaded"],
        "device": model_status["device"],
        "endpoints": {
            "batch_transcription": "/api/transcribe",
            "live_transcription": "/live/transcribe",
            "test_client": "/live/test-client",
            "api_docs": "/docs",
            "health": "/api/health",
        },
    }


@app.get("/status", response_model=APIStatusResponse)
async def status():
    """API status endpoint"""
    return APIStatusResponse(status=Config.STATUS.value)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
