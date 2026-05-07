"""
EYES Backend – FastAPI entry point.

Provides a single /api/analyze endpoint that receives a camera frame,
runs low-light enhancement (Zero-DCE), object detection (YOLOv8s),
depth estimation (MiDaS), and returns a structured JSON response
matching the Flutter ResultModel.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.model_manager import ModelManager
from app.routers import analyze, health

logger = logging.getLogger("eyes")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize model manager (lazy loading on first request)."""
    logger.info("Initializing model manager (lazy loading enabled)")
    manager = ModelManager()
    application.state.model_manager = manager
    logger.info("Server ready ✓")
    yield
    logger.info("Shutting down – releasing models …")
    manager.unload_all()


app = FastAPI(
    title="EYES Vision API",
    version="1.0.0",
    description="Backend for the EYES assistive mobile application.",
    lifespan=lifespan,
)

# CORS - Cross Origin Resource Sharing
# List of allowed origins for the API to be accessed from
_ALLOWED_ORIGINS = [
    "https://eyes-web-app.netlify.app",
    "https://eyes-web-application.netlify.app",
    "http://localhost:5173",
    "http://localhost:4173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,          # no cookies / auth headers needed
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(analyze.router, prefix="/api")
