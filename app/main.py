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
    """Load all AI models once on startup, release on shutdown."""
    logger.info("Loading AI models …")
    manager = ModelManager()
    manager.load_all()
    application.state.model_manager = manager
    logger.info("All models loaded ✓")
    yield
    logger.info("Shutting down – releasing models …")
    manager.unload_all()


app = FastAPI(
    title="EYES Vision API",
    version="1.0.0",
    description="Backend for the EYES assistive mobile application.",
    lifespan=lifespan,
)

# CORS – allow requests from the mobile app / web test client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(analyze.router, prefix="/api")
