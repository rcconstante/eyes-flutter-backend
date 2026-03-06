"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


# Some reverse-proxy / Railway configs probe the root path.
@router.get("/")
async def root():
    return {"status": "ok", "service": "EYES Vision API"}
