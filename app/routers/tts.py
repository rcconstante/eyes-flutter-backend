"""ElevenLabs text-to-speech proxy endpoint."""

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.config import settings

router = APIRouter()


class TtsRequest(BaseModel):
    text: str
    voiceId: str | None = None
    speed: float | None = None


@router.post("/tts")
async def text_to_speech(payload: TtsRequest):
    """Return ElevenLabs audio for frontend playback."""
    if not settings.ELEVENLABS_API_KEY:
        raise HTTPException(status_code=503, detail="ElevenLabs API key is not configured")

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing text")

    voice_id = payload.voiceId or settings.ELEVENLABS_VOICE_ID
    request_body = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
    }

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        elevenlabs_response = await client.post(
            url,
            headers={
                "xi-api-key": settings.ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json=request_body,
        )

    if elevenlabs_response.status_code >= 400:
        raise HTTPException(
            status_code=elevenlabs_response.status_code,
            detail=elevenlabs_response.text,
        )

    return Response(
        content=elevenlabs_response.content,
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store"},
    )