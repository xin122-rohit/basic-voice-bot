from dotenv import load_dotenv

load_dotenv()
import asyncio
import os
from typing import List
from datetime import datetime, date
from pydantic import BaseModel
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware

from livekit_bot.generate_token import create_token
from fastapi import (
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    APIRouter,
    HTTPException,
)
import uuid

router = APIRouter()


@router.get("/")
async def health_check():
    return {"status": "ok"}


@router.get("/get-token")
def get_token(email: str = Query(), room: str = Query(None)):
    """
    Issue a LiveKit participant token.

    Session-isolation contract:
    - Frontends should always pass a per-tab isolated room name (e.g.
      `voice-<emailTag>-<sessionId>`) so that different users and different
      browser tabs never share a room.
    - If no room is supplied (legacy / health-check callers), we generate a
      random one instead of using a shared constant like "insurance-voice".
    - The resolved room is echoed back in the response so callers can verify
      which room they were granted.
    """
    if not room or not room.strip():
        # Generate an isolated random room rather than falling back to a shared name.
        room = f"voice-anon-{uuid.uuid4().hex[:12]}"

    room = room.strip()

    # Warn in server logs when a generic/shared room name slips through.
    # This should not happen with a properly upgraded frontend.
    if room in ("banking-voice", "loan-"):
        import logging

        logging.getLogger(__name__).warning(
            "[get-token] Shared fallback room requested for email=%s room=%s — "
            "upgrade the frontend to use isolated per-tab room names.",
            email,
            room,
        )

    token = create_token(email=email, room=room)

    return {"token": token, "identity": email, "room": room}
