import asyncio
import io
import time

import jwt
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

from .database import SessionLocal
from .dependencies import Db, find_api_key
from .live_runtime import runtime
from .models import User
from .security import decode_token
from .stage_two import camera_or_404

router = APIRouter(tags=["stream"])


def validate_frame(jpeg: bytes):
    if len(jpeg) > 5 * 1024 * 1024:
        raise HTTPException(413, "Размер кадра превышает 5 МБ")
    try:
        with Image.open(io.BytesIO(jpeg)) as image:
            if image.format != "JPEG" or image.width * image.height > 16_000_000:
                raise ValueError("Expected a bounded JPEG")
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(422, "Некорректный JPEG-кадр")
    return jpeg


def stream_identity(request, token, db):
    credential = token
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        credential = authorization.split(" ", 1)[1]
    if credential:
        try:
            user_id = decode_token(credential, "access")
        except jwt.InvalidTokenError:
            raise HTTPException(401, "Недействительный токен потока")
        if db.get(User, user_id) is None:
            raise HTTPException(401, "Пользователь не найден")
        return ("jwt", credential)
    key = request.headers.get("x-api-key")
    if find_api_key(db, key):
        return ("key", key)
    raise HTTPException(401, "Для потока требуется JWT или API-ключ")


def valid_identity(identity):
    with SessionLocal() as db:
        if identity[0] == "key":
            return find_api_key(db, identity[1]) is not None
        try:
            return db.get(User, decode_token(identity[1], "access")) is not None
        except jwt.InvalidTokenError:
            return False


def mjpeg_part(jpeg: bytes):
    return (
        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
        + str(len(jpeg)).encode()
        + b"\r\n\r\n"
        + jpeg
        + b"\r\n"
    )


async def frame_stream(camera_id, request, identity):
    sequence = -1
    condition = runtime.conditions.setdefault(camera_id, asyncio.Condition())
    checked = time.monotonic()
    while not await request.is_disconnected():
        frame = runtime.frames.get(camera_id)
        if frame is None or time.monotonic() - frame.received_at > 5:
            break
        if time.monotonic() - checked >= 5:
            if not valid_identity(identity):
                break
            checked = time.monotonic()
        if frame.sequence != sequence:
            sequence = frame.sequence
            yield mjpeg_part(frame.jpeg)
            await asyncio.sleep(1 / 15)
        try:
            async with condition:
                await asyncio.wait_for(
                    condition.wait_for(
                        lambda: runtime.frames.get(camera_id) is None
                        or runtime.frames[camera_id].sequence != sequence
                    ),
                    timeout=5,
                )
        except asyncio.TimeoutError:
            break
    yield b"--frame--\r\n"


@router.get("/api/v1/stream/{camera_id}.mjpeg")
async def mjpeg(camera_id: str, request: Request, db: Db, token: str | None = None):
    identity = stream_identity(request, token, db)
    camera_or_404(db, camera_id)
    frame = runtime.frames.get(camera_id)
    if frame is None or time.monotonic() - frame.received_at > 5:
        raise HTTPException(
            503, "Кадры от ИИ-модуля ещё не поступили или поток прерван"
        )
    return StreamingResponse(
        frame_stream(camera_id, request, identity),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
