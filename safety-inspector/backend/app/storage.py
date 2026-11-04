import base64
import binascii
import io
import os
from pathlib import Path

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from .database import DATABASE_URL

MAX_SNAPSHOT_BYTES = 5 * 1024 * 1024
DATA_DIR = Path(
    os.getenv(
        "DATA_DIR",
        str(Path(DATABASE_URL.removeprefix("sqlite:///")).resolve().parent)
        if DATABASE_URL.startswith("sqlite:///")
        else "./data",
    )
).resolve()
SNAPSHOT_DIR = DATA_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def decode_snapshot(encoded: str | None):
    if encoded is None:
        return None
    if encoded.startswith("data:image/jpeg;base64,"):
        encoded = encoded.split(",", 1)[1]
    if len(encoded) > (MAX_SNAPSHOT_BYTES + 2) // 3 * 4:
        raise HTTPException(413, "Размер снимка превышает 5 МБ")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(422, "Снимок должен быть корректной base64-строкой JPEG")
    return raw


def save_snapshot(incident_id: str, raw: bytes | None):
    if raw is None:
        return None, None
    if len(raw) > MAX_SNAPSHOT_BYTES:
        raise HTTPException(413, "Размер снимка превышает 5 МБ")
    paths = [
        SNAPSHOT_DIR / f"{incident_id}.jpg",
        SNAPSHOT_DIR / f"{incident_id}-thumb.jpg",
    ]
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "JPEG" or image.width * image.height > 16_000_000:
                raise HTTPException(422, "Требуется JPEG размером до 16 мегапикселей")
            image.load()
            image = image.convert("RGB")
            # Re-encoding strips EXIF and any unrelated metadata.
            image.save(paths[0], format="JPEG", quality=88)
            image.thumbnail((320, 320))
            image.save(paths[1], format="JPEG", quality=82)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        for path in paths:
            path.unlink(missing_ok=True)
        raise HTTPException(422, "Не удалось прочитать JPEG-снимок")
    return f"/static/snapshots/{paths[0].name}", f"/static/snapshots/{paths[1].name}"


def remove_snapshot(incident_id: str):
    for suffix in (".jpg", "-thumb.jpg"):
        (SNAPSHOT_DIR / f"{incident_id}{suffix}").unlink(missing_ok=True)


def purge_expired_snapshots():
    from datetime import timedelta

    from sqlalchemy import or_, select

    from .database import SessionLocal
    from .models import Incident, utc_now
    from .stage_two import setting

    with SessionLocal.begin() as db:
        cutoff = utc_now() - timedelta(days=setting(db, "snapshot_retention_days", 30))
        rows = db.scalars(
            select(Incident)
            .where(
                Incident.created_at < cutoff,
                or_(
                    Incident.snapshot_path.is_not(None),
                    Incident.thumbnail_path.is_not(None),
                ),
            )
            .limit(500)
        )
        for row in rows:
            remove_snapshot(row.id)
            row.snapshot_path = row.thumbnail_path = None


async def snapshot_retention_worker():
    import asyncio
    import logging

    from starlette.concurrency import run_in_threadpool

    while True:
        await asyncio.sleep(3600)
        try:
            await run_in_threadpool(purge_expired_snapshots)
        except Exception:
            logging.getLogger("inspector").exception("Snapshot cleanup failed")
