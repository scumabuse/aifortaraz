from datetime import date
from io import BytesIO
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from .analytics import Period, incident_rows, period_bounds
from .dependencies import Db
from .models import Area, Camera, FloorPlan, Incident, Zone
from .realtime import hub
from .schemas import StrictModel
from .stage_two import Editor, Reader, bump_config, camera_or_404, iso, status_payload
from .storage import DATA_DIR

router = APIRouter(prefix="/api/v1/plan", tags=["plan"])
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def plan_image(row):
    return dict(url=row.image_path, width=row.width, height=row.height) if row else None


@router.get("")
def plans(
    db: Db,
    user: Reader,
    period: Period = "7d",
    date_from: date | None = None,
    date_to: date | None = None,
    area_id: str | None = None,
):
    start, end, *_ = period_bounds(db, period, date_from, date_to)
    areas_query = select(Area).order_by(Area.name)
    if area_id:
        areas_query = areas_query.where(Area.id == area_id)
    areas = db.scalars(areas_query).all()
    area_ids = [area.id for area in areas]
    cameras = db.scalars(
        select(Camera).where(Camera.area_id.in_(area_ids)).order_by(Camera.name)
    ).all()
    camera_ids = [camera.id for camera in cameras]
    rows = incident_rows(start, end, camera_ids)
    totals = dict(
        db.execute(
            select(rows.c.camera_id, func.count()).group_by(rows.c.camera_id)
        ).all()
    )
    latest = dict(
        db.execute(
            select(Incident.camera_id, func.max(Incident.started_at))
            .where(Incident.camera_id.in_(camera_ids))
            .group_by(Incident.camera_id)
        ).all()
    )
    zone_lists = {camera_id: [] for camera_id in camera_ids}
    for zone in db.scalars(
        select(Zone)
        .where(Zone.camera_id.in_(camera_ids), Zone.enabled)
        .order_by(Zone.sort_order)
    ):
        zone_lists[zone.camera_id].append(
            dict(id=zone.id, name=zone.name, color=zone.color)
        )
    images = {
        row.area_id: plan_image(row)
        for row in db.scalars(select(FloorPlan).where(FloorPlan.area_id.in_(area_ids)))
    }
    markers = {area_id: [] for area_id in area_ids}
    for camera in cameras:
        markers[camera.area_id].append(
            dict(
                id=camera.id,
                name=camera.name,
                x=camera.plan_x,
                y=camera.plan_y,
                status=status_payload(db, camera)["status"],
                count=totals.get(camera.id, 0),
                latest=iso(latest.get(camera.id)),
                zones=zone_lists[camera.id],
            )
        )
    return dict(
        start=iso(start),
        end=iso(end),
        areas=[
            dict(
                id=area.id,
                name=area.name,
                image=images.get(area.id),
                cameras=markers[area.id],
            )
            for area in areas
        ],
    )


@router.put("/{area_id}/image")
def upload_plan(
    area_id: str, file: Annotated[UploadFile, File()], db: Db, user: Editor
):
    if db.get(Area, area_id) is None:
        raise HTTPException(404, "Участок не найден")
    raw = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Размер плана превышает 10 МБ")
    try:
        with Image.open(BytesIO(raw)) as source:
            if (
                source.format not in ("PNG", "JPEG")
                or source.width * source.height > 16_000_000
            ):
                raise ValueError
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        SyntaxError,
    ):
        raise HTTPException(422, "Загрузите PNG или JPEG до 16 мегапикселей")
    row = db.scalar(
        select(FloorPlan).where(FloorPlan.area_id == area_id).with_for_update()
    )
    if row is None:
        row = FloorPlan(area_id=area_id)
        db.add(row)
    previous = row.image_path
    name = f"upload-{uuid4()}.png"
    path = DATA_DIR / "plans" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        image.save(path)
        row.image_path = f"/static/plans/{name}"
        row.width, row.height = image.size
        db.commit()
    except IntegrityError:
        path.unlink(missing_ok=True)
        raise HTTPException(409, "План был изменён. Повторите загрузку")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if previous and previous.startswith("/static/plans/upload-"):
        (path.parent / previous.rsplit("/", 1)[-1]).unlink(missing_ok=True)
    return plan_image(row)


class Position(StrictModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


@router.patch("/cameras/{camera_id}")
async def move_camera(camera_id: str, body: Position, db: Db, user: Editor):
    camera = camera_or_404(db, camera_id)
    camera.plan_x, camera.plan_y = body.x, body.y
    version = bump_config(db)
    db.commit()
    await hub.publish(camera_id, version)
    return body
