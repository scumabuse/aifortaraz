import asyncio
import logging
from datetime import time, timedelta
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import Field, field_validator, model_validator
from sqlalchemy import func, select, update
from starlette.concurrency import run_in_threadpool

from .database import SessionLocal
from .dependencies import Db
from .models import (
    Incident,
    IncidentType,
    NotificationSettings,
    PpeType,
    TelegramDelivery,
    utc_now,
)
from .schemas import StrictModel
from .settings_admin import http_base_url, local_date
from .stage_two import Editor, iso
from .storage import SNAPSHOT_DIR

router = APIRouter(prefix="/api/v1/settings/telegram", tags=["telegram"])
logger = logging.getLogger("inspector")
MAX_ATTEMPTS = 4
SEVERITY = {
    "low": "низкая",
    "medium": "средняя",
    "high": "высокая",
    "critical": "критическая",
}


class Recipient(StrictModel):
    chat_id: str = Field(pattern=r"^-?[1-9][0-9]{0,18}$")
    label: str = Field(min_length=1, max_length=100)


class Rules(StrictModel):
    enabled: bool = False
    include_demo: bool = False
    types: list[str] = Field(default_factory=list, max_length=100)
    severities: list[Literal["low", "medium", "high", "critical"]] = Field(
        default_factory=list, max_length=4
    )
    quiet_start: time | None = None
    quiet_end: time | None = None
    cooldown_sec: int = Field(default=120, ge=0, le=86400)
    base_url: str = Field(default="", max_length=400)

    @field_validator("base_url")
    @classmethod
    def valid_url(cls, value):
        return http_base_url(value)

    @model_validator(mode="after")
    def validate_quiet_hours(self):
        if (self.quiet_start is None) != (self.quiet_end is None):
            raise ValueError("Укажите обе границы тихих часов")
        for value in (self.quiet_start, self.quiet_end):
            if value and value.tzinfo:
                raise ValueError("Тихие часы задаются в часовом поясе предприятия")
        return self


class TelegramIn(StrictModel):
    token: str | None = Field(
        default=None, max_length=200, pattern=r"^(?:[0-9]+:[A-Za-z0-9_-]{20,})?$"
    )
    recipients: list[Recipient] = Field(max_length=30)
    rules: Rules


def configuration(row):
    return {
        "has_token": bool(row.telegram_bot_token),
        "recipients": row.chat_ids,
        "rules": Rules.model_validate(row.rules),
    }


@router.get("")
def get_telegram(db: Db, user: Editor):
    return configuration(db.get(NotificationSettings, 1))


@router.put("")
def save_telegram(body: TelegramIn, db: Db, user: Editor):
    row = db.get(NotificationSettings, 1)
    token = row.telegram_bot_token if body.token is None else body.token or None
    ids = {recipient.chat_id for recipient in body.recipients}
    if len(ids) != len(body.recipients):
        raise HTTPException(422, "Chat ID получателей должны быть уникальными")
    if body.rules.enabled and (not token or not ids or not body.rules.base_url):
        raise HTTPException(
            422, "Для автоотправки укажите токен, получателя и базовый URL"
        )
    if any(db.get(IncidentType, code) is None for code in body.rules.types):
        raise HTTPException(422, "В фильтре есть неизвестный тип инцидента")
    row.telegram_bot_token = token
    row.chat_ids = [recipient.model_dump() for recipient in body.recipients]
    row.rules = body.rules.model_dump(mode="json")
    db.commit()
    return configuration(row)


def delivery_payload(row):
    return dict(
        id=row.id,
        channel="telegram",
        incident_id=row.incident_id,
        recipient=row.recipient,
        label=row.label,
        status=row.status,
        attempts=row.attempts,
        error=row.error,
        created_at=iso(row.created_at),
        sent_at=iso(row.sent_at),
    )


class TestIn(StrictModel):
    chat_id: str


@router.post("/test", status_code=202)
def test_message(body: TestIn, db: Db, user: Editor):
    config = db.get(NotificationSettings, 1)
    chat = next(
        (chat for chat in config.chat_ids if chat["chat_id"] == body.chat_id), None
    )
    if not config.telegram_bot_token or not chat:
        raise HTTPException(422, "Сначала сохраните токен и получателя")
    row = TelegramDelivery(
        recipient=chat["chat_id"],
        label=chat["label"],
        status="pending",
        next_attempt_at=utc_now(),
    )
    db.add(row)
    db.commit()
    return delivery_payload(row)


@router.get("/journal")
def journal(db: Db, user: Editor, page: int = Query(1, ge=1, le=100000)):
    rows = db.scalars(
        select(TelegramDelivery)
        .order_by(TelegramDelivery.created_at.desc(), TelegramDelivery.id)
        .offset((page - 1) * 25)
        .limit(25)
    )
    return dict(
        items=[delivery_payload(row) for row in rows],
        total=db.scalar(select(func.count()).select_from(TelegramDelivery)),
    )


def is_quiet(start, end, moment):
    if start is None or start == end:
        return False
    return start <= moment < end if start < end else moment >= start or moment < end


def enqueue_incident(db, incident):
    config = db.get(NotificationSettings, 1)
    if not config or not config.telegram_bot_token or not config.rules.get("enabled"):
        return
    rules = Rules.model_validate(config.rules)
    if incident.source == "mock" and not rules.include_demo:
        return
    if rules.types and incident.type_code not in rules.types:
        return
    if rules.severities and incident.severity not in rules.severities:
        return
    now = utc_now()
    quiet = incident.severity != "critical" and is_quiet(
        rules.quiet_start, rules.quiet_end, local_date(db, now).time()
    )
    for chat in config.chat_ids:
        error = "Тихие часы" if quiet else None
        if (
            not error
            and rules.cooldown_sec
            and db.scalar(
                select(TelegramDelivery.id)
                .where(
                    TelegramDelivery.camera_id == incident.camera_id,
                    TelegramDelivery.type_code == incident.type_code,
                    TelegramDelivery.recipient == chat["chat_id"],
                    TelegramDelivery.created_at
                    >= now - timedelta(seconds=rules.cooldown_sec),
                    TelegramDelivery.status.in_(("pending", "sent")),
                )
                .limit(1)
            )
        ):
            error = "Кулдаун камеры и типа"
        db.add(
            TelegramDelivery(
                incident_id=incident.id,
                camera_id=incident.camera_id,
                type_code=incident.type_code,
                recipient=chat["chat_id"],
                label=chat["label"],
                status="skipped" if error else "pending",
                error=error,
                next_attempt_at=None if error else now,
            )
        )


def claim():
    from .incidents import serialize_incident

    now = utc_now()
    with SessionLocal.begin() as db:
        row = db.scalar(
            select(TelegramDelivery)
            .where(
                TelegramDelivery.status == "pending",
                TelegramDelivery.next_attempt_at <= now,
            )
            .order_by(TelegramDelivery.next_attempt_at, TelegramDelivery.id)
            .limit(1)
        )
        if row is None:
            return None
        attempt = row.attempts + 1
        exhausted = attempt > MAX_ATTEMPTS
        values = (
            {
                "status": "failed",
                "error": "Доставка прервалась на последней попытке",
                "next_attempt_at": None,
            }
            if exhausted
            else {"attempts": attempt, "next_attempt_at": now + timedelta(minutes=2)}
        )
        updated = db.execute(
            update(TelegramDelivery)
            .where(
                TelegramDelivery.id == row.id,
                TelegramDelivery.status == "pending",
                TelegramDelivery.next_attempt_at <= now,
                TelegramDelivery.attempts == row.attempts,
            )
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        if not updated.rowcount or exhausted:
            return None
        config = db.get(NotificationSettings, 1)
        token = config.telegram_bot_token
        incident = db.get(Incident, row.incident_id) if row.incident_id else None
        rules = Rules.model_validate(config.rules)
        recipient_exists = any(
            chat["chat_id"] == row.recipient for chat in config.chat_ids
        )
        if not token or not recipient_exists or (incident and not rules.enabled):
            row.status, row.error = (
                "skipped",
                "Отправка отключена или получатель удалён",
            )
            row.next_attempt_at = None
            return None
        photo = None
        text = "Цифровой инспектор ОТ. Тестовое сообщение."
        if incident:
            item = serialize_incident(db, incident)
            names = {p.code: p.name_ru for p in db.scalars(select(PpeType))}
            missing = ", ".join(
                names.get(code, code) for code in item["details"].get("missing_ppe", [])
            )
            stamp = local_date(db, incident.started_at).strftime("%d.%m.%Y %H:%M")
            link = f"{rules.base_url}/incidents/{incident.id}"
            text = (
                f"⚠ {item['type_name']} · {item['camera_name']} → {item['zone_name'] or 'без зоны'} · "
                f"{stamp} · критичность: {SEVERITY[incident.severity]}"
            )
            if missing:
                text += f" · нет: {missing}"
            text = text[: max(1, 1020 - len(link))] + "\n" + link
            if incident.snapshot_path:
                try:
                    photo = (SNAPSHOT_DIR / f"{incident.id}.jpg").read_bytes()
                except OSError:
                    pass
        return row.id, attempt, token, row.recipient, text, photo


async def send(client, token, recipient, text, photo):
    method = "sendPhoto" if photo else "sendMessage"
    url = f"https://api.telegram.org/bot{token}/{method}"
    try:
        if photo:
            response = await client.post(
                url,
                data={"chat_id": recipient, "caption": text},
                files={"photo": ("incident.jpg", photo, "image/jpeg")},
            )
        else:
            response = await client.post(url, json={"chat_id": recipient, "text": text})
    except httpx.RequestError:
        return "Сеть Telegram недоступна или истекло время ожидания", 0
    try:
        body = response.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    if response.is_success and body.get("ok") is True:
        return None, None
    code = body.get("error_code", response.status_code)
    if code == 429:
        params = body.get("parameters")
        retry = params.get("retry_after", 0) if isinstance(params, dict) else 0
        return "Telegram ограничил частоту отправки", retry if isinstance(
            retry, int
        ) else 0
    if response.status_code >= 500 or isinstance(code, int) and code >= 500:
        return "Временная ошибка Telegram", 0
    message = {
        400: "Telegram отклонил запрос. Проверьте chat ID и снимок",
        401: "Telegram отклонил токен бота",
        403: "У бота нет доступа к чату",
    }.get(code if isinstance(code, int) else 0, "Telegram вернул неожиданный ответ")
    return message, None


def finish(delivery_id, attempt, error, retry):
    with SessionLocal.begin() as db:
        row = db.get(TelegramDelivery, delivery_id)
        if row is None or row.status != "pending" or row.attempts != attempt:
            return
        now = utc_now()
        row.error = error
        if error is None:
            row.status, row.sent_at, row.next_attempt_at = "sent", now, None
        elif retry is not None and attempt < MAX_ATTEMPTS:
            row.next_attempt_at = now + timedelta(seconds=max(retry, 2**attempt))
        else:
            row.status, row.next_attempt_at = "failed", None


async def telegram_worker():
    async with httpx.AsyncClient(timeout=8, follow_redirects=False) as client:
        while True:
            try:
                job = await run_in_threadpool(claim)
                if job:
                    delivery_id, attempt, *args = job
                    error, retry = await send(client, *args)
                    await run_in_threadpool(finish, delivery_id, attempt, error, retry)
                else:
                    await asyncio.sleep(2)
            except Exception:
                logger.error("Telegram queue processing failed")
                await asyncio.sleep(5)
