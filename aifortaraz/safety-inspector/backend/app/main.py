import logging
import asyncio
from contextlib import suppress
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import jwt
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, text
from starlette.exceptions import HTTPException as StarletteHTTPException

from .dependencies import Db, current_user, require_roles
from .models import Area, IncidentType, PpeType, Setting, User
from .security import decode_token, issue_token, password_hasher
from .seed import seed
from .logging_utils import configure_logging

configure_logging()

logger = logging.getLogger("inspector")


@asynccontextmanager
async def lifespan(app):
    seed()
    from .ingest import heartbeat_monitor
    from .telegram import telegram_worker
    from .storage import snapshot_retention_worker

    workers = [
        asyncio.create_task(worker())
        for worker in (heartbeat_monitor, telegram_worker, snapshot_retention_worker)
    ]
    from .demo import controller

    demo_task = (
        asyncio.create_task(controller.run())
        if os.getenv("DEMO_AUTOSTART", "true").lower() == "true"
        else None
    )
    try:
        yield
    finally:
        for worker in workers:
            worker.cancel()
        for worker in workers:
            with suppress(asyncio.CancelledError):
                await worker
        if demo_task:
            demo_task.cancel()
            with suppress(asyncio.CancelledError):
                await demo_task


app = FastAPI(
    title="Digital Safety Inspector",
    version="0.7.0",
    description="Цифровой инспектор ОТ · Telegram, отчёты, акты и настройки.",
    docs_url=None,
    redoc_url=None,
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)
app.mount(
    "/api/swagger-assets",
    StaticFiles(directory=Path(__file__).resolve().parents[1] / "static/swagger"),
    name="swagger-assets",
)


@app.get("/api/docs", include_in_schema=False)
def swagger():
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",
        title="Digital Safety Inspector — API",
        swagger_js_url="/api/swagger-assets/swagger-ui-bundle.js",
        swagger_css_url="/api/swagger-assets/swagger-ui.css",
        swagger_favicon_url="data:,",
    )


import os  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:8080"
    ).split(","),
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    codes = {
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
    }
    message, details = str(exc.detail), {}
    if isinstance(exc.detail, dict):
        # Field-level problems raised by hand (the snapshot) keep the contract error shape.
        message = str(exc.detail.get("message", ""))
        details = {"fields": exc.detail.get("fields", [])}
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": codes.get(exc.status_code, "http_error"),
                "message": message,
                "details": details,
            }
        },
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    # Any unexpected failure still answers with the contract error body (contract 4.8).
    logger.error(
        "Unhandled error on %s %s", request.method, request.url.path, exc_info=exc
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "Внутренняя ошибка сервера. Подробности в журнале сервера",
                "details": {},
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Проверьте поля запроса",
                "details": {
                    "fields": [
                        {"location": list(e["loc"]), "message": e["msg"]}
                        for e in exc.errors()
                    ]
                },
            }
        },
    )


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    username: str
    full_name: str
    role: Literal["admin", "engineer", "master", "viewer"]
    area_id: str | None


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=256)


class RefreshIn(BaseModel):
    refresh_token: str


class Tokens(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserOut


def tokens(user):
    return Tokens(
        access_token=issue_token(user.id, "access"),
        refresh_token=issue_token(user.id, "refresh"),
        user=UserOut.model_validate(user),
    )


dummy_hash = password_hasher.hash("invalid-login-placeholder")


@app.post("/api/v1/auth/login", response_model=Tokens, tags=["auth"])
def login(body: LoginIn, db: Db):
    user = db.scalar(select(User).where(User.username == body.username))
    valid = password_hasher.verify(
        body.password, user.password_hash if user else dummy_hash
    )
    if not user or not valid:
        raise HTTPException(401, "Неверный логин или пароль")
    return tokens(user)


@app.post("/api/v1/auth/refresh", response_model=Tokens, tags=["auth"])
def refresh(body: RefreshIn, db: Db):
    try:
        user_id = decode_token(body.refresh_token, "refresh")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Сессия истекла")
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(401, "Пользователь не найден")
    return tokens(user)


@app.get("/api/v1/auth/me", response_model=UserOut, tags=["auth"])
def me(user: Annotated[User, Depends(current_user)]):
    return user


@app.get("/api/v1/users", response_model=list[UserOut], tags=["users"])
def users(db: Db, user: Annotated[User, Depends(require_roles("admin"))]):
    return db.scalars(select(User).order_by(User.username)).all()


@app.get("/api/v1/settings", tags=["settings"])
def settings(db: Db, user: Annotated[User, Depends(current_user)]):
    return {row.key: row.value for row in db.scalars(select(Setting))}


@app.get("/health", tags=["system"])
def health(db: Db):
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        logger.exception("Database health check failed")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "database": "unavailable",
                "connected_ai_modules": 0,
                "stage": 7,
            },
        )
    from .realtime import hub

    return {
        "status": "ok",
        "database": "ok",
        "connected_ai_modules": sum(ai for ai, _ in hub.clients.values()),
        "stage": 7,
    }


from .stage_two import router as stage_two_router  # noqa: E402
from .realtime import router as realtime_router  # noqa: E402

app.include_router(stage_two_router)
app.include_router(realtime_router)
from .ingest import router as ingest_router  # noqa: E402
from .incident_routes import router as incident_router  # noqa: E402
from .storage import SNAPSHOT_DIR  # noqa: E402

app.include_router(ingest_router)
app.include_router(incident_router)
app.mount("/static/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")
from .demo import router as demo_router  # noqa: E402
from .relay import router as relay_router  # noqa: E402
from .storage import DATA_DIR  # noqa: E402

app.include_router(demo_router)
app.include_router(relay_router)

from .analytics import router as analytics_router  # noqa: E402
from .plan import router as plan_router  # noqa: E402

app.include_router(analytics_router)
app.include_router(plan_router)
from .settings_admin import router as settings_router  # noqa: E402
from .telegram import router as telegram_router  # noqa: E402
from .reports import router as reports_router  # noqa: E402

app.include_router(settings_router)
app.include_router(telegram_router)
app.include_router(reports_router)
(DATA_DIR / "plans").mkdir(exist_ok=True)
app.mount("/static/plans", StaticFiles(directory=DATA_DIR / "plans"), name="plans")


@app.get("/api/v1/settings/{dictionary}", tags=["settings"])
def dictionaries(
    dictionary: Literal["ppe-types", "incident-types", "areas"],
    db: Db,
    user: Annotated[User, Depends(current_user)],
):
    model = {"ppe-types": PpeType, "incident-types": IncidentType, "areas": Area}[
        dictionary
    ]
    return [
        {c.name: getattr(row, c.name) for c in model.__table__.columns}
        for row in db.scalars(select(model))
    ]
