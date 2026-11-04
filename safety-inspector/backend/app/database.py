import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/inspector.db")
if DATABASE_URL.startswith("sqlite:///"):
    Path(DATABASE_URL.removeprefix("sqlite:///")).parent.mkdir(
        parents=True, exist_ok=True
    )
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 10}
    if DATABASE_URL.startswith("sqlite")
    else {},
)
if DATABASE_URL.startswith("sqlite"):

    def local_datetime(value, zone):
        stamp = datetime.fromisoformat(value)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(ZoneInfo(zone)).isoformat(sep=" ", timespec="seconds")[
            :19
        ]

    @event.listens_for(engine, "connect")
    def configure_sqlite(connection, _):
        connection.create_function(
            "local_datetime", 2, local_datetime, deterministic=True
        )
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")


SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    with SessionLocal() as session:
        yield session
