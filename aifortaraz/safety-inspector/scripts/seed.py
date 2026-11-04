"""Run the idempotent seed after applying migrations."""

import sys
import os
from pathlib import Path

from dotenv import load_dotenv

root = Path(__file__).resolve().parents[1]
load_dotenv(root / ".env")
if os.getenv("DATABASE_URL", "").startswith("sqlite:///./"):
    os.environ["DATABASE_URL"] = (
        "sqlite:///" + (root / os.environ["DATABASE_URL"][12:]).as_posix()
    )
sys.path.insert(0, str(root / "backend"))
from app.seed import seed  # noqa: E402

seed()
from app.database import SessionLocal  # noqa: E402
from app.models import Setting  # noqa: E402
from app.demo_seed import seed_demo_history  # noqa: E402

with SessionLocal() as db:
    enabled = db.get(Setting, "demo_mode")
    if enabled and enabled.value:
        print(seed_demo_history())
