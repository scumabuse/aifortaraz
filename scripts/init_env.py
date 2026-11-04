"""`make init`: create the root .env, generate its secrets and copy it into both repositories.

The root .env is the single source of the AI API key and the URLs (integration spec 7.2): the
website reads SEED_AI_API_KEY (it seeds its "ai-module" key with that value), the AI module reads
SAFETY_AI_API_KEY (the same value). Both repositories load a file named .env from their own root,
so the root file is copied there byte for byte (symlinks need admin rights on Windows).

    python scripts/init_env.py            # create or complete .env, then copy it
    python scripts/init_env.py --rotate   # generate a new AI key (the website re-seeds it on start)
    python scripts/init_env.py --check    # exit 1 when .env is missing, incomplete or out of sync
"""

from __future__ import annotations

import argparse
import secrets
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import AI, ROOT_ENV, ROOT_ENV_EXAMPLE, WEBSITE, random_key, read_env, render_env  # noqa: E402

COPIES = (WEBSITE / ".env", AI / ".env")
REQUIRED = ("SEED_AI_API_KEY", "SAFETY_AI_API_KEY", "SAFETY_AI_BACKEND_URL", "JWT_SECRET", "AI_CAMERAS",
            "AI_MJPEG_BASE_URL", "TZ")


def problems() -> list[str]:
    """What is wrong with the current .env setup (empty list when everything is in order)."""
    found: list[str] = []
    if not ROOT_ENV.exists():
        return [f"{ROOT_ENV} is missing — run `make init`"]
    values = read_env(ROOT_ENV)
    for key in REQUIRED:
        if not values.get(key):
            found.append(f"{key} is empty in .env")
    if values.get("SEED_AI_API_KEY") != values.get("SAFETY_AI_API_KEY"):
        found.append("SAFETY_AI_API_KEY differs from SEED_AI_API_KEY")
    if len(values.get("SEED_AI_API_KEY", "")) < 32:
        found.append("SEED_AI_API_KEY is shorter than 32 characters")
    if len(values.get("JWT_SECRET", "")) < 32:
        found.append("JWT_SECRET is shorter than 32 characters")
    root_bytes = ROOT_ENV.read_bytes()
    for copy in COPIES:
        if not copy.exists():
            found.append(f"{copy} is missing — run `make init`")
        elif copy.read_bytes() != root_bytes:
            found.append(f"{copy} differs from the root .env — run `make init`")
    return found


def init(rotate: bool) -> int:
    """Create or complete .env, then copy it into both repositories."""
    template = read_env(ROOT_ENV_EXAMPLE)
    current = read_env(ROOT_ENV) if ROOT_ENV.exists() else {}
    values = {**template, **current}
    if rotate or len(values.get("SEED_AI_API_KEY", "")) < 32:
        values["SEED_AI_API_KEY"] = random_key(32)
        print("generated a new AI API key (SEED_AI_API_KEY = SAFETY_AI_API_KEY)")
    values["SAFETY_AI_API_KEY"] = values["SEED_AI_API_KEY"]
    if len(values.get("JWT_SECRET", "")) < 32:
        values["JWT_SECRET"] = secrets.token_hex(32)
        print("generated JWT_SECRET")
    ROOT_ENV.write_text(render_env(ROOT_ENV_EXAMPLE, values), encoding="utf-8")
    for copy in COPIES:
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT_ENV, copy)
        print(f"copied .env -> {copy.relative_to(ROOT_ENV.parent)}")
    remaining = problems()
    for line in remaining:
        print(f"problem: {line}", file=sys.stderr)
    return 1 if remaining else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the root .env and copy it into both repositories.")
    parser.add_argument("--rotate", action="store_true", help="generate a new AI API key")
    parser.add_argument("--check", action="store_true", help="only report problems (exit 1 when any)")
    args = parser.parse_args()
    if args.check:
        found = problems()
        for line in found:
            print(f"problem: {line}")
        print("env: ok" if not found else f"env: {len(found)} problem(s)")
        return 1 if found else 0
    return init(args.rotate)


if __name__ == "__main__":
    sys.exit(main())
