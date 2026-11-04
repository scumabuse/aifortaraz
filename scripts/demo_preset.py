"""`make demo-preset`: presentation preset against the running website (integration spec 7.7).

Idempotent — a second run changes nothing and says so:
  1. thresholds confirm_frames=6, cooldown_sec=20 (a violation can be repeated on stage every 20 s);
  2. demo mode enabled (mocks for the cameras without AI; they pause while the AI sends heartbeats);
  3. AI camera cards (AI_CAMERAS) point to the AI's MJPEG: stream_type mjpeg,
     <AI_MJPEG_BASE_URL>/stream/<camera>.mjpeg, ai_enabled;
  4. cam-1/cam-2 zones exist — loaded from safety-ai/tools/demo_zones_cam1.yaml / _cam2.yaml through
     the zones API when the camera has none;
  5. Telegram: when TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS are set in .env, the bot is configured
     and a test message is queued;
  6. prints the pre-flight checklist of docs/DEMO_SCRIPT.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ADMIN_LOGIN, AI, BACKEND_PORT, ROOT, ROOT_ENV, read_env  # noqa: E402

ZONE_FILES = {"cam-1": AI / "tools" / "demo_zones_cam1.yaml", "cam-2": AI / "tools" / "demo_zones_cam2.yaml"}
CAMERA_FIELDS = ("name", "area_id", "stream_type", "stream_url", "preview_url", "plan_x", "plan_y",
                 "description", "enabled", "ai_enabled")
RULE_FIELDS = ("required_ppe", "entry_forbidden", "max_persons", "min_vehicle_distance_m", "schedule",
               "confirm_frames", "cooldown_sec", "severity_override")


def website_zone(zone: dict[str, Any], order: int) -> dict[str, Any]:
    """Contract zone (days 1..7, 1 = Monday) → the website's zone editor format (days 0..6)."""
    rules = {key: (zone.get("rules") or {}).get(key) for key in RULE_FIELDS}
    rules["required_ppe"] = rules["required_ppe"] or []
    rules["entry_forbidden"] = bool(rules["entry_forbidden"])
    if rules["schedule"]:
        rules["schedule"] = [{"days": [d - 1 for d in w["days"]], "from": w["from"], "to": w["to"]}
                             for w in rules["schedule"]]
    return {"name": zone.get("name") or zone["id"], "zone_type": zone.get("type", "ppe"),
            "polygon": zone["polygon"], "rules": rules, "enabled": True, "sort_order": order}


def apply(url: str, user: str, password: str, confirm_frames: int, cooldown: int) -> list[str]:
    """Apply the preset; returns the list of changes made (empty when everything was in place)."""
    env = read_env(ROOT_ENV)
    changes: list[str] = []
    with httpx.Client(base_url=url, timeout=15) as client:
        login = client.post("/api/v1/auth/login", json={"username": user, "password": password})
        login.raise_for_status()
        auth = {"Authorization": f"Bearer {login.json()['access_token']}"}
        ai_key = {"X-API-Key": env.get("SAFETY_AI_API_KEY", "")}
        cams = [c.strip() for c in env.get("AI_CAMERAS", "cam-1,cam-2").split(",") if c.strip()]

        current = client.get(f"/api/v1/ingest/config/{cams[0]}", headers=ai_key)
        current.raise_for_status()
        thresholds = current.json()["thresholds"]
        if (thresholds["confirm_frames"], thresholds["cooldown_sec"]) != (confirm_frames, cooldown):
            client.put("/api/v1/settings", headers=auth, json={
                "confirm_frames": confirm_frames, "cooldown_sec": cooldown,
                "min_confidence": thresholds["min_confidence"]}).raise_for_status()
            changes.append(f"thresholds confirm_frames={confirm_frames}, cooldown_sec={cooldown}")

        demo = client.get("/api/v1/demo", headers=auth)
        demo.raise_for_status()
        if not demo.json().get("enabled"):
            client.put("/api/v1/demo", headers=auth,
                       json={"enabled": True, "rate": float(demo.json().get("rate") or 1.0)}).raise_for_status()
            changes.append("demo mode enabled")

        base = env.get("AI_MJPEG_BASE_URL", "http://localhost:8001").rstrip("/")
        for cam in cams:
            card = client.get(f"/api/v1/cameras/{cam}", headers=auth)
            card.raise_for_status()
            body = {key: card.json().get(key) for key in CAMERA_FIELDS}
            wanted = {"stream_type": "mjpeg", "stream_url": f"{base}/stream/{cam}.mjpeg", "ai_enabled": True}
            if any(body.get(k) != v for k, v in wanted.items()):
                client.put(f"/api/v1/cameras/{cam}", headers=auth, json={**body, **wanted}).raise_for_status()
                changes.append(f"{cam}: camera card -> {wanted['stream_url']}")

        for cam, path in ZONE_FILES.items():
            if cam not in cams:
                continue
            zones = client.get(f"/api/v1/cameras/{cam}/zones", headers=auth)
            zones.raise_for_status()
            if zones.json():
                continue
            spec = yaml.safe_load(path.read_text(encoding="utf-8"))
            payload = [website_zone(z, i) for i, z in enumerate(spec.get("zones") or [])]
            client.put(f"/api/v1/cameras/{cam}/zones", headers=auth, json=payload).raise_for_status()
            changes.append(f"{cam}: {len(payload)} zone(s) loaded from {path.relative_to(ROOT)}")

        token, chats = env.get("TELEGRAM_BOT_TOKEN", ""), env.get("TELEGRAM_CHAT_IDS", "")
        if token and chats:
            recipients = [{"chat_id": c.strip(), "label": f"demo {i + 1}"} for i, c in enumerate(chats.split(",")) if c.strip()]
            state = client.get("/api/v1/settings/telegram", headers=auth)
            state.raise_for_status()
            rules = {**state.json().get("rules", {}), "enabled": True, "base_url": "http://localhost:5173"}
            if not state.json().get("has_token") or state.json().get("recipients") != recipients or not state.json()["rules"].get("enabled"):
                client.put("/api/v1/settings/telegram", headers=auth,
                           json={"token": token, "recipients": recipients, "rules": rules}).raise_for_status()
                changes.append("telegram bot configured")
            client.post("/api/v1/settings/telegram/test", headers=auth).raise_for_status()
            print("telegram: test message queued (result in Settings → Notifications → journal)")
    return changes


def checklist() -> str:
    """The pre-flight section of docs/DEMO_SCRIPT.md."""
    script = ROOT / "docs" / "DEMO_SCRIPT.md"
    if not script.exists():
        return ""
    text = script.read_text(encoding="utf-8")
    start = text.find("## Pre-flight")
    end = text.find("\n## ", start + 1)
    return text[start:end if end > 0 else None].strip() if start >= 0 else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Presentation preset for the running website.")
    env = read_env(ROOT_ENV)
    parser.add_argument("--url", default=env.get("SAFETY_AI_BACKEND_URL", f"http://127.0.0.1:{BACKEND_PORT}"))
    parser.add_argument("--user", default=ADMIN_LOGIN[0])
    parser.add_argument("--password", default=ADMIN_LOGIN[1])
    parser.add_argument("--confirm-frames", type=int, default=6)
    parser.add_argument("--cooldown", type=int, default=20)
    args = parser.parse_args()
    try:
        changes = apply(args.url.rstrip("/"), args.user, args.password, args.confirm_frames, args.cooldown)
    except httpx.HTTPError as exc:
        print(f"demo-preset: the website at {args.url} is not reachable or refused: {exc}", file=sys.stderr)
        return 1
    if changes:
        print("demo-preset: applied\n  - " + "\n  - ".join(changes))
    else:
        print("demo-preset: already applied (no changes)")
    text = checklist()
    if text:
        print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
