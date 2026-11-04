"""Exercise the external AI contract using synthetic metadata and optional video frames."""

import argparse
import asyncio
import base64
import getpass
import io
import json
import math
import os
import shutil
import sys
import time
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import httpx
import websockets
from PIL import Image, ImageDraw
from websockets.exceptions import WebSocketException

SIZE = (960, 540)


def position():
    return 0.3 + 0.18 * math.sin(time.monotonic() * 0.4)


def detect(frame):
    """Replace this function with a real model in the external AI process."""
    return [
        {
            "track_id": 88,
            "label": "person",
            "bbox": [position(), 0.25, 0.16, 0.5],
            "conf": 0.96,
            "ppe": {"helmet": True, "vest": True, "glasses": False, "gloves": True},
            "zones": [],
            "state": "violation",
        }
    ]


def synthetic_frame():
    image = Image.new("RGB", SIZE, "#233941")
    draw = ImageDraw.Draw(image)
    for column in range(0, SIZE[0], 60):
        draw.line((column, 0, column, SIZE[1]), fill="#34535c")
    draw.rectangle((25, 25, 935, 515), outline="#89c9b1", width=3)
    draw.text((45, 45), "EXTERNAL AI CLIENT / SYNTHETIC TRACK #88", fill="#d8f1e5")
    left = int(position() * SIZE[0])
    draw.ellipse((left + 35, 135, left + 115, 215), fill="#dbc56d")
    draw.rounded_rectangle((left + 15, 220, left + 135, 405), radius=14, fill="#71a995")
    return image


def encode_jpeg(frame):
    output = io.BytesIO()
    frame.save(output, "JPEG", quality=80)
    return base64.b64encode(output.getvalue()).decode("ascii")


def ffmpeg_command(args):
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("Для --video / --webcam нужен FFmpeg в PATH")
    if args.video:
        path = Path(args.video).resolve()
        if not path.is_file():
            raise RuntimeError("Видеофайл не найден")
        source = ["-stream_loop", "-1", "-i", str(path)]
    else:
        if sys.platform == "win32":
            source = ["-f", "dshow", "-i", f"video={args.webcam}"]
        elif sys.platform == "darwin":
            source = ["-f", "avfoundation", "-i", f"{args.webcam}:none"]
        else:
            source = ["-f", "v4l2", "-i", args.webcam]
    return [
        executable,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        *source,
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"fps={args.fps},scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]


@asynccontextmanager
async def frame_reader(args):
    if not args.video and not args.webcam:

        async def generated():
            return synthetic_frame()

        yield generated
        return
    process = await asyncio.create_subprocess_exec(
        *ffmpeg_command(args), stdout=asyncio.subprocess.PIPE
    )

    async def capture():
        try:
            raw = await asyncio.wait_for(
                process.stdout.readexactly(SIZE[0] * SIZE[1] * 3), 12
            )
        except (asyncio.IncompleteReadError, TimeoutError):
            raise RuntimeError(
                "FFmpeg не передал кадр. Проверьте файл или устройство"
            ) from None
        return Image.frombytes("RGB", SIZE, raw)

    try:
        yield capture
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await process.stdout.read()
        await process.wait()


async def run(args, key):
    parts = urlsplit(args.host)
    address = urlunsplit(
        (
            "wss" if parts.scheme == "https" else "ws",
            parts.netloc,
            parts.path.rstrip("/") + "/ws/ingest",
            urlencode({"api_key": key}),
            "",
        )
    )
    config_url = f"api/v1/ingest/config/{quote(args.camera, safe='')}"
    async with httpx.AsyncClient(
        base_url=args.host.rstrip("/") + "/", headers={"X-API-Key": key}, timeout=10
    ) as http:

        async def refresh_config():
            response = await http.get(config_url)
            response.raise_for_status()
            config = response.json()
            print(
                f"Камера: {config['camera']['name']}, конфиг {config['version']}",
                flush=True,
            )

        await refresh_config()
        async with (
            frame_reader(args) as get_frame,
            websockets.connect(address) as socket,
        ):

            async def emit(message_type, **fields):
                await socket.send(
                    json.dumps(
                        {"type": message_type, "camera_id": args.camera, **fields}
                    )
                )

            async def receive():
                async for text in socket:
                    message = json.loads(text)
                    if message.get("type") == "ping":
                        await socket.send(json.dumps({"type": "pong"}))
                    elif (
                        message.get("type") == "config.updated"
                        and message.get("camera_id") == args.camera
                    ):
                        await refresh_config()
                    elif message.get("type") == "error":
                        raise RuntimeError(
                            f"Сервер отклонил сообщение: {message.get('message', '')}"
                        )

            receiver = asyncio.create_task(receive())
            started = time.monotonic()
            heartbeat_at = event_at = frame_at = float("-inf")
            frame_id = 0
            try:
                while not args.duration or time.monotonic() - started < args.duration:
                    if receiver.done():
                        receiver.result()
                        raise RuntimeError("Сервер закрыл WebSocket")
                    tick = time.monotonic()
                    if tick - heartbeat_at >= 5:
                        await emit(
                            "heartbeat",
                            fps=args.fps,
                            model_name="external-contract-client",
                            model_version="0.7",
                            status="ok",
                            extra={"synthetic_metadata": True},
                        )
                        heartbeat_at = tick
                    frame = await get_frame()
                    timestamp = datetime.now(timezone.utc).isoformat()
                    objects = detect(frame)
                    await emit(
                        "detections", ts=timestamp, frame_id=frame_id, objects=objects
                    )
                    event_due = args.events and tick - event_at >= 30
                    frame_due = args.push_frames and tick - frame_at >= 0.19
                    jpeg = encode_jpeg(frame) if event_due or frame_due else None
                    if frame_due:
                        await emit("frame", ts=timestamp, jpeg_base64=jpeg)
                        frame_at = tick
                    if event_due:
                        event = await http.post(
                            "api/v1/ingest/events",
                            json={
                                "external_id": f"external-test-{uuid4()}",
                                "camera_id": args.camera,
                                "type": "ppe_violation",
                                "confidence": 0.96,
                                "track_id": 88,
                                "started_at": timestamp,
                                "details": {
                                    "missing_ppe": ["glasses"],
                                    "present_ppe": ["helmet", "vest", "gloves"],
                                    "boxes": [
                                        {
                                            key: obj[key]
                                            for key in (
                                                "track_id",
                                                "label",
                                                "bbox",
                                                "conf",
                                            )
                                        }
                                        for obj in objects
                                    ],
                                    "frame_size": list(SIZE),
                                },
                                "snapshot_base64": jpeg,
                            },
                        )
                        event.raise_for_status()
                        print(
                            f"Инцидент: HTTP {event.status_code}, {event.json()['id']}",
                            flush=True,
                        )
                        event_at = tick
                    frame_id += 1
                    await asyncio.sleep(
                        max(0, 1 / args.fps - (time.monotonic() - tick))
                    )
            finally:
                receiver.cancel()
                with suppress(asyncio.CancelledError):
                    await receiver


def main():
    parser = argparse.ArgumentParser(description="Тестовый клиент внешнего ИИ")
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--camera", required=True, help="UUID камеры")
    parser.add_argument(
        "--api-key", help="API-ключ ИИ. Можно использовать переменную AI_API_KEY"
    )
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--push-frames", action="store_true")
    parser.add_argument("--events", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--duration", type=float, default=0, help="Секунды. 0 до Ctrl+C"
    )
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--video", help="Путь к локальному видеофайлу")
    sources.add_argument(
        "--webcam",
        help="Устройство FFmpeg: имя Windows, индекс macOS или /dev/video0 Linux",
    )
    args = parser.parse_args()
    if (
        not 0.5 <= args.fps <= 15
        or not math.isfinite(args.duration)
        or args.duration < 0
    ):
        parser.error("FPS: от 0.5 до 15. Длительность должна быть неотрицательной.")
    try:
        host = urlsplit(args.host)
        valid_host = (
            host.scheme in ("http", "https")
            and host.hostname
            and not (host.username or host.password or host.query or host.fragment)
        )
        _ = host.port
    except ValueError:
        valid_host = False
    if not valid_host:
        parser.error("Укажите базовый HTTP(S) URL без учётных данных и параметров")
    key = args.api_key or os.getenv("AI_API_KEY") or os.getenv("INGEST_API_KEY")
    if not key:
        key = getpass.getpass("API-ключ ИИ (скрыто): ").strip()
    if not key:
        parser.error("Требуется API-ключ ИИ")
    try:
        asyncio.run(run(args, key))
    except KeyboardInterrupt:
        return
    except httpx.HTTPStatusError as error:
        parser.exit(
            1,
            f"API вернул HTTP {error.response.status_code}. Проверьте камеру и ключ.\n",
        )
    except (httpx.RequestError, WebSocketException, OSError):
        parser.exit(
            1, "Ошибка подключения. Проверьте адрес API, ключ и источник кадров.\n"
        )
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
