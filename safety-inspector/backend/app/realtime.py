"""Authenticated realtime metadata hub with per-camera detection subscriptions."""

import asyncio
import time
from contextlib import suppress

import jwt
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from .database import SessionLocal
from .dependencies import find_api_key
from .incident_schemas import DetectionsIn, HeartbeatIn, FrameIn
from .live_runtime import runtime
from .storage import decode_snapshot
from .models import Camera, User
from .security import decode_token

router = APIRouter()


class LiveHub:
    def __init__(self):
        self.clients: dict[asyncio.Queue, tuple[bool, str]] = {}
        self.users: dict[asyncio.Queue, str] = {}
        self.subscriptions: dict[asyncio.Queue, set[str]] = {}
        self.latest_detections: dict[str, dict] = {}
        self.pending_detections: dict[asyncio.Queue, dict[str, dict]] = {}

    def enqueue(self, queue, message):
        if queue.full():
            # Reconnect forces a fresh REST snapshot instead of silently losing alerts.
            while not queue.empty():
                queue.get_nowait()
            queue.put_nowait({"type": "_overflow"})
        else:
            queue.put_nowait(message)

    async def publish(self, camera_id: str, version: int):
        message = {"type": "config.updated", "camera_id": camera_id, "version": version}
        for queue in list(self.clients):
            self.enqueue(queue, message)

    async def broadcast(self, message: dict, recipient: str | None = None):
        for queue, (ai, _) in list(self.clients.items()):
            if ai or (recipient is not None and self.users.get(queue) != recipient):
                continue
            if message["type"] == "detections":
                camera_id = message["camera_id"]
                if camera_id not in self.subscriptions.get(queue, set()):
                    continue
                pending = self.pending_detections[queue]
                if camera_id not in pending:
                    self.enqueue(queue, {"type": "_detection", "camera_id": camera_id})
                pending[camera_id] = message
            else:
                self.enqueue(queue, message)


hub = LiveHub()


async def serve(websocket: WebSocket, ai: bool):
    credential_id, user_id = "", ""
    with SessionLocal() as db:
        if ai:
            key = find_api_key(db, websocket.query_params.get("api_key"))
            if key is None:
                await websocket.close(code=4401)
                return
            credential_id = key.id
        else:
            try:
                user_id = decode_token(
                    websocket.query_params.get("token", ""), "access"
                )
            except jwt.InvalidTokenError:
                await websocket.close(code=4401)
                return
            if db.get(User, user_id) is None:
                await websocket.close(code=4401)
                return
    await websocket.accept()
    queue = asyncio.Queue(maxsize=64)
    hub.clients[queue] = (ai, credential_id)
    hub.users[queue] = user_id
    hub.subscriptions[queue] = set()
    hub.pending_detections[queue] = {}
    last_received = time.monotonic()

    def authenticated():
        with SessionLocal() as db:
            if ai:
                return (
                    find_api_key(db, websocket.query_params.get("api_key")) is not None
                )
            try:
                return (
                    db.get(
                        User,
                        decode_token(websocket.query_params.get("token", ""), "access"),
                    )
                    is not None
                )
            except jwt.InvalidTokenError:
                return False

    async def send():
        next_ping = time.monotonic() + 20
        while True:
            try:
                message = await asyncio.wait_for(
                    queue.get(), timeout=max(0.01, next_ping - time.monotonic())
                )
            except asyncio.TimeoutError:
                message = None
            if not authenticated():
                await websocket.close(code=4401)
                return
            if time.monotonic() - last_received > 60:
                await websocket.close(code=4408)
                return
            if message and message["type"] == "_overflow":
                await websocket.close(code=1013)
                return
            if time.monotonic() >= next_ping:
                await websocket.send_json({"type": "ping"})
                next_ping = time.monotonic() + 20
            if message:
                if message["type"] == "_detection":
                    message = hub.pending_detections[queue].pop(
                        message["camera_id"], None
                    )
                if message is not None:
                    await websocket.send_json(message)

    async def receive():
        nonlocal last_received
        while True:
            message = await websocket.receive_json()
            last_received = time.monotonic()
            if not isinstance(message, dict):
                hub.enqueue(
                    queue, {"type": "error", "message": "Expected a JSON object"}
                )
                continue
            kind = message.get("type")
            if kind == "ping":
                hub.enqueue(queue, {"type": "pong"})
            elif kind == "pong":
                continue
            elif not ai and kind in ("subscribe", "unsubscribe"):
                cameras = message.get("cameras")
                if (
                    not isinstance(cameras, list)
                    or len(cameras) > 100
                    or not all(isinstance(c, str) for c in cameras)
                ):
                    hub.enqueue(
                        queue,
                        {"type": "error", "message": "Invalid camera subscriptions"},
                    )
                    continue
                if kind == "subscribe":
                    hub.subscriptions[queue].update(cameras)
                    for camera_id in cameras:
                        if camera_id in hub.latest_detections:
                            hub.enqueue(queue, hub.latest_detections[camera_id])
                else:
                    hub.subscriptions[queue].difference_update(cameras)
                    for camera_id in cameras:
                        hub.pending_detections[queue].pop(camera_id, None)
            elif ai and kind in ("detections", "heartbeat", "frame"):
                if not authenticated():
                    await websocket.close(code=4401)
                    return
                try:
                    if kind == "detections":
                        frame = DetectionsIn.model_validate(message)
                        with SessionLocal() as db:
                            if db.get(Camera, frame.camera_id) is None:
                                raise HTTPException(404, "Camera not found")
                        result = frame.model_dump(mode="json")
                        result["source"] = "ai"
                        runtime.mark_real(frame.camera_id)
                        hub.latest_detections[frame.camera_id] = result
                        await hub.broadcast(result)
                    elif kind == "frame":
                        from .relay import validate_frame
                        from starlette.concurrency import run_in_threadpool

                        frame = FrameIn.model_validate(message)
                        with SessionLocal() as db:
                            if db.get(Camera, frame.camera_id) is None:
                                raise HTTPException(404, "Camera not found")
                        jpeg = await run_in_threadpool(
                            validate_frame, decode_snapshot(frame.jpeg_base64)
                        )
                        await runtime.push_frame(frame.camera_id, jpeg)
                    else:
                        from .ingest import save_heartbeat

                        body = HeartbeatIn.model_validate(
                            {k: v for k, v in message.items() if k != "type"}
                        )
                        with SessionLocal() as db:
                            status = save_heartbeat(db, body)
                        await hub.broadcast({"type": "camera.status", **status})
                except (ValidationError, HTTPException):
                    hub.enqueue(
                        queue,
                        {
                            "type": "error",
                            "message": "Invalid metadata or unknown camera",
                        },
                    )
            else:
                hub.enqueue(
                    queue, {"type": "error", "message": "Unsupported message type"}
                )

    tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except (WebSocketDisconnect, ValueError, RuntimeError):
        pass
    finally:
        hub.clients.pop(queue, None)
        hub.users.pop(queue, None)
        hub.subscriptions.pop(queue, None)
        hub.pending_detections.pop(queue, None)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(
                asyncio.CancelledError, WebSocketDisconnect, RuntimeError, ValueError
            ):
                await task


@router.websocket("/ws/live")
async def browser_socket(websocket: WebSocket):
    await serve(websocket, False)


@router.websocket("/ws/ingest")
async def ai_socket(websocket: WebSocket):
    await serve(websocket, True)
