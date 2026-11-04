"""Ephemeral ownership and last-JPEG state, never persisted as video."""

import asyncio
import time
from dataclasses import dataclass


@dataclass
class RelayFrame:
    jpeg: bytes
    sequence: int
    received_at: float


class LiveRuntime:
    def __init__(self):
        self.real_activity: dict[str, float] = {}
        self.frames: dict[str, RelayFrame] = {}
        self.conditions: dict[str, asyncio.Condition] = {}
        self.demo_resetting = False

    def mark_real(self, camera_id: str, age: float = 0):
        self.real_activity[camera_id] = time.monotonic() - max(0, age)

    def real_active(self, camera_id: str):
        return time.monotonic() - self.real_activity.get(camera_id, -1e20) < 60

    async def push_frame(self, camera_id: str, jpeg: bytes):
        previous = self.frames.get(camera_id)
        self.frames[camera_id] = RelayFrame(
            jpeg, previous.sequence + 1 if previous else 1, time.monotonic()
        )
        self.mark_real(camera_id)
        condition = self.conditions.setdefault(camera_id, asyncio.Condition())
        async with condition:
            condition.notify_all()

    def clear_camera(self, camera_id: str):
        self.frames.pop(camera_id, None)
        self.real_activity.pop(camera_id, None)


runtime = LiveRuntime()
