import { strict as assert } from "node:assert";
import { test } from "node:test";
import {
  HEARTBEAT_TIMEOUT_MS,
  isAiLive,
  isAiOffline,
} from "../src/features/live/status.ts";

const now = Date.parse("2026-11-04T12:00:10Z");
const camera = {
  ai_enabled: true,
  status: "online",
  last_heartbeat_at: "2026-11-04T12:00:05Z",
  source: "ai",
};

test("a fresh heartbeat from the real AI shows the AI badge", () => {
  assert.equal(isAiOffline(camera, now), false);
  assert.equal(isAiLive(camera, now), true);
  assert.equal(isAiLive({ ...camera, status: "degraded" }, now), true);
});

test("mock heartbeats and older backends without source never show it", () => {
  assert.equal(isAiLive({ ...camera, source: "mock" }, now), false);
  assert.equal(isAiLive({ ...camera, source: null }, now), false);
  const { source, ...older } = camera;
  assert.equal(source, "ai");
  assert.equal(isAiLive(older, now), false);
  assert.equal(isAiOffline(older, now), false);
});

test("an offline, stale or disabled camera is offline and has no AI badge", () => {
  const cases = [
    { ...camera, status: "offline" },
    { ...camera, ai_enabled: false },
    { ...camera, last_heartbeat_at: null },
    { ...camera, last_heartbeat_at: "not a timestamp" },
    {
      ...camera,
      last_heartbeat_at: new Date(now - HEARTBEAT_TIMEOUT_MS - 1).toISOString(),
    },
  ];
  for (const value of cases) {
    assert.equal(isAiOffline(value, now), true, JSON.stringify(value));
    assert.equal(isAiLive(value, now), false, JSON.stringify(value));
  }
});

test("the heartbeat timeout is inclusive at 15 seconds", () => {
  const edge = {
    ...camera,
    last_heartbeat_at: new Date(now - HEARTBEAT_TIMEOUT_MS).toISOString(),
  };
  assert.equal(HEARTBEAT_TIMEOUT_MS, 15000);
  assert.equal(isAiOffline(edge, now), false);
});
