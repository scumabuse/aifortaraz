import type { CameraRecord } from "../cameras/types";

export const HEARTBEAT_TIMEOUT_MS = 15000;
export type CameraLiveStatus = Pick<
  CameraRecord,
  "ai_enabled" | "status" | "last_heartbeat_at" | "source"
>;

/** No usable AI signal: analysis disabled, no heartbeat yet, a stale heartbeat or status offline. */
export function isAiOffline(camera: CameraLiveStatus, now: number): boolean {
  if (!camera.ai_enabled || !camera.last_heartbeat_at) return true;
  const age = now - Date.parse(camera.last_heartbeat_at);
  return !(age <= HEARTBEAT_TIMEOUT_MS) || camera.status === "offline";
}
/** The camera's live status comes from the real AI module (heartbeat source "ai") and it is online. */
export function isAiLive(camera: CameraLiveStatus, now: number): boolean {
  return camera.source === "ai" && !isAiOffline(camera, now);
}
