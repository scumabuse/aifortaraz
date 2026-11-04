import type { Box } from "../incidents/types";
export interface DetectionObject extends Box {
  ppe?: Record<string, boolean>;
  zones?: string[];
  state?: "ok" | "violation" | "danger";
}
export interface DetectionFrame {
  type: "detections";
  camera_id: string;
  ts: string;
  frame_id: number;
  objects: DetectionObject[];
  source?: "ai" | "mock";
  receivedAt: number;
}
export interface Overlays {
  boxes: boolean;
  zones: boolean;
  trackIds: boolean;
  labels: boolean;
  violationsOnly: boolean;
}
export interface DemoStatus {
  enabled: boolean;
  rate: number;
  seeding: boolean;
  error: string | null;
  active_camera_ids: string[];
  real_camera_ids: string[];
}
