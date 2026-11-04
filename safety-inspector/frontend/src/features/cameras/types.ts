export type StreamType = "rtsp" | "mjpeg" | "hls" | "file" | "webcam";
export interface CameraInput {
  name: string;
  area_id: string;
  stream_type: StreamType;
  stream_url: string;
  preview_url: string | null;
  plan_x: number | null;
  plan_y: number | null;
  description: string | null;
  enabled: boolean;
  ai_enabled: boolean;
}
export interface CameraRecord extends CameraInput {
  id: string;
  created_at: string;
  area_name: string;
  status: string;
  fps: number | null;
  model_name: string | null;
  model_version: string | null;
  last_heartbeat_at: string | null;
  zones_count: number;
  incidents_7d: number;
  source?: "ai" | "mock" | null;
}
export type Point = [number, number];
export type ZoneType = "ppe" | "danger" | "vehicle" | "counting";
export interface Schedule {
  days: number[];
  from: string;
  to: string;
}
export interface Rules {
  required_ppe: string[];
  entry_forbidden: boolean;
  max_persons: number | null;
  min_vehicle_distance_m: number | null;
  schedule: Schedule[] | null;
  confirm_frames: number | null;
  cooldown_sec: number | null;
  severity_override: "low" | "medium" | "high" | "critical" | null;
  regulation_ref: string | null;
  responsible_user_id: string | null;
}
export interface ZoneRecord {
  id: string;
  camera_id?: string;
  name: string;
  color: string;
  zone_type: ZoneType;
  polygon: Point[];
  rules: Rules;
  enabled: boolean;
  sort_order: number;
  updated_at?: string;
}
export interface Config {
  camera: { id: string; name: string; stream_url: string; stream_type: string };
  frame_size_hint: [number, number];
  thresholds: {
    confirm_frames: number;
    cooldown_sec: number;
    min_confidence: number;
  };
  ppe_types: string[];
  zones: {
    id: string;
    name: string;
    type: ZoneType;
    polygon: Point[];
    rules: Omit<Rules, "regulation_ref" | "responsible_user_id">;
  }[];
  version: number;
  updated_at: string;
}
export const blankRules = (): Rules => ({
  required_ppe: [],
  entry_forbidden: false,
  max_persons: null,
  min_vehicle_distance_m: null,
  schedule: null,
  confirm_frames: null,
  cooldown_sec: null,
  severity_override: null,
  regulation_ref: null,
  responsible_user_id: null,
});
