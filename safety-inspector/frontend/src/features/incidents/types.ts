export type IncidentStatus =
  "new" | "acknowledged" | "resolved" | "false_positive";
export type Severity = "low" | "medium" | "high" | "critical";
export interface Box {
  track_id?: number | null;
  label: string;
  bbox: [number, number, number, number];
  conf: number;
}
export interface IncidentRecord {
  id: string;
  external_id: string;
  source: string;
  camera_id: string;
  camera_name: string;
  area_id: string;
  area_name: string;
  zone_id: string | null;
  zone_name: string | null;
  type_code: string;
  type_name: string;
  severity: Severity;
  status: IncidentStatus;
  confidence: number;
  track_id: number | null;
  started_at: string;
  ended_at: string | null;
  duration_sec: number | null;
  snapshot_url: string | null;
  thumbnail_url: string | null;
  clip_url: string | null;
  details: {
    missing_ppe?: string[];
    present_ppe?: string[];
    boxes?: Box[];
    frame_size?: [number, number];
    title?: string;
    [key: string]: unknown;
  };
  assignee_id: string | null;
  assignee_name: string | null;
  acknowledged_at: string | null;
  acknowledged_by: string | null;
  acknowledged_name: string | null;
  resolved_at: string | null;
  resolved_by: string | null;
  resolved_name: string | null;
  resolution_note: string | null;
  false_positive_reason: string | null;
  created_at: string;
  reaction_sec: number | null;
}
export interface IncidentPage {
  items: IncidentRecord[];
  total: number;
  page: number;
  page_size: number;
}
export interface IncidentComment {
  id: string;
  user_id: string;
  user_name: string;
  text: string;
  created_at: string;
}
export interface TimelineEvent {
  id: string;
  event_type: string;
  user_name: string | null;
  payload: {
    note?: string;
    reason?: string;
    text?: string;
    channel?: string;
    user_id?: string | null;
    [key: string]: unknown;
  };
  created_at: string;
}
export interface IncidentDetail extends IncidentRecord {
  timeline: TimelineEvent[];
  comments: IncidentComment[];
  previous_id: string | null;
  next_id: string | null;
}
export interface Assignee {
  id: string;
  full_name: string;
  role: string;
  area_id: string | null;
}
export interface InboxItem {
  id: string;
  incident_id: string;
  type_code: string;
  severity: Severity;
  text: string;
  camera_name: string;
  created_at: string;
  read_at: string | null;
}
export interface Inbox {
  items: InboxItem[];
  unread_count: number;
  total: number;
  page: number;
  page_size: number;
}
export interface Summary {
  incidents_total: number;
  open_incidents: number;
  critical_incidents: number;
  mean_reaction_sec: number | null;
  compliance_percent: number | null;
  online_cameras: number;
  total_cameras: number;
}
