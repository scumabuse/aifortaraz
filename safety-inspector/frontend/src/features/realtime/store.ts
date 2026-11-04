import { create } from "zustand";
import type { IncidentRecord } from "../incidents/types";
import type { DetectionFrame } from "../live/types";
interface Toast {
  id: string;
  incident: IncidentRecord;
}
interface LiveState {
  connection: "connecting" | "connected" | "reconnecting" | "offline";
  critical: IncidentRecord[];
  toasts: Toast[];
  clock: number;
  sound: boolean;
  needsUnlock: boolean;
  detections: Record<string, DetectionFrame>;
  visibleCameras: string[];
  setVisibleCameras: (cameras: string[]) => void;
  setConnection: (value: LiveState["connection"]) => void;
  setCritical: (rows: IncidentRecord[]) => void;
  receiveIncident: (incident: IncidentRecord, created: boolean) => void;
  removeToast: (id: string) => void;
  tick: () => void;
  setSound: (value: boolean) => void;
  setNeedsUnlock: (value: boolean) => void;
  reset: () => void;
}
export const useLiveStore = create<LiveState>((set) => ({
  connection: "connecting",
  critical: [],
  toasts: [],
  clock: Date.now(),
  sound: localStorage.getItem("alarm-sound") === "true",
  needsUnlock: true,
  detections: {},
  visibleCameras: [],
  setVisibleCameras: (visibleCameras) => set({ visibleCameras }),
  setConnection: (connection) => set({ connection }),
  setCritical: (critical) => set({ critical }),
  receiveIncident: (incident, created) =>
    set((state) => ({
      critical:
        incident.severity === "critical" && incident.status === "new"
          ? [
              incident,
              ...state.critical.filter((row) => row.id !== incident.id),
            ]
          : state.critical.filter((row) => row.id !== incident.id),
      toasts: created
        ? [
            { id: incident.id, incident },
            ...state.toasts.filter((t) => t.id !== incident.id),
          ].slice(0, 4)
        : state.toasts.map((t) =>
            t.id === incident.id ? { ...t, incident } : t,
          ),
    })),
  removeToast: (id) =>
    set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
  tick: () => set({ clock: Date.now() }),
  setSound: (sound) => {
    localStorage.setItem("alarm-sound", String(sound));
    set({ sound });
  },
  setNeedsUnlock: (needsUnlock) => set({ needsUnlock }),
  reset: () =>
    set({ connection: "offline", critical: [], toasts: [], detections: {} }),
}));
