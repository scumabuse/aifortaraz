import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Bell,
  CheckCheck,
  Volume2,
  VolumeX,
  X,
  AlertTriangle,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { api, readSession, type Session } from "../../api";
import { queryClient } from "../../lib/query";
import { Button } from "../../components/ui/button";
import type { Inbox, IncidentPage, IncidentRecord } from "../incidents/types";
import {
  canAct,
  formatDate,
  relativeTime,
  setTimezone,
} from "../incidents/utils";
import { useLiveStore } from "./store";
import ru from "../../locales/ru.json";
import "./realtime.css";
import type { DetectionFrame } from "../live/types";

const t = ru.realtimeUi;
let alarm: HTMLAudioElement | null = null;
function sound() {
  alarm ??= new Audio("/sounds/alarm.wav");
  alarm.currentTime = 0;
  return alarm.play();
}
export function invalidateIncidents() {
  for (const key of ["incidents", "incident", "summary", "plan", "inbox"])
    void queryClient.invalidateQueries({ queryKey: [key] });
}
export function RealtimeSession({
  session,
  children,
}: {
  session: Session;
  children: React.ReactNode;
}) {
  const [epoch, setEpoch] = useState(0);
  const settings = useQuery({
    queryKey: ["enterprise-settings", session.user.id],
    queryFn: () =>
      api<{ timezone: string; theme: string; alarm_sound: boolean }>(
        "/api/v1/settings",
      ),
  });
  useEffect(() => {
    if (!settings.data) return;
    setTimezone(settings.data.timezone);
    document.documentElement.dataset.theme = settings.data.theme;
    if (localStorage.getItem("alarm-sound") === null)
      useLiveStore.setState({ sound: settings.data.alarm_sound });
    return () => {
      delete document.documentElement.dataset.theme;
    };
  }, [settings.data]);
  const critical = useQuery({
    queryKey: ["incidents", "critical", session.user.id],
    queryFn: ({ signal }) =>
      api<IncidentPage>(
        "/api/v1/incidents?severity=critical&status=new&page_size=100",
        { signal },
      ),
    refetchInterval: 30_000,
  });
  useEffect(() => {
    if (critical.data) useLiveStore.getState().setCritical(critical.data.items);
  }, [critical.data, epoch]);
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let socket: WebSocket | null = null;
    let attempt = 0;
    let visibleCameras: string[] = [];
    const store = useLiveStore.getState;
    const connect = () => {
      if (stopped) return;
      const tokens = readSession();
      if (!tokens) return;
      store().setConnection(attempt ? "reconnecting" : "connecting");
      socket = new WebSocket(
        `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live?token=${encodeURIComponent(tokens.access_token)}`,
      );
      socket.onopen = () => {
        if (stopped) {
          socket?.close();
          return;
        }
        attempt = 0;
        store().setConnection("connected");
        visibleCameras = [...store().visibleCameras];
        if (visibleCameras.length)
          socket?.send(
            JSON.stringify({ type: "subscribe", cameras: visibleCameras }),
          );
        invalidateIncidents();
        window.dispatchEvent(new Event("realtime-reconnected"));
      };
      socket.onmessage = (event) => {
        let message: {
          type: string;
          incident?: IncidentRecord;
          camera_id?: string;
          [key: string]: unknown;
        };
        try {
          message = JSON.parse(event.data);
        } catch {
          return;
        }
        if (message.type === "ping") {
          socket?.send(JSON.stringify({ type: "pong" }));
          return;
        }
        if (
          message.type === "incident.created" ||
          message.type === "incident.updated"
        ) {
          if (!message.incident) return;
          store().receiveIncident(
            message.incident,
            message.type === "incident.created",
          );
          invalidateIncidents();
          if (message.type === "incident.created") {
            const id = message.incident.id;
            setTimeout(() => store().removeToast(id), 10_000);
            if (
              store().sound &&
              ["high", "critical"].includes(message.incident.severity)
            )
              void sound().catch(() => store().setNeedsUnlock(true));
          }
        } else if (message.type === "notification")
          void queryClient.invalidateQueries({ queryKey: ["inbox"] });
        else if (message.type === "camera.status") {
          window.dispatchEvent(
            new CustomEvent("camera-status", { detail: message }),
          );
          void queryClient.invalidateQueries({ queryKey: ["summary"] });
          if (message.source !== "mock")
            void queryClient.invalidateQueries({ queryKey: ["demo"] });
        } else if (message.type === "config.updated") {
          window.dispatchEvent(
            new CustomEvent("config-updated", { detail: message }),
          );
          void queryClient.invalidateQueries({ queryKey: ["plan"] });
          void queryClient.invalidateQueries({ queryKey: ["summary"] });
        } else if (message.type === "detections" && message.camera_id) {
          const previous = store().detections[message.camera_id]?.source;
          const frame = {
            ...message,
            receivedAt: performance.now(),
          } as unknown as DetectionFrame;
          useLiveStore.setState((state) => ({
            detections: { ...state.detections, [message.camera_id!]: frame },
          }));
          if (frame.source === "ai" && previous !== "ai")
            void queryClient.invalidateQueries({ queryKey: ["demo"] });
        } else if (message.type === "settings.updated") {
          for (const key of [
            "enterprise-settings",
            "public-project",
            "settings-catalog",
            "analytics-areas",
            "incident-types",
            "ppe-types",
            "cameras-options",
            "summary",
            "plan",
          ])
            void queryClient.invalidateQueries({ queryKey: [key] });
        } else if (message.type === "demo.updated") {
          invalidateIncidents();
          for (const key of ["demo", "live-cameras", "cameras-options"])
            void queryClient.invalidateQueries({ queryKey: [key] });
        }
      };
      socket.onerror = () => socket?.close();
      socket.onclose = (event) => {
        if (stopped) return;
        store().setConnection(navigator.onLine ? "reconnecting" : "offline");
        const delay = Math.min(30_000, 1000 * 2 ** Math.min(attempt++, 5));
        if (event.code === 4401) {
          void api("/api/v1/auth/me")
            .then(() => {
              if (!stopped) timer = setTimeout(connect, delay);
            })
            .catch(() => {
              if (!stopped && readSession()) timer = setTimeout(connect, delay);
            });
        } else timer = setTimeout(connect, delay);
      };
    };
    connect();
    const clock = setInterval(() => store().tick(), 30_000);
    const online = () => {
      socket?.close();
    };
    const synchronize = (cameras: string[]) => {
      if (socket?.readyState === WebSocket.OPEN) {
        const removed = visibleCameras.filter((id) => !cameras.includes(id));
        const added = cameras.filter((id) => !visibleCameras.includes(id));
        if (removed.length)
          socket.send(
            JSON.stringify({ type: "unsubscribe", cameras: removed }),
          );
        if (added.length)
          socket.send(JSON.stringify({ type: "subscribe", cameras: added }));
      }
      visibleCameras = [...cameras];
    };
    const unsubscribe = useLiveStore.subscribe((state, previous) => {
      if (state.visibleCameras !== previous.visibleCameras)
        synchronize(state.visibleCameras);
    });
    const storage = (event: StorageEvent) => {
      if (event.key === "inbox-read")
        void queryClient.invalidateQueries({ queryKey: ["inbox"] });
    };
    window.addEventListener("online", online);
    window.addEventListener("storage", storage);
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
      socket?.close();
      clearInterval(clock);
      window.removeEventListener("online", online);
      window.removeEventListener("storage", storage);
      unsubscribe();
      store().reset();
    };
  }, [session.user.id, epoch]);
  return (
    <>
      {children}
      <GlobalAlerts retry={() => setEpoch((x) => x + 1)} />
    </>
  );
}
export function ConnectionStatus() {
  const connection = useLiveStore((s) => s.connection);
  return (
    <span className={`connection connection-${connection}`} role="status">
      <span
        className={`status-dot ${connection === "connected" ? "" : "gray"}`}
      />
      {t.connections[connection]}
    </span>
  );
}
export function SoundButton() {
  const enabled = useLiveStore((s) => s.sound);
  const locked = useLiveStore((s) => s.needsUnlock);
  const [error, setError] = useState(false);
  async function toggle() {
    setError(false);
    if (enabled && !locked) {
      useLiveStore.getState().setSound(false);
      return;
    }
    try {
      await sound();
      useLiveStore.getState().setSound(true);
      useLiveStore.getState().setNeedsUnlock(false);
    } catch {
      setError(true);
    }
  }
  return (
    <button
      className="icon-button"
      onClick={() => void toggle()}
      title={
        error ? t.soundBlocked : enabled && !locked ? t.mute : t.enableSound
      }
      aria-label={enabled && !locked ? t.mute : t.enableSound}
    >
      {enabled && !locked ? <Volume2 size={19} /> : <VolumeX size={19} />}
    </button>
  );
}
export function NotificationCenter() {
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState(1);
  const [error, setError] = useState("");
  const userId = readSession()?.user.id;
  useLiveStore((s) => s.clock);
  const inbox = useQuery({
    queryKey: ["inbox", userId, page],
    queryFn: () => api<Inbox>(`/api/v1/notifications/inbox?page=${page}`),
    refetchInterval: 30_000,
  });
  async function readAll() {
    try {
      await api("/api/v1/notifications/read-all", { method: "POST" });
      localStorage.setItem("inbox-read", String(Date.now()));
      await queryClient.invalidateQueries({ queryKey: ["inbox"] });
    } catch (e) {
      setError((e as Error).message);
    }
  }
  return (
    <div className="notification-container">
      <button
        className="icon-button notification-bell"
        title={t.notifications}
        aria-label={t.notifications}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <Bell size={19} />
        {!!inbox.data?.unread_count && (
          <span>
            {inbox.data.unread_count > 99 ? "99+" : inbox.data.unread_count}
          </span>
        )}
      </button>
      {open && (
        <section
          className="notification-popover panel"
          role="dialog"
          aria-label={t.notifications}
        >
          <div className="panel-heading">
            <h2>{t.notifications}</h2>
            <button
              className="icon-button"
              aria-label={t.close}
              onClick={() => setOpen(false)}
            >
              <X size={17} />
            </button>
          </div>
          <Button
            variant="outline"
            disabled={!inbox.data?.unread_count}
            onClick={() => void readAll()}
          >
            <CheckCheck size={16} />
            {t.readAll}
          </Button>
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          {inbox.isPending ? (
            <p>{ru.loading}</p>
          ) : inbox.isError ? (
            <div>
              <p className="error">{ru.error}</p>
              <Button onClick={() => void inbox.refetch()}>{ru.retry}</Button>
            </div>
          ) : !inbox.data.items.length ? (
            <p className="notification-empty">{t.empty}</p>
          ) : (
            <div className="notification-list">
              {inbox.data.items.map((item) => (
                <Link
                  to={`/incidents/${item.incident_id}`}
                  key={item.id}
                  className={item.read_at ? "" : "unread"}
                  onClick={() => setOpen(false)}
                >
                  <span
                    className={`notification-severity severity-${item.severity}`}
                  />
                  <div>
                    <strong>
                      {ru.incidentUi.types[
                        item.type_code as keyof typeof ru.incidentUi.types
                      ] ?? item.text}
                    </strong>
                    <p>{item.camera_name}</p>
                    <small>{relativeTime(item.created_at)}</small>
                  </div>
                </Link>
              ))}
            </div>
          )}
          {inbox.data && inbox.data.total > 30 && (
            <div className="action-row">
              <Button
                variant="ghost"
                disabled={page <= 1}
                onClick={() => setPage((p) => p - 1)}
              >
                ←
              </Button>
              <span>{page}</span>
              <Button
                variant="ghost"
                disabled={page * 30 >= inbox.data.total}
                onClick={() => setPage((p) => p + 1)}
              >
                →
              </Button>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
function GlobalAlerts({ retry }: { retry: () => void }) {
  const critical = useLiveStore((s) => s.critical);
  const toasts = useLiveStore((s) => s.toasts);
  const connection = useLiveStore((s) => s.connection);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const incident = critical[0];
  async function acknowledge() {
    if (!incident) return;
    setPending(true);
    setError("");
    try {
      const row = await api<IncidentRecord>(
        `/api/v1/incidents/${incident.id}/acknowledge`,
        { method: "POST" },
      );
      useLiveStore.getState().receiveIncident(row, false);
      invalidateIncidents();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  return (
    <>
      {incident && (
        <aside className="critical-banner" role="alert">
          <AlertTriangle size={25} />
          <div>
            <strong>
              {t.critical} · {incident.type_name}
            </strong>
            <p>
              {incident.camera_name}
              {incident.zone_name ? ` → ${incident.zone_name}` : ""} ·{" "}
              {formatDate(incident.started_at)}
              {critical.length > 1
                ? ` · ${t.more}: ${critical.length - 1}`
                : ""}
            </p>
            {error && <p>{error}</p>}
          </div>
          <Button asChild variant="outline">
            <Link to={`/incidents/${incident.id}`}>{t.open}</Link>
          </Button>
          {canAct(incident, "acknowledge") && (
            <Button disabled={pending} onClick={() => void acknowledge()}>
              {ru.incidentUi.acknowledge}
            </Button>
          )}
        </aside>
      )}
      <div className="toast-stack" aria-live="polite">
        {toasts.map((item) => (
          <article
            key={item.id}
            className={`incident-toast toast-${item.incident.severity}`}
          >
            <div>
              <strong>{item.incident.type_name}</strong>
              <button
                className="icon-button"
                aria-label={t.close}
                onClick={() => useLiveStore.getState().removeToast(item.id)}
              >
                <X size={16} />
              </button>
            </div>
            <p>
              {item.incident.camera_name} ·{" "}
              {ru.incidentUi.severities[item.incident.severity]}
            </p>
            <Link to={`/incidents/${item.id}`}>{t.open} →</Link>
          </article>
        ))}
      </div>
      {connection === "offline" && (
        <button className="reconnect-button" onClick={retry}>
          {t.retry}
        </button>
      )}
    </>
  );
}
