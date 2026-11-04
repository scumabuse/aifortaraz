import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Camera, Copy, Plus, Settings2, Trash2 } from "lucide-react";
import { api, readSession, type DictionaryItem } from "../../api";
import { Button } from "../../components/ui/button";
import ru from "../../locales/ru.json";
import type {
  CameraInput,
  CameraRecord,
  StreamType,
  ZoneRecord,
} from "./types";

const t = ru.cameraUi;
export function canEdit() {
  return ["admin", "engineer"].includes(readSession()?.user.role ?? "");
}
export function CameraPreview({ camera }: { camera: CameraRecord }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [camera.preview_url]);
  return (
    <div className="camera-preview">
      {camera.preview_url && !failed ? (
        <img
          src={camera.preview_url}
          onError={() => setFailed(true)}
          alt={camera.name}
        />
      ) : (
        <div>
          <Camera size={42} />
          <p>{t.noPreview}</p>
          <small>{t.previewHint}</small>
        </div>
      )}
    </div>
  );
}
function CameraForm({
  value,
  areas,
  onClose,
  onSaved,
}: {
  value: CameraRecord | null;
  areas: DictionaryItem[];
  onClose: () => void;
  onSaved: () => void;
}) {
  const [form, setForm] = useState<CameraInput>(
    value ?? {
      name: "",
      area_id: areas[0]?.id ?? "",
      stream_type: "rtsp",
      stream_url: "",
      preview_url: null,
      plan_x: null,
      plan_y: null,
      description: null,
      enabled: true,
      ai_enabled: true,
    },
  );
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  function set<K extends keyof CameraInput>(key: K, next: CameraInput[K]) {
    setForm((s) => ({ ...s, [key]: next }));
  }
  async function save(event: React.FormEvent) {
    event.preventDefault();
    setPending(true);
    setError("");
    try {
      const payload: CameraInput = {
        name: form.name,
        area_id: form.area_id,
        stream_type: form.stream_type,
        stream_url: form.stream_url,
        preview_url: form.preview_url || null,
        plan_x: form.plan_x,
        plan_y: form.plan_y,
        description: form.description || null,
        enabled: form.enabled,
        ai_enabled: form.ai_enabled,
      };
      await api(value ? `/api/v1/cameras/${value.id}` : "/api/v1/cameras", {
        method: value ? "PUT" : "POST",
        body: JSON.stringify(payload),
      });
      onSaved();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  return (
    <div className="modal-backdrop">
      <section
        className="modal panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="camera-form-title"
      >
        <h2 id="camera-form-title">{value ? t.edit : t.add}</h2>
        <form onSubmit={save} className="edit-form">
          <label>
            {t.name}
            <input
              autoFocus
              required
              maxLength={200}
              value={form.name}
              onChange={(e) => set("name", e.target.value)}
            />
          </label>
          <div className="form-grid">
            <label>
              {t.area}
              <select
                required
                value={form.area_id}
                onChange={(e) => set("area_id", e.target.value)}
              >
                {areas.map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              {t.streamType}
              <select
                value={form.stream_type}
                onChange={(e) =>
                  set("stream_type", e.target.value as StreamType)
                }
              >
                {(["rtsp", "mjpeg", "hls", "file", "webcam"] as const).map(
                  (s) => (
                    <option key={s} value={s}>
                      {t.streamTypes[s]}
                    </option>
                  ),
                )}
              </select>
            </label>
          </div>
          <label>
            {t.url}
            <input
              required={form.stream_type !== "webcam"}
              maxLength={2048}
              value={form.stream_url}
              placeholder={
                form.stream_type === "rtsp"
                  ? "rtsp://192.168.1.10/live"
                  : "/demo/cam1.mp4"
              }
              onChange={(e) => set("stream_url", e.target.value)}
            />
          </label>
          <label>
            {t.previewUrl}
            <input
              maxLength={2048}
              value={form.preview_url ?? ""}
              onChange={(e) => set("preview_url", e.target.value || null)}
            />
          </label>
          <div className="form-grid">
            {(["plan_x", "plan_y"] as const).map((key) => (
              <label key={key}>
                {key === "plan_x" ? t.planX : t.planY}
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.001"
                  value={form[key] ?? ""}
                  onChange={(e) =>
                    set(
                      key,
                      e.target.value === "" ? null : Number(e.target.value),
                    )
                  }
                />
              </label>
            ))}
          </div>
          <label>
            {t.description}
            <textarea
              maxLength={4000}
              value={form.description ?? ""}
              onChange={(e) => set("description", e.target.value)}
            />
          </label>
          <div className="form-grid">
            <label className="check-label">
              <input
                type="checkbox"
                checked={form.enabled}
                onChange={(e) => set("enabled", e.target.checked)}
              />
              {t.enabled}
            </label>
            <label className="check-label">
              <input
                type="checkbox"
                checked={form.ai_enabled}
                onChange={(e) => set("ai_enabled", e.target.checked)}
              />
              {t.aiEnabled}
            </label>
          </div>
          {error && (
            <p role="alert" className="error">
              {error}
            </p>
          )}
          <div className="action-row">
            <Button disabled={pending}>{pending ? ru.loading : t.save}</Button>
            <Button
              type="button"
              variant="outline"
              disabled={pending}
              onClick={onClose}
            >
              {t.cancel}
            </Button>
          </div>
        </form>
      </section>
    </div>
  );
}
export function CamerasPage() {
  const [cameras, setCameras] = useState<CameraRecord[]>([]);
  const [areas, setAreas] = useState<DictionaryItem[]>([]);
  const [state, setState] = useState("loading");
  const [error, setError] = useState("");
  const [form, setForm] = useState<CameraRecord | null | undefined>();
  const [notice, setNotice] = useState("");
  const [working, setWorking] = useState(false);
  const load = useCallback(() => {
    setState("loading");
    Promise.all([
      api<CameraRecord[]>("/api/v1/cameras"),
      api<DictionaryItem[]>("/api/v1/settings/areas"),
    ])
      .then(([c, a]) => {
        setCameras(c);
        setAreas(a);
        setState("ready");
      })
      .catch((e) => {
        setError(e.message);
        setState("error");
      });
  }, []);
  useEffect(load, [load]);
  useEffect(() => {
    const status = (event: Event) => {
      const next = (event as CustomEvent).detail as Partial<CameraRecord> & {
        camera_id: string;
      };
      setCameras((rows) =>
        rows.map((row) =>
          row.id === next.camera_id ? { ...row, ...next } : row,
        ),
      );
    };
    window.addEventListener("camera-status", status);
    window.addEventListener("realtime-reconnected", load);
    return () => {
      window.removeEventListener("camera-status", status);
      window.removeEventListener("realtime-reconnected", load);
    };
  }, [load]);
  async function remove(camera: CameraRecord) {
    if (!confirm(`${t.deleteConfirm} «${camera.name}»?`)) return;
    setWorking(true);
    setError("");
    try {
      await api(`/api/v1/cameras/${camera.id}`, { method: "DELETE" });
      load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setWorking(false);
    }
  }
  async function copy(id: string) {
    try {
      await navigator.clipboard.writeText(id);
      setNotice(t.copied);
    } catch {
      setNotice(id);
    }
  }
  async function check(id: string) {
    setWorking(true);
    setError("");
    try {
      const result = await api<{ ok: boolean; message: string }>(
        `/api/v1/cameras/${id}/check-stream`,
        { method: "POST" },
      );
      setNotice(result.message);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setWorking(false);
    }
  }
  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">{ru.enterprise}</span>
          <h1>{ru.cameras}</h1>
          <p>{t.subtitle}</p>
        </div>
        {canEdit() && (
          <Button onClick={() => setForm(null)} disabled={!areas.length}>
            <Plus size={17} />
            {t.add}
          </Button>
        )}
      </div>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {notice && (
        <div role="status" className="notice">
          {notice}
          <button onClick={() => setNotice("")} aria-label={t.dismiss}>
            ×
          </button>
        </div>
      )}
      {state === "loading" ? (
        <div className="panel skeleton" role="status">
          {ru.loading}
        </div>
      ) : state === "error" ? (
        <Button onClick={load} variant="outline">
          {ru.retry}
        </Button>
      ) : !cameras.length ? (
        <section className="empty panel">
          <Camera size={38} />
          <h2>{t.empty}</h2>
          <p>{t.emptyHint}</p>
        </section>
      ) : (
        <section className="panel table-scroll">
          <table className="camera-table">
            <thead>
              <tr>
                {[
                  t.name,
                  t.area,
                  t.streamType,
                  t.url,
                  t.status,
                  t.fps,
                  t.model,
                  t.zones,
                  t.incidents,
                  t.enabled,
                  t.actions,
                ].map((label) => (
                  <th key={label}>{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cameras.map((c) => (
                <tr key={c.id}>
                  <td>
                    <Link className="accent" to={`/cameras/${c.id}`}>
                      {c.name}
                    </Link>
                  </td>
                  <td>{c.area_name}</td>
                  <td>{t.streamTypes[c.stream_type]}</td>
                  <td>
                    <span className="truncate" title={c.stream_url}>
                      {c.stream_url || "—"}
                    </span>
                  </td>
                  <td>
                    <span
                      className={`status-dot ${c.status === "online" ? "" : "gray"}`}
                    />
                    {t.statuses[c.status as keyof typeof t.statuses] ??
                      c.status}
                  </td>
                  <td>{c.fps ?? "—"}</td>
                  <td>{c.model_name ?? "—"}</td>
                  <td>
                    <Link to={`/cameras/${c.id}/zones`}>{c.zones_count}</Link>
                  </td>
                  <td>{c.incidents_7d}</td>
                  <td>{c.enabled ? t.yes : t.no}</td>
                  <td>
                    <div className="action-row">
                      <button
                        className="icon-button"
                        title={t.copyId}
                        aria-label={t.copyId}
                        onClick={() => void copy(c.id)}
                      >
                        <Copy size={16} />
                      </button>
                      {canEdit() && (
                        <>
                          <button
                            className="icon-button"
                            title={t.edit}
                            aria-label={t.edit}
                            onClick={() => setForm(c)}
                          >
                            <Settings2 size={16} />
                          </button>
                          <button
                            className="icon-button"
                            disabled={working}
                            title={t.remove}
                            aria-label={t.remove}
                            onClick={() => void remove(c)}
                          >
                            <Trash2 size={16} />
                          </button>
                          <Button
                            variant="outline"
                            disabled={working}
                            onClick={() => void check(c.id)}
                          >
                            {t.check}
                          </Button>
                        </>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
      {form !== undefined && (
        <CameraForm
          value={form}
          areas={areas}
          onClose={() => setForm(undefined)}
          onSaved={() => {
            setForm(undefined);
            load();
          }}
        />
      )}
    </>
  );
}
export function CameraPage() {
  const { id } = useParams();
  const [camera, setCamera] = useState<CameraRecord | null>(null);
  const [zones, setZones] = useState<ZoneRecord[]>([]);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let active = true;
    setCamera(null);
    setError("");
    Promise.all([
      api<CameraRecord>(`/api/v1/cameras/${id}`),
      api<ZoneRecord[]>(`/api/v1/cameras/${id}/zones`),
    ])
      .then(([c, z]) => {
        if (active) {
          setCamera(c);
          setZones(z);
        }
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, [id, attempt]);
  useEffect(() => {
    const status = (event: Event) => {
      const next = (event as CustomEvent).detail as Partial<CameraRecord> & {
        camera_id: string;
      };
      if (next.camera_id === id)
        setCamera((current) => (current ? { ...current, ...next } : current));
    };
    const reconnect = () => setAttempt((value) => value + 1);
    window.addEventListener("camera-status", status);
    window.addEventListener("realtime-reconnected", reconnect);
    return () => {
      window.removeEventListener("camera-status", status);
      window.removeEventListener("realtime-reconnected", reconnect);
    };
  }, [id]);
  if (error)
    return (
      <div className="panel">
        <p className="error" role="alert">
          {error}
        </p>
        <Button onClick={() => setAttempt((x) => x + 1)}>{ru.retry}</Button>
      </div>
    );
  if (!camera) return <div className="skeleton">{ru.loading}</div>;
  return (
    <>
      <div className="page-heading">
        <div>
          <Link className="accent" to="/cameras">
            ← {ru.cameras}
          </Link>
          <h1>{camera.name}</h1>
          <p>
            {camera.area_name} ·{" "}
            {t.statuses[camera.status as keyof typeof t.statuses]}
          </p>
        </div>
        <Button asChild>
          <Link to={`/cameras/${id}/zones`}>{ru.zones}</Link>
        </Button>
      </div>
      <div className="camera-detail-grid">
        <section className="panel">
          <CameraPreview camera={camera} />
          <p className="muted camera-description">{camera.description}</p>
          <dl className="metadata">
            {[
              [t.streamType, t.streamTypes[camera.stream_type]],
              [t.url, camera.stream_url],
              [t.fps, camera.fps ?? "—"],
              [t.model, camera.model_name ?? "—"],
              [
                t.lastHeartbeat,
                camera.last_heartbeat_at
                  ? new Date(camera.last_heartbeat_at).toLocaleString("ru-RU")
                  : "—",
              ],
            ].map(([k, v]) => (
              <div key={k}>
                <dt>{k}</dt>
                <dd>{v}</dd>
              </div>
            ))}
          </dl>
        </section>
        <section className="panel">
          <h2>{t.zones}</h2>
          <div className="dictionary-list">
            {zones.length ? (
              zones.map((z) => (
                <Link
                  to={`/cameras/${id}/zones?zone=${z.id}`}
                  key={z.id}
                  className="zone-link"
                >
                  <span style={{ background: z.color }} className="zone-dot" />
                  {z.name}
                  <span className="muted">{ru.zoneUi.types[z.zone_type]}</span>
                </Link>
              ))
            ) : (
              <p className="muted">{ru.zoneUi.empty}</p>
            )}
          </div>
          <div className="deferred-note">
            <h3>{t.history}</h3>
            <p>{t.historyHint}</p>
          </div>
        </section>
      </div>
    </>
  );
}
