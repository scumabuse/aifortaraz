import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Camera, Radio, Settings2, Zap } from "lucide-react";
import { api, type DictionaryItem } from "../../api";
import { queryClient } from "../../lib/query";
import { Button } from "../../components/ui/button";
import { canEdit } from "../cameras/Cameras";
import type { CameraRecord, ZoneRecord } from "../cameras/types";
import { IncidentActions } from "../incidents/IncidentActions";
import { SeverityBadge, SourceBadge } from "../incidents/IncidentsPage";
import { relativeTime } from "../incidents/utils";
import type { IncidentPage, IncidentRecord } from "../incidents/types";
import { useLiveStore } from "../realtime/store";
import { CameraPlayer } from "./CameraPlayer";
import { DemoSettings, useDemo } from "./DemoSettings";
import { objectLabel, ppeMark, ppeState } from "./labels";
import type { DetectionFrame, Overlays } from "./types";
import ru from "../../locales/ru.json";
import "./live.css";

const t = ru.liveUi;
const defaults: Overlays = {
  boxes: true,
  zones: true,
  trackIds: true,
  labels: true,
  violationsOnly: false,
};
export function CameraTile({
  camera,
  overlays,
  demoActive,
}: {
  camera: CameraRecord;
  overlays: Overlays;
  demoActive: boolean;
}) {
  const zones = useQuery({
    queryKey: ["live-zones", camera.id],
    queryFn: () => api<ZoneRecord[]>(`/api/v1/cameras/${camera.id}/zones`),
  });
  const source = useLiveStore((state) => state.detections[camera.id]?.source);
  return (
    <>
      <CameraPlayer
        camera={camera}
        overlays={overlays}
        zones={zones.data ?? []}
        demoActive={demoActive && source !== "ai"}
      />
      {zones.isError && (
        <div className="zone-load-error">
          {t.zoneError}
          <Button variant="ghost" onClick={() => void zones.refetch()}>
            {ru.retry}
          </Button>
        </div>
      )}
    </>
  );
}
function ObjectsPanel({ cameraId }: { cameraId: string }) {
  const [frame, setFrame] = useState<DetectionFrame | undefined>();
  const zones = useQuery({
    queryKey: ["live-zones", cameraId],
    queryFn: () => api<ZoneRecord[]>(`/api/v1/cameras/${cameraId}/zones`),
  });
  useEffect(() => {
    const update = () => {
      const next = useLiveStore.getState().detections[cameraId];
      setFrame(
        next && performance.now() - next.receivedAt < 3000 ? next : undefined,
      );
    };
    update();
    const timer = setInterval(update, 300);
    return () => clearInterval(timer);
  }, [cameraId]);
  return (
    <section className="panel objects-panel">
      <div className="panel-heading">
        <h2>{t.objects}</h2>
        <span className="tag">{frame?.objects.length ?? 0}</span>
      </div>
      {!frame?.objects.length ? (
        <p className="muted">{t.noObjects}</p>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>{t.track}</th>
                <th>{t.objectClass}</th>
                {["helmet", "vest", "glasses", "gloves"].map((code) => (
                  <th key={code}>
                    {
                      ru.incidentUi.ppeNames[
                        code as keyof typeof ru.incidentUi.ppeNames
                      ]
                    }
                  </th>
                ))}
                <th>{t.zone}</th>
                <th>{t.state}</th>
              </tr>
            </thead>
            <tbody>
              {frame.objects.map((object, index) => (
                <tr key={`${object.track_id}-${index}`}>
                  <td>#{object.track_id ?? "—"}</td>
                  <td>{objectLabel(object.label, t.objectTypes)}</td>
                  {["helmet", "vest", "glasses", "gloves"].map((code) => {
                    const value = object.ppe?.[code];
                    return (
                      <td key={code}>
                        {value === undefined ? (
                          "—"
                        ) : (
                          <span title={t[ppeState(value)]}>
                            {ppeMark(value)}
                          </span>
                        )}
                      </td>
                    );
                  })}
                  <td>
                    {object.zones
                      ?.map(
                        (id) =>
                          zones.data?.find((z) => z.id === id)?.name ?? t.zone,
                      )
                      .join(", ") || "—"}
                  </td>
                  <td>
                    <span
                      className={`object-state object-${object.state ?? "ok"}`}
                    >
                      {t.states[object.state ?? "ok"]}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
export function useLiveCameras() {
  const cameras = useQuery({
    queryKey: ["live-cameras"],
    queryFn: () => api<CameraRecord[]>("/api/v1/cameras"),
    refetchInterval: 10000,
  });
  useEffect(() => {
    const status = (event: Event) => {
      const value = (event as CustomEvent).detail as Partial<CameraRecord> & {
        camera_id: string;
      };
      queryClient.setQueryData<CameraRecord[]>(["live-cameras"], (rows) =>
        rows?.map((c) => (c.id === value.camera_id ? { ...c, ...value } : c)),
      );
    };
    const config = () => {
      void queryClient.invalidateQueries({ queryKey: ["live-zones"] });
      void queryClient.invalidateQueries({ queryKey: ["live-cameras"] });
    };
    const reconnect = () => {
      void queryClient.invalidateQueries({ queryKey: ["live-cameras"] });
      config();
    };
    window.addEventListener("camera-status", status);
    window.addEventListener("config-updated", config);
    window.addEventListener("realtime-reconnected", reconnect);
    return () => {
      window.removeEventListener("camera-status", status);
      window.removeEventListener("config-updated", config);
      window.removeEventListener("realtime-reconnected", reconnect);
    };
  }, []);
  return cameras;
}

export function LivePage() {
  const { cameraId } = useParams();
  const [grid, setGrid] = useState(() => {
    const value = Number(localStorage.getItem("live-grid") ?? 4);
    return [1, 4, 9].includes(value) ? value : 4;
  });
  const [chosen, setChosen] = useState<string[]>([]);
  const [overlays, setOverlays] = useState<Overlays>(() => {
    try {
      return {
        ...defaults,
        ...JSON.parse(localStorage.getItem("live-overlays") ?? "{}"),
      };
    } catch {
      return defaults;
    }
  });
  const [settings, setSettings] = useState(false);
  const [simulation, setSimulation] = useState(false);
  const [simulationCamera, setSimulationCamera] = useState("");
  const [simulationType, setSimulationType] = useState("ppe_violation");
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const cameras = useLiveCameras();
  const demo = useDemo();
  const types = useQuery({
    queryKey: ["incident-types"],
    queryFn: () => api<DictionaryItem[]>("/api/v1/settings/incident-types"),
  });
  const available = useMemo(
    () => cameras.data?.filter((c) => c.enabled) ?? [],
    [cameras.data],
  );
  const slots = cameraId ? 1 : grid;
  const visible = useMemo(() => {
    if (cameraId) return [cameraId];
    const result = chosen
      .slice(0, slots)
      .map((id) => (available.some((c) => c.id === id) ? id : ""));
    while (result.length < slots) result.push("");
    if (!chosen.length) {
      available.slice(0, slots).forEach((c, i) => {
        result[i] = c.id;
      });
    }
    return result;
  }, [cameraId, slots, chosen, available]);
  const ids = visible.filter(Boolean).join(",");
  useEffect(() => {
    const selectedIds = ids.split(",").filter(Boolean);
    useLiveStore.getState().setVisibleCameras([...new Set(selectedIds)]);
    return () => useLiveStore.getState().setVisibleCameras([]);
  }, [ids]);
  const journal = useQuery({
    queryKey: ["incidents", "live", ids],
    queryFn: () =>
      api<IncidentPage>(
        `/api/v1/incidents?camera_ids=${encodeURIComponent(ids)}&page_size=20&sort=-created_at`,
      ),
    enabled: !!ids,
  });
  useLiveStore((s) => s.clock);
  function choose(index: number, value: string) {
    setChosen(() => {
      const next = [...visible];
      const existing = next.indexOf(value);
      if (value && existing >= 0 && existing !== index)
        next[existing] = next[index];
      next[index] = value;
      return next;
    });
  }
  function toggle(key: keyof Overlays) {
    setOverlays((current) => {
      const next = { ...current, [key]: !current[key] };
      localStorage.setItem("live-overlays", JSON.stringify(next));
      return next;
    });
  }
  async function simulate(event: React.FormEvent) {
    event.preventDefault();
    setPending(true);
    setError("");
    try {
      const incident = await api<IncidentRecord>("/api/v1/demo/simulate", {
        method: "POST",
        body: JSON.stringify({
          camera_id: simulationCamera,
          type: simulationType,
        }),
      });
      void incident;
      setSimulation(false);
      void queryClient.invalidateQueries({ queryKey: ["incidents"] });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  function openSimulation() {
    setError("");
    setSimulationCamera(
      visible.find((id) => demo.data?.active_camera_ids.includes(id)) ??
        demo.data?.active_camera_ids[0] ??
        "",
    );
    setSimulation(true);
  }
  if (cameras.isPending)
    return (
      <section className="panel skeleton" role="status">
        {ru.loading}
      </section>
    );
  if (cameras.isError)
    return (
      <section className="panel">
        <p className="error" role="alert">
          {cameras.error.message}
        </p>
        <Button onClick={() => void cameras.refetch()}>{ru.retry}</Button>
      </section>
    );
  const focused = cameraId
    ? cameras.data.find((c) => c.id === cameraId)
    : undefined;
  return (
    <>
      <div className="page-heading">
        <div>
          {cameraId && (
            <Link className="accent" to="/live">
              ← {ru.live}
            </Link>
          )}
          <span className="eyebrow">{ru.enterprise}</span>
          <h1>{focused?.name ?? ru.live}</h1>
          <p>{t.subtitle}</p>
        </div>
        <div className="action-row">
          {demo.data?.enabled && canEdit() && (
            <Button
              onClick={openSimulation}
              disabled={
                !demo.data.active_camera_ids.length || demo.data.seeding
              }
            >
              <Zap size={17} />
              {t.simulate}
            </Button>
          )}
          {canEdit() && (
            <Button variant="outline" onClick={() => setSettings((v) => !v)}>
              <Settings2 size={17} />
              {t.demoSettings}
            </Button>
          )}
        </div>
      </div>
      {demo.data?.seeding && (
        <p className="notice" role="status">
          {t.seeding}
        </p>
      )}
      {demo.isError && (
        <p className="error">
          {t.demoUnavailable}
          <Button variant="outline" onClick={() => void demo.refetch()}>
            {ru.retry}
          </Button>
        </p>
      )}
      {demo.data?.error && (
        <p className="error" role="alert">
          {demo.data.error}
        </p>
      )}
      {settings && (
        <section className="panel live-demo-settings">
          <DemoSettings />
        </section>
      )}
      <section className="panel live-controls">
        {!cameraId && (
          <div className="layout-switcher">
            {[1, 4, 9].map((size) => (
              <Button
                key={size}
                variant={size === grid ? "default" : "outline"}
                onClick={() => {
                  setGrid(size);
                  localStorage.setItem("live-grid", String(size));
                }}
              >
                {Math.sqrt(size)} × {Math.sqrt(size)}
              </Button>
            ))}
          </div>
        )}
        <div className="overlay-switches">
          {Object.keys(defaults).map((key) => (
            <label key={key}>
              <input
                type="checkbox"
                checked={overlays[key as keyof Overlays]}
                onChange={() => toggle(key as keyof Overlays)}
              />
              {t.overlays[key as keyof Overlays]}
            </label>
          ))}
        </div>
      </section>
      {cameraId && !focused ? (
        <section className="panel empty">
          <h2>{t.cameraNotFound}</h2>
          <Button asChild variant="outline">
            <Link to="/live">{ru.live}</Link>
          </Button>
        </section>
      ) : !available.length && !focused ? (
        <section className="panel empty">
          <Camera size={40} />
          <h2>{t.noCameras}</h2>
          <p>{t.addCameraHint}</p>
          <Button asChild variant="outline">
            <Link to="/cameras">{ru.cameras}</Link>
          </Button>
        </section>
      ) : (
        <div className={`live-layout ${cameraId ? "live-focused" : ""}`}>
          <div>
            <div className={`camera-grid grid-${slots}`}>
              {visible.map((id, index) => {
                const camera = cameras.data.find((c) => c.id === id);
                return (
                  <div className="camera-slot" key={`${index}-${id}`}>
                    {!cameraId && (
                      <label className="camera-picker">
                        <span>
                          {t.cameraSlot} {index + 1}
                        </span>
                        <select
                          aria-label={`${t.cameraSlot} ${index + 1}`}
                          value={id}
                          onChange={(e) => choose(index, e.target.value)}
                        >
                          <option value="">{t.selectCamera}</option>
                          {available.map((c) => (
                            <option value={c.id} key={c.id}>
                              {c.name}
                            </option>
                          ))}
                        </select>
                      </label>
                    )}
                    {camera ? (
                      <CameraTile
                        key={camera.id}
                        camera={camera}
                        overlays={overlays}
                        demoActive={
                          !!demo.data?.active_camera_ids.includes(camera.id)
                        }
                      />
                    ) : (
                      <div className="empty-camera">
                        <Camera size={35} />
                        <p>{t.selectCamera}</p>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            {cameraId && focused && <ObjectsPanel cameraId={cameraId} />}
          </div>
          <aside className="panel live-journal">
            <div className="panel-heading">
              <h2>
                <Radio size={18} />
                {t.journal}
              </h2>
              <span className="tag">{journal.data?.total ?? 0}</span>
            </div>
            <p className="journal-hint">{t.journalHint}</p>
            {!ids ? (
              <p className="muted">{t.selectCamera}</p>
            ) : journal.isPending ? (
              <div className="skeleton">{ru.loading}</div>
            ) : journal.isError ? (
              <div>
                <p className="error">{ru.error}</p>
                <Button onClick={() => void journal.refetch()}>
                  {ru.retry}
                </Button>
              </div>
            ) : !journal.data?.items.length ? (
              <p className="muted">{t.emptyJournal}</p>
            ) : (
              <div className="live-events">
                {journal.data.items.map((incident) => (
                  <article key={incident.id}>
                    <div>
                      <SeverityBadge value={incident.severity} />
                      <time>{relativeTime(incident.created_at)}</time>
                    </div>
                    <Link to={`/incidents/${incident.id}`}>
                      <strong>{incident.type_name}</strong>
                    </Link>{" "}
                    <SourceBadge value={incident.source} />
                    <p>
                      {incident.camera_name} →{" "}
                      {incident.zone_name ?? ru.incidentUi.noZone}
                    </p>
                    <IncidentActions incident={incident} compact />
                  </article>
                ))}
              </div>
            )}
          </aside>
        </div>
      )}
      {simulation && (
        <div className="modal-backdrop">
          <section
            className="modal panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="simulation-title"
          >
            <h2 id="simulation-title">{t.simulate}</h2>
            <form className="edit-form" onSubmit={simulate}>
              <label>
                {t.camera}
                <select
                  required
                  value={simulationCamera}
                  onChange={(e) => setSimulationCamera(e.target.value)}
                >
                  <option value="">{t.selectCamera}</option>
                  {available.map((c) => (
                    <option
                      key={c.id}
                      value={c.id}
                      disabled={!demo.data?.active_camera_ids.includes(c.id)}
                    >
                      {c.name}
                      {demo.data?.real_camera_ids.includes(c.id)
                        ? ` · ${t.realAi}`
                        : ""}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                {t.eventType}
                <select
                  value={simulationType}
                  onChange={(e) => setSimulationType(e.target.value)}
                >
                  {types.data?.map((kind) => (
                    <option value={kind.code} key={kind.code}>
                      {kind.name_ru}
                    </option>
                  ))}
                </select>
              </label>
              {types.isError && (
                <p className="error" role="alert">
                  Не удалось загрузить типы инцидентов.{" "}
                  <button type="button" onClick={() => void types.refetch()}>
                    Повторить
                  </button>
                </p>
              )}
              {error && (
                <p className="error" role="alert">
                  {error}
                </p>
              )}
              <p className="muted">{t.simulateHint}</p>
              <div className="action-row">
                <Button
                  disabled={
                    pending ||
                    !simulationCamera ||
                    types.isPending ||
                    types.isError
                  }
                >
                  {pending ? ru.loading : t.createIncident}
                </Button>
                <Button
                  variant="outline"
                  type="button"
                  disabled={pending}
                  onClick={() => setSimulation(false)}
                >
                  {t.cancel}
                </Button>
              </div>
            </form>
          </section>
        </div>
      )}
    </>
  );
}
