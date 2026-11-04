import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Camera, Maximize, Minimize, ShieldCheck } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "../../api";
import { Button } from "../../components/ui/button";
import { type Dashboard } from "../analytics/data";
import { QueryState } from "../analytics/Controls";
import { ConnectionStatus, SoundButton } from "../realtime/Realtime";
import { useLiveStore } from "../realtime/store";
import { type IncidentPage } from "../incidents/types";
import { SeverityBadge } from "../incidents/IncidentsPage";
import { relativeTime } from "../incidents/utils";
import { CameraTile, useLiveCameras } from "../live/LivePage";
import { useDemo } from "../live/DemoSettings";
import type { Overlays } from "../live/types";
import "./project.css";

const overlays: Overlays = {
  boxes: true,
  zones: true,
  trackIds: true,
  labels: true,
  violationsOnly: false,
};

export default function WallPage() {
  const cameras = useLiveCameras();
  const demo = useDemo();
  const stats = useQuery({
    queryKey: ["summary", "wall"],
    queryFn: ({ signal }) =>
      api<Dashboard>("/api/v1/analytics/overview?period=today&dashboard=true", {
        signal,
      }),
    refetchInterval: 15000,
  });
  const feed = useQuery({
    queryKey: ["incidents", "wall"],
    queryFn: ({ signal }) =>
      api<IncidentPage>("/api/v1/incidents?page_size=10&sort=-created_at", {
        signal,
      }),
    refetchInterval: 30000,
  });
  const [fullscreen, setFullscreen] = useState(false);
  const [fullError, setFullError] = useState(false);
  const [rotation, setRotation] = useState(15);
  const [offset, setOffset] = useState(0);
  const [paused, setPaused] = useState(false);
  const active = useMemo(
    () => cameras.data?.filter((c) => c.enabled) ?? [],
    [cameras.data],
  );
  const count = active.length;
  const offsetSafe = count > 4 ? offset % count : 0;
  const visible = Array.from(
    { length: Math.min(4, count) },
    (_, i) => active[(offsetSafe + i) % count],
  );
  const ids = visible.map((c) => c.id).join(",");
  useEffect(() => {
    useLiveStore.getState().setVisibleCameras(ids ? ids.split(",") : []);
    return () => useLiveStore.getState().setVisibleCameras([]);
  }, [ids]);
  useEffect(() => {
    if (count <= 4 || rotation === 0) return;
    const timer = setInterval(
      () => setOffset((value) => (value + 4) % count),
      rotation * 1000,
    );
    return () => clearInterval(timer);
  }, [count, rotation]);
  useEffect(() => {
    const changed = () =>
      setFullscreen(document.fullscreenElement === document.documentElement);
    document.addEventListener("fullscreenchange", changed);
    return () => document.removeEventListener("fullscreenchange", changed);
  }, []);
  async function toggleFull() {
    setFullError(false);
    try {
      if (document.fullscreenElement === document.documentElement)
        await document.exitFullscreen();
      else await document.documentElement.requestFullscreen();
    } catch {
      setFullError(true);
    }
  }
  const kpi = stats.data?.current;
  const online = stats.data?.cameras.filter(
    (c) => c.status === "online",
  ).length;
  const ticker = feed.data?.items ?? [];
  return (
    <main className="monitor-wall">
      <header className="monitor-header">
        <div className="project-brand">
          <ShieldCheck size={25} />
          <h1>Видеостена</h1>
        </div>
        <div className="monitor-actions">
          <ConnectionStatus />
          {demo.data?.enabled && <span className="badge">ДЕМО</span>}
          <SoundButton />
          <label>
            Смена камер{" "}
            <select
              value={rotation}
              onChange={(e) => setRotation(Number(e.target.value))}
            >
              <option value={0}>Вручную</option>
              {[10, 15, 30, 60].map((seconds) => (
                <option key={seconds} value={seconds}>
                  {seconds} с
                </option>
              ))}
            </select>
          </label>
          <Button
            variant="outline"
            disabled={count <= 4}
            onClick={() => setOffset((value) => (value + 4) % count)}
          >
            Следующие камеры
          </Button>
          <Button variant="outline" onClick={() => void toggleFull()}>
            {fullscreen ? <Minimize size={16} /> : <Maximize size={16} />}
            {fullscreen ? "Свернуть" : "На весь экран"}
          </Button>
          <Button variant="ghost" asChild>
            <Link to="/">Дашборд</Link>
          </Button>
        </div>
      </header>
      {fullError && (
        <p className="error" role="alert">
          Браузер не разрешил полноэкранный режим.
        </p>
      )}
      {stats.isError ? (
        <QueryState query={stats} />
      ) : (
        <section className="monitor-kpis" aria-label="Показатели за сегодня">
          {[
            ["Инциденты сегодня", kpi?.incidents_total ?? "—"],
            ["Критические", kpi?.critical_incidents ?? "—"],
            [
              "Соответствие СИЗ",
              kpi?.compliance_percent == null
                ? "—"
                : `${kpi.compliance_percent} %`,
            ],
            [
              "Камеры онлайн",
              online == null
                ? "—"
                : `${online} / ${stats.data?.cameras.length}`,
            ],
          ].map(([label, value]) => (
            <div className="panel" key={label}>
              <span>{label}</span>
              <strong>{value}</strong>
            </div>
          ))}
        </section>
      )}
      {!cameras.data ? (
        <QueryState query={cameras} />
      ) : (
        <section className="monitor-grid" aria-label="Камеры">
          {Array.from({ length: 4 }, (_, i) => {
            const camera = visible[i];
            return camera ? (
              <div className="monitor-tile" key={camera.id}>
                <CameraTile
                  camera={camera}
                  overlays={overlays}
                  demoActive={
                    !!demo.data?.active_camera_ids.includes(camera.id)
                  }
                />
              </div>
            ) : (
              <div className="monitor-tile monitor-empty" key={`empty-${i}`}>
                <Camera size={28} />
                <span>Камера не назначена</span>
              </div>
            );
          })}
        </section>
      )}
      <section className="monitor-ticker" aria-label="Последние инциденты">
        <strong>Последние инциденты</strong>
        {feed.isError ? (
          <Button variant="ghost" onClick={() => void feed.refetch()}>
            Повторить
          </Button>
        ) : ticker.length ? (
          <>
            <div className={`monitor-ticker-window ${paused ? "paused" : ""}`}>
              <div className="monitor-ticker-track">
                {[false, true].map((copy) => (
                  <div
                    className="monitor-ticker-group"
                    key={String(copy)}
                    aria-hidden={copy || undefined}
                  >
                    {ticker.map((item) => (
                      <Link
                        key={item.id}
                        to={`/incidents/${item.id}`}
                        tabIndex={copy ? -1 : undefined}
                      >
                        <SeverityBadge value={item.severity} /> {item.type_name}{" "}
                        · {item.camera_name} · {relativeTime(item.created_at)}
                      </Link>
                    ))}
                  </div>
                ))}
              </div>
            </div>
            <Button
              variant="ghost"
              aria-pressed={paused}
              onClick={() => setPaused((value) => !value)}
            >
              {paused ? "Продолжить" : "Пауза"}
            </Button>
          </>
        ) : (
          <span className="muted">
            {feed.isPending ? "Загрузка…" : "Инцидентов пока нет"}
          </span>
        )}
      </section>
    </main>
  );
}
