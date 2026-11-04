import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../../api";
import { DashboardFeed } from "../incidents/DashboardFeed";
import { relativeTime } from "../incidents/utils";
import { Bars, ChartCard, Donut } from "./Charts";
import {
  Comparison,
  MetricTiles,
  PeriodControls,
  QueryState,
} from "./Controls";
import { type Dashboard, usePeriod, usePlans } from "./data";
import { PlanMap } from "./PlanMap";
import "./metrics.css";

export default function DashboardPage() {
  const filter = usePeriod("today");
  const query = useQuery({
    queryKey: ["summary", "dashboard", filter.params.toString()],
    queryFn: ({ signal }) =>
      api<Dashboard>(
        `/api/v1/analytics/overview?${filter.params}&dashboard=true`,
        { signal },
      ),
    refetchInterval: 15_000,
    enabled: filter.valid,
  });
  const plans = usePlans(filter.params, filter.valid);
  const data = query.data;
  const mapped =
    plans.data?.areas.filter((a) =>
      a.cameras.some((c) => c.x != null && c.y != null),
    ) ?? [];
  const max = Math.max(
    0,
    ...mapped.flatMap((a) => a.cameras.map((c) => c.count)),
  );
  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">Механический цех №3</span>
          <h1>Дашборд</h1>
          <p>Оперативная сводка по охране труда</p>
        </div>
      </div>
      <PeriodControls filter={filter} />
      {!filter.valid ? (
        <p className="muted">Выберите начало и конец периода.</p>
      ) : !data ? (
        <QueryState query={query} />
      ) : (
        <>
          <Comparison data={data} />
          <MetricTiles
            current={data.current}
            previous={data.previous}
            online={`${data.cameras.filter((c) => c.status === "online").length} / ${data.cameras.length}`}
          />
          <div className="metric-grid">
            <ChartCard
              title="Инциденты по часам сегодня"
              empty={!data.hourly.some((h) => h.count)}
            >
              <Bars data={data.hourly} />
            </ChartCard>
            <ChartCard
              title="Распределение по типам"
              empty={!data.by_type.length}
            >
              <Donut data={data.by_type} />
            </ChartCard>
          </div>
          <section className="panel dashboard-cameras">
            <h2>Статус камер</h2>
            {!data.cameras.length && (
              <p className="muted">Добавьте камеры, чтобы увидеть их статус.</p>
            )}
            <div className="metric-camera-grid">
              {data.cameras.map((c) => (
                <Link to={`/live/${c.id}`} key={c.id}>
                  <div className="metric-camera-preview">
                    {c.preview_url ? (
                      <img src={c.preview_url} alt="" loading="lazy" />
                    ) : (
                      <span>Нет превью</span>
                    )}
                  </div>
                  <strong>{c.name}</strong>
                  <span className={`plan-status-key ${c.status}`}>
                    {c.status === "online"
                      ? "Онлайн"
                      : c.status === "degraded"
                        ? "Нестабильно"
                        : "Офлайн"}
                  </span>
                  <small className="muted">
                    {c.fps == null ? "—" : `${c.fps} кадр/с`} ·{" "}
                    {c.last_heartbeat_at
                      ? relativeTime(c.last_heartbeat_at)
                      : "нет heartbeat"}
                  </small>
                </Link>
              ))}
            </div>
          </section>
        </>
      )}
      <div className="metric-grid dashboard-details">
        <DashboardFeed />
        <section className="panel">
          <div className="panel-heading">
            <h2>Инциденты на плане</h2>
            <Link className="accent" to={`/plan?${filter.params}`}>
              Открыть план →
            </Link>
          </div>
          {!filter.valid ? (
            <p className="muted">
              Выберите период, чтобы увидеть инциденты на плане.
            </p>
          ) : !plans.data ? (
            <QueryState query={plans} />
          ) : !mapped.length ? (
            <p className="muted">Разместите камеры на плане цеха.</p>
          ) : (
            <div className="metric-mini-grid">
              {mapped.map((area) => (
                <Link
                  to={`/plan?${filter.params}&area_id=${encodeURIComponent(area.id)}`}
                  key={area.id}
                >
                  <strong>{area.name}</strong>
                  <PlanMap area={area} data={plans.data!} max={max} mini />
                </Link>
              ))}
            </div>
          )}
        </section>
      </div>
    </>
  );
}
