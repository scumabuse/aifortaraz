import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api, type DictionaryItem } from "../../api";
import { Button } from "../../components/ui/button";
import { duration } from "../incidents/utils";
import type { CameraRecord } from "../cameras/types";
import { Bars, ChartCard, Series } from "./Charts";
import {
  Comparison,
  MetricTiles,
  PeriodControls,
  QueryState,
  signed,
} from "./Controls";
import { type Analytics, usePeriod } from "./data";
import "./metrics.css";

const weekdays = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"];
function WeekHeat({ points }: { points: Analytics["heat"] }) {
  const values = new Map(points.map((p) => [p.day * 24 + p.hour, p.count]));
  const max = Math.max(1, ...points.map((p) => p.count));
  return (
    <div className="metric-heat-scroll">
      <table className="metric-heat">
        <thead>
          <tr>
            <th scope="col">День</th>
            {Array.from({ length: 24 }, (_, hour) => (
              <th scope="col" key={hour}>
                {hour}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {weekdays.map((name, day) => (
            <tr key={name}>
              <th scope="row">{name}</th>
              {Array.from({ length: 24 }, (_, hour) => {
                const count = values.get(day * 24 + hour) ?? 0;
                return (
                  <td
                    key={hour}
                    title={`${name} ${hour}:00 · ${count}`}
                    style={{
                      background: count
                        ? `rgba(50,206,162,${0.15 + (0.62 * count) / max})`
                        : undefined,
                    }}
                  >
                    {count || "·"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function exportCsv(data: Analytics) {
  const rows: (number | string | null)[][] = [
    [
      "Раздел",
      "Период / объект",
      "Показатель",
      "Значение",
      "Предыдущий период",
    ],
    ["Период", data.start, "До", data.end, null],
    ["Сравнение", data.previous_start, "До", data.previous_end, null],
    ...Object.entries(data.current).map(([name, value]) => [
      "KPI",
      "",
      name,
      value,
      data.previous[name as keyof Analytics["previous"]],
    ]),
    ...data.daily.flatMap((day) => [
      ["Динамика", day.date, "Соответствие СИЗ, %", day.compliance, null],
      ...Object.entries(day.counts).map(([code, value]) => [
        "Динамика",
        day.date,
        data.types[code] ?? code,
        value,
        null,
      ]),
    ]),
    ...data.cameras.flatMap((c) => [
      ["Камеры", c.name, "Инциденты", c.count, c.previous],
      ["Камеры", c.name, "Критические", c.critical, null],
      ["Камеры", c.name, "Реакция, с", c.reaction, null],
    ]),
    ...data.zones.map((z) => ["Зоны", z.name, "Инциденты", z.count, null]),
    ...data.by_type.map((t) => ["Типы", t.name, "Инциденты", t.count, null]),
    ...data.reaction.map((b) => [
      "Реакция",
      b.name,
      "Инциденты",
      b.count,
      null,
    ]),
    ...data.heat.map((h) => [
      "Час × день",
      `${weekdays[h.day]} ${h.hour}:00`,
      "Инциденты",
      h.count,
      null,
    ]),
  ];
  const csv = rows
    .map((row) =>
      row
        .map((value) => {
          const raw = String(value ?? "");
          const cell =
            typeof value === "string" && /^[\s]*[=+@-]/.test(raw)
              ? "'" + raw
              : raw;
          return '"' + cell.replaceAll('"', '""') + '"';
        })
        .join(";"),
    )
    .join("\r\n");
  const url = URL.createObjectURL(
    new Blob(["\uFEFF", csv], { type: "text/csv;charset=utf-8" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = "inspector-analytics.csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function AnalyticsPage() {
  const filter = usePeriod("7d");
  const areaId = filter.search.get("area_id") ?? "";
  const cameraId = filter.search.get("camera_id") ?? "";
  const areas = useQuery({
    queryKey: ["analytics-areas"],
    queryFn: () => api<DictionaryItem[]>("/api/v1/settings/areas"),
  });
  const cameras = useQuery({
    queryKey: ["cameras-options"],
    queryFn: () => api<CameraRecord[]>("/api/v1/cameras"),
  });
  if (areaId) filter.params.set("area_id", areaId);
  if (cameraId) filter.params.set("camera_id", cameraId);
  const path = "/api/v1/analytics/overview?" + filter.params;
  const query = useQuery({
    queryKey: ["summary", "analytics", path],
    queryFn: ({ signal }) => api<Analytics>(path, { signal }),
    enabled: filter.valid,
    refetchInterval: 30_000,
  });
  const data = query.data;
  return (
    <div className="analytics-print">
      <div className="page-heading">
        <div>
          <span className="eyebrow">Механический цех №3</span>
          <h1>Аналитика</h1>
          <p>Сводка за период и сравнение с предыдущим</p>
        </div>
        <div className="metric-actions">
          <Button
            variant="outline"
            disabled={!data}
            onClick={() => data && exportCsv(data)}
          >
            CSV данных
          </Button>
          <Button
            variant="outline"
            disabled={!data}
            onClick={() => window.print()}
          >
            Печать / PDF
          </Button>
        </div>
      </div>
      <PeriodControls filter={filter}>
        <label>
          Участок{" "}
          <select
            value={areaId}
            onChange={(e) =>
              filter.change("area_id", e.target.value, "camera_id")
            }
          >
            <option value="">Все участки</option>
            {areas.data?.map((a) => (
              <option key={a.id} value={a.id}>
                {a.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Камера{" "}
          <select
            value={cameraId}
            onChange={(e) => filter.change("camera_id", e.target.value)}
          >
            <option value="">Все камеры</option>
            {cameras.data
              ?.filter((c) => !areaId || c.area_id === areaId)
              .map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
          </select>
        </label>
      </PeriodControls>
      {(areas.isError || cameras.isError) && (
        <QueryState query={areas.isError ? areas : cameras} />
      )}
      {!filter.valid ? (
        <p className="muted">Выберите начало и конец периода.</p>
      ) : !data ? (
        <QueryState query={query} />
      ) : (
        <>
          <p className="metric-print-filter">
            Участок: {areas.data?.find((a) => a.id === areaId)?.name ?? "все"} ·
            Камера:{" "}
            {cameras.data?.find((c) => c.id === cameraId)?.name ?? "все"}
          </p>
          <Comparison data={data} />
          <MetricTiles
            current={data.current}
            previous={data.previous}
            falseRate
          />
          <ChartCard
            title="Динамика инцидентов по дням"
            empty={!data.current.incidents_total}
          >
            <Series
              data={data.daily}
              series={Object.entries(data.types)
                .filter(([key]) => data.daily.some((d) => d.counts[key]))
                .map(([key, name]) => ({ key: `counts.${key}`, name }))}
            />
          </ChartCard>
          <div className="metric-grid">
            <ChartCard
              title="Инциденты по камерам"
              empty={!data.current.incidents_total}
            >
              <Bars
                data={[...data.cameras].sort((a, b) => b.count - a.count)}
                horizontal
              />
            </ChartCard>
            <ChartCard title="Инциденты по зонам" empty={!data.zones.length}>
              <Bars data={data.zones} horizontal />
            </ChartCard>
            <ChartCard
              title="Соответствие СИЗ по дням"
              empty={!data.daily.some((d) => d.compliance !== null)}
            >
              <Series
                data={data.daily}
                series={[{ key: "compliance", name: "Соответствие СИЗ, %" }]}
                percent
              />
            </ChartCard>
            <ChartCard
              title="Распределение времени реакции"
              empty={!data.reaction.some((b) => b.count)}
            >
              <Bars data={data.reaction} />
            </ChartCard>
            <ChartCard
              title="Топ-5 типов нарушений"
              empty={!data.by_type.length}
            >
              <Bars data={data.by_type.slice(0, 5)} horizontal />
            </ChartCard>
            <ChartCard
              title="Инциденты по часу и дню недели"
              empty={!data.heat.length}
            >
              <WeekHeat points={data.heat} />
            </ChartCard>
          </div>
          <section className="panel metric-table-panel">
            <h2>Сводка по камерам</h2>
            {!data.cameras.length ? (
              <p className="muted">Нет камер по выбранным фильтрам.</p>
            ) : (
              <div className="metric-table-scroll">
                <table className="metric-table">
                  <thead>
                    <tr>
                      <th scope="col">Камера</th>
                      <th scope="col">Инциденты</th>
                      <th scope="col">Критические</th>
                      <th scope="col">Средняя реакция</th>
                      <th scope="col">Пред. период</th>
                      <th scope="col">Тренд</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.cameras.map((c) => (
                      <tr key={c.id}>
                        <th scope="row">
                          <Link className="accent" to={`/cameras/${c.id}`}>
                            {c.name}
                          </Link>
                        </th>
                        <td>{c.count}</td>
                        <td>{c.critical}</td>
                        <td>{duration(c.reaction)}</td>
                        <td>{c.previous}</td>
                        <td>{signed(c.count - c.previous)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}
