import type { ReactNode } from "react";
import { Button } from "../../components/ui/button";
import { duration, formatDate } from "../incidents/utils";
import { periodOptions, type Kpis, type usePeriod } from "./data";

export function PeriodControls({
  filter,
  children,
}: {
  filter: ReturnType<typeof usePeriod>;
  children?: ReactNode;
}) {
  return (
    <div className="panel metric-filters">
      <label>
        Период{" "}
        <select
          value={filter.period}
          onChange={(e) => filter.change("period", e.target.value)}
        >
          {Object.entries(periodOptions).map(([key, name]) => (
            <option key={key} value={key}>
              {name}
            </option>
          ))}
        </select>
      </label>
      {filter.period === "custom" && (
        <>
          <label>
            С{" "}
            <input
              type="date"
              value={filter.from}
              onChange={(e) => filter.change("date_from", e.target.value)}
            />
          </label>
          <label>
            По{" "}
            <input
              type="date"
              min={filter.from}
              value={filter.to}
              onChange={(e) => filter.change("date_to", e.target.value)}
            />
          </label>
        </>
      )}
      {children}
    </div>
  );
}
export function QueryState({
  query,
}: {
  query: { isPending: boolean; error: Error | null; refetch: () => unknown };
}) {
  return query.isPending ? (
    <div className="skeleton">Загрузка…</div>
  ) : (
    <div className="panel" role="alert">
      <p>{query.error?.message ?? "Не удалось загрузить данные"}</p>
      <Button variant="outline" onClick={() => void query.refetch()}>
        Повторить
      </Button>
    </div>
  );
}
export function signed(value: number) {
  return new Intl.NumberFormat("ru", {
    signDisplay: "exceptZero",
    maximumFractionDigits: 2,
  }).format(value);
}
export function MetricTiles({
  current,
  previous,
  online,
  falseRate,
}: {
  current: Kpis;
  previous: Kpis;
  online?: string;
  falseRate?: boolean;
}) {
  const tiles: { key: keyof Kpis; label: string; unit?: string }[] = [
    { key: "incidents_total", label: "Инциденты" },
    { key: "open_incidents", label: "Открытые" },
    { key: "critical_incidents", label: "Критические" },
    { key: "compliance_percent", label: "Соответствие СИЗ", unit: "п. п." },
    { key: "mean_reaction_sec", label: "Среднее время реакции", unit: "с" },
  ];
  if (falseRate)
    tiles.push({
      key: "false_positive_percent",
      label: "Ложные срабатывания",
      unit: "п. п.",
    });
  return (
    <div className="incident-kpis metric-kpis">
      {tiles.map(({ key, label, unit }) => {
        const value = current[key],
          before = previous[key];
        const isPercent = unit === "п. п.";
        const shown =
          key === "mean_reaction_sec"
            ? duration(value)
            : value == null
              ? "—"
              : `${value.toLocaleString("ru")}${isPercent ? " %" : ""}`;
        const delta = value == null || before == null ? null : value - before;
        return (
          <div className="panel" key={key}>
            <span>{label}</span>
            <strong>{shown}</strong>
            <small className="muted">
              {delta == null
                ? "Нет данных для сравнения"
                : `${signed(delta)}${unit ? " " + unit : ""} к пред. периоду`}
            </small>
          </div>
        );
      })}
      {online && (
        <div className="panel">
          <span>Камеры онлайн</span>
          <strong>{online}</strong>
          <small className="muted">Текущий статус</small>
        </div>
      )}
    </div>
  );
}
export function Comparison({
  data,
}: {
  data: {
    start: string;
    end: string;
    previous_start: string;
    previous_end: string;
  };
}) {
  return (
    <p className="muted metric-comparison">
      Период {formatDate(data.start)} — {formatDate(data.end)}
      <br />
      Сравнение с {formatDate(data.previous_start)} —{" "}
      {formatDate(data.previous_end)}
    </p>
  );
}
