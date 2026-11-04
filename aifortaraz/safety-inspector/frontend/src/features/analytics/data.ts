import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { api } from "../../api";

export type Period = "today" | "shift" | "7d" | "30d" | "custom";
export type Count = { name: string; count: number };
export type Kpis = {
  incidents_total: number;
  open_incidents: number;
  critical_incidents: number;
  mean_reaction_sec: number | null;
  compliance_percent: number | null;
  false_positive_percent: number | null;
};
export type CameraStatus = {
  id: string;
  name: string;
  status: "online" | "offline" | "degraded";
};
export type DashboardCamera = CameraStatus & {
  preview_url: string | null;
  fps: number | null;
  last_heartbeat_at: string | null;
};
export type CameraRollup = Count & {
  id: string;
  critical: number;
  reaction: number | null;
  previous: number;
};
type Overview = {
  start: string;
  end: string;
  previous_start: string;
  previous_end: string;
  timezone: string;
  current: Kpis;
  previous: Kpis;
  by_type: Count[];
};
export type Dashboard = Overview & {
  cameras: DashboardCamera[];
  hourly: Count[];
};
export type Analytics = Overview & {
  types: Record<string, string>;
  daily: {
    date: string;
    counts: Record<string, number>;
    compliance: number | null;
  }[];
  heat: { day: number; hour: number; count: number }[];
  zones: Count[];
  reaction: Count[];
  cameras: CameraRollup[];
};
export type Marker = CameraStatus & {
  x: number | null;
  y: number | null;
  count: number;
  latest: string | null;
  zones: { id: string; name: string; color: string }[];
};
export type PlanArea = {
  id: string;
  name: string;
  image: { url: string; width: number; height: number } | null;
  cameras: Marker[];
};
export type Plans = { start: string; end: string; areas: PlanArea[] };

export const periodOptions: Record<Period, string> = {
  today: "Сегодня",
  shift: "Текущая смена",
  "7d": "7 дней",
  "30d": "30 дней",
  custom: "Свой период",
};

export function usePeriod(fallback: Period) {
  const [search, setSearch] = useSearchParams();
  const candidate = search.get("period");
  const period =
    candidate && Object.hasOwn(periodOptions, candidate)
      ? (candidate as Period)
      : fallback;
  const from = search.get("date_from") ?? "";
  const to = search.get("date_to") ?? "";
  const valid = period !== "custom" || (!!from && !!to && from <= to);
  const params = new URLSearchParams({ period });
  if (period === "custom") {
    params.set("date_from", from);
    params.set("date_to", to);
  }
  function change(key: string, value: string, clear?: string) {
    setSearch(
      (previous) => {
        const next = new URLSearchParams(previous);
        if (value) next.set(key, value);
        else next.delete(key);
        if (clear) next.delete(clear);
        return next;
      },
      { replace: true },
    );
  }
  return { search, period, from, to, valid, params, change };
}

export function usePlans(params: URLSearchParams, enabled = true) {
  const path = "/api/v1/plan?" + params;
  return useQuery({
    queryKey: ["plan", path],
    queryFn: ({ signal }) => api<Plans>(path, { signal }),
    refetchInterval: 15_000,
    enabled,
  });
}
