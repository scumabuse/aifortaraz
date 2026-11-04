import { readSession } from "../../api";
import ru from "../../locales/ru.json";
import type { IncidentRecord } from "./types";
let timezone = "Asia/Yekaterinburg";
export function setTimezone(value: string) {
  try {
    new Intl.DateTimeFormat("ru-RU", { timeZone: value }).format();
    timezone = value;
  } catch {
    /* Keep the valid enterprise fallback. */
  }
}
export function getTimezone() {
  return timezone;
}
export function formatDate(value: string | null) {
  return value
    ? new Intl.DateTimeFormat("ru-RU", {
        timeZone: timezone,
        day: "2-digit",
        month: "2-digit",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      }).format(new Date(value))
    : "—";
}
export function relativeTime(value: string) {
  const seconds = Math.max(
    0,
    Math.floor((Date.now() - new Date(value).getTime()) / 1000),
  );
  if (seconds < 60) return ru.incidentUi.justNow;
  if (seconds < 3600)
    return `${Math.floor(seconds / 60)} ${ru.incidentUi.minutesAgo}`;
  if (seconds < 86400)
    return `${Math.floor(seconds / 3600)} ${ru.incidentUi.hoursAgo}`;
  return `${Math.floor(seconds / 86400)} ${ru.incidentUi.daysAgo}`;
}
export function duration(value: number | null) {
  if (value === null) return "—";
  if (value < 60) return `${Math.round(value)} ${ru.incidentUi.seconds}`;
  return `${Math.floor(value / 60)} ${ru.incidentUi.minutes} ${Math.round(value % 60)} ${ru.incidentUi.seconds}`;
}
export function canAct(incident: IncidentRecord, action: string) {
  const user = readSession()?.user;
  if (!user) return false;
  if (["admin", "engineer"].includes(user.role)) return true;
  return (
    user.role === "master" &&
    user.area_id === incident.area_id &&
    ["acknowledge", "resolve", "comment"].includes(action)
  );
}
export function ppeName(code: string) {
  return (
    ru.incidentUi.ppeNames[code as keyof typeof ru.incidentUi.ppeNames] ?? code
  );
}
export function incidentDescription(incident: IncidentRecord) {
  const missing = incident.details.missing_ppe;
  if (missing?.length)
    return `${ru.incidentUi.missing}: ${missing.map(ppeName).join(", ")}`;
  return (
    incident.details.title ??
    (incident.track_id !== null
      ? `${ru.incidentUi.track} #${incident.track_id}`
      : incident.type_name)
  );
}
export function periodDate(date: string, end = false) {
  if (!date) return "";
  const [year, month, day] = date.split("-").map(Number);
  const target = Date.UTC(
    year,
    month - 1,
    day,
    end ? 23 : 0,
    end ? 59 : 0,
    end ? 59 : 0,
    end ? 999 : 0,
  );
  let guess = target;
  for (let i = 0; i < 3; i++) {
    const parts = new Intl.DateTimeFormat("en-GB", {
      timeZone: timezone,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hourCycle: "h23",
    }).formatToParts(new Date(guess));
    const get = (type: string) =>
      Number(parts.find((p) => p.type === type)?.value);
    const wall = Date.UTC(
      get("year"),
      get("month") - 1,
      get("day"),
      get("hour"),
      get("minute"),
      get("second"),
      end ? 999 : 0,
    );
    guess += target - wall;
  }
  return new Date(guess).toISOString();
}
export function incidentParams(search: URLSearchParams) {
  const result = new URLSearchParams();
  for (const key of [
    "camera_id",
    "zone_id",
    "type",
    "severity",
    "status",
    "assignee_id",
    "q",
    "page",
    "page_size",
    "sort",
    "only_unacknowledged",
  ]) {
    const value = search.get(key);
    if (value) result.set(key, value);
  }
  if (search.get("fromDate"))
    result.set("from", periodDate(search.get("fromDate")!));
  if (search.get("toDate"))
    result.set("to", periodDate(search.get("toDate")!, true));
  for (const key of ["from", "to"]) {
    if (search.has(key)) result.set(key, search.get(key)!);
  }
  return result;
}
