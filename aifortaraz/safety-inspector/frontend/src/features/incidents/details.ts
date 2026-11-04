export const INCIDENT_SOURCES = ["ai", "mock", "manual"] as const;
export type IncidentSource = (typeof INCIDENT_SOURCES)[number];

export function sourceBadgeKind(source: unknown): IncidentSource | null {
  return (INCIDENT_SOURCES as readonly unknown[]).includes(source)
    ? (source as IncidentSource)
    : null;
}
/** Value of the incident list `source` query parameter: known codes only, comma-separated, no repeats. */
export function normalizeSourceFilter(
  value: string | null | undefined,
): string {
  const codes = (value ?? "")
    .split(",")
    .map((code) => code.trim())
    .filter((code) => sourceBadgeKind(code) !== null);
  return [...new Set(codes)].join(",");
}

type Format =
  "ppe" | "seconds" | "meters" | "track" | "percent" | "count" | "text";
const FIELD_FORMATS = {
  missing_ppe: "ppe",
  present_ppe: "ppe",
  duration_sec: "seconds",
  distance_m: "meters",
  person_track_id: "track",
  vehicle_track_id: "track",
  pose_conf: "percent",
  immobile_sec: "seconds",
  count: "count",
  max_persons: "count",
  title: "text",
} satisfies Record<string, Format>;
export type DetailField = keyof typeof FIELD_FORMATS;
/** details keys shown for each incident type (contract section 4.1). */
export const DETAIL_FIELDS: Readonly<Record<string, readonly DetailField[]>> = {
  ppe_violation: ["missing_ppe", "present_ppe"],
  zone_intrusion: ["duration_sec"],
  vehicle_proximity: ["distance_m", "person_track_id", "vehicle_track_id"],
  fall_detected: ["pose_conf"],
  person_immobile: ["immobile_sec"],
  overcrowding: ["count", "max_persons"],
  custom: ["title"],
};
export interface DetailDictionary {
  detailFields: Readonly<Record<DetailField, string>>;
  ppeNames: Readonly<Record<string, string>>;
  seconds: string;
  meters: string;
}

const decimal = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 });
function formatValue(
  format: Format,
  value: unknown,
  dictionary: DetailDictionary,
): string | null {
  switch (format) {
    case "ppe":
      if (!Array.isArray(value) || !value.every((v) => typeof v === "string"))
        return null;
      return value.length
        ? value
            .map((code) =>
              Object.hasOwn(dictionary.ppeNames, code)
                ? dictionary.ppeNames[code]
                : code,
            )
            .join(", ")
        : "—";
    case "seconds":
    case "meters":
      if (typeof value !== "number" || !Number.isFinite(value) || value < 0)
        return null;
      return `${decimal.format(value)} ${format === "seconds" ? dictionary.seconds : dictionary.meters}`;
    case "track":
      return Number.isInteger(value) ? `#${value}` : null;
    case "percent":
      if (typeof value !== "number" || !(value >= 0 && value <= 1)) return null;
      return `${Math.round(value * 100)}%`;
    case "count":
      return Number.isInteger(value) && (value as number) >= 0
        ? String(value)
        : null;
    case "text":
      return typeof value === "string" && value.trim() ? value : null;
  }
}
/** Russian label/value pairs of the per-type details; missing or malformed values are left out. */
export function incidentDetailRows(
  incident: { type_code: string; details: Readonly<Record<string, unknown>> },
  dictionary: DetailDictionary,
): [string, string][] {
  const fields = Object.hasOwn(DETAIL_FIELDS, incident.type_code)
    ? DETAIL_FIELDS[incident.type_code]
    : [];
  const rows: [string, string][] = [];
  for (const field of fields) {
    const text = formatValue(
      FIELD_FORMATS[field],
      incident.details[field],
      dictionary,
    );
    if (text !== null) rows.push([dictionary.detailFields[field], text]);
  }
  return rows;
}
