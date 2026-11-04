import type { Box } from "../incidents/types";

/** PPE codes of the integration contract (section 4.1), in contract order. */
export const PPE_CODES = [
  "helmet",
  "vest",
  "glasses",
  "gloves",
  "mask",
  "boots",
  "earmuffs",
  "harness",
  "face_shield",
] as const;
export const VEHICLE_LABELS = ["vehicle", "forklift", "truck", "car"] as const;
/** Every box label the AI module may send (contract section 4.1). */
export const BOX_LABELS = [
  "person",
  ...VEHICLE_LABELS,
  ...PPE_CODES,
  "head",
  "face",
  "hands",
  "foot",
  "ear",
] as const;
export const OBJECT_COLORS = {
  vehicle: "#73b9ff",
  danger: "#ff6673",
  violation: "#f3a15c",
  ok: "#6ee0b7",
} as const;
export const PPE_MARKS = { present: "✓", missing: "✗", unknown: "?" } as const;
export type PpeState = keyof typeof PPE_MARKS;

/** Russian name of a box label; labels outside the dictionary use its `other` entry. */
export function objectLabel(
  label: string,
  names: Readonly<Record<string, string>> & { other: string },
): string {
  return Object.hasOwn(names, label) ? names[label] : names.other;
}
export function isVehicleLabel(label: string): boolean {
  return (VEHICLE_LABELS as readonly string[]).includes(label);
}
export function objectColor(label: string, state?: string | null): string {
  if (isVehicleLabel(label)) return OBJECT_COLORS.vehicle;
  if (state === "danger") return OBJECT_COLORS.danger;
  if (state === "violation") return OBJECT_COLORS.violation;
  return OBJECT_COLORS.ok;
}
/** A person's PPE map value: true = worn, false = missing, null (or anything else) = unknown. */
export function ppeState(value: unknown): PpeState {
  return value === true ? "present" : value === false ? "missing" : "unknown";
}
export function ppeMark(value: unknown): string {
  return PPE_MARKS[ppeState(value)];
}
/** Caption drawn next to an incident snapshot box, e.g. "#17 · Человек". */
export function boxCaption(
  box: Pick<Box, "label" | "track_id">,
  names: Readonly<Record<string, string>> & { other: string },
): string {
  const label = objectLabel(box.label, names);
  return box.track_id === null || box.track_id === undefined
    ? label
    : `#${box.track_id} · ${label}`;
}
