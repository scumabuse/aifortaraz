import type { ZoneRecord } from "./types";

/** PPE codes the camera's AI model can detect; null or absent means unknown (no AI heartbeat with extra.supported_ppe yet). */
export type SupportedPpe = readonly string[] | null | undefined;

export function isPpeDetectable(code: string, supported: SupportedPpe) {
  return !Array.isArray(supported) || supported.includes(code);
}
export function undetectablePpe(
  codes: readonly string[],
  supported: SupportedPpe,
): string[] {
  return [
    ...new Set(codes.filter((code) => !isPpeDetectable(code, supported))),
  ];
}
/** Required PPE of the enabled zones (the ones sent to the AI) that the current model cannot detect. */
export function zonesUndetectablePpe(
  zones: readonly Pick<ZoneRecord, "enabled" | "rules">[],
  supported: SupportedPpe,
): string[] {
  return undetectablePpe(
    zones
      .filter((zone) => zone.enabled)
      .flatMap((zone) => zone.rules.required_ppe),
    supported,
  );
}
