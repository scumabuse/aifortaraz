export type EfficiencyInput = {
  cameras: number;
  incidents: number;
  injuryCost: number;
  cameraCost: number;
  reductionPercent: number;
};

export function estimateEfficiency(input: EfficiencyInput) {
  const { cameras, incidents, injuryCost, cameraCost, reductionPercent } =
    input;
  if (
    !Object.values(input).every(
      (value) => Number.isFinite(value) && value >= 0,
    ) ||
    !Number.isInteger(cameras) ||
    !Number.isInteger(incidents) ||
    reductionPercent > 100
  )
    return null;
  const annualSavings = incidents * (reductionPercent / 100) * injuryCost;
  const investment = cameras * cameraCost;
  const paybackMonths =
    annualSavings > 0 ? (12 * investment) / annualSavings : null;
  if (![annualSavings, investment, paybackMonths ?? 0].every(Number.isFinite))
    return null;
  return { annualSavings, investment, paybackMonths };
}
