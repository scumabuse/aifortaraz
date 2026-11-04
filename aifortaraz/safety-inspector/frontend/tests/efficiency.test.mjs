import { strict as assert } from "node:assert";
import { test } from "node:test";
import { estimateEfficiency } from "../src/features/project/efficiency.ts";

const example = {
  cameras: 4,
  incidents: 10,
  injuryCost: 500_000,
  cameraCost: 150_000,
  reductionPercent: 20,
};

test("scenario reports annual savings, investment, and months to pay back", () => {
  assert.deepEqual(estimateEfficiency(example), {
    annualSavings: 1_000_000,
    investment: 600_000,
    paybackMonths: 7.2,
  });
});

test("zero avoided cost has no finite payback", () => {
  assert.equal(estimateEfficiency({ ...example, reductionPercent: 0 }).paybackMonths, null);
});

test("invalid counts and percentages are rejected", () => {
  assert.equal(estimateEfficiency({ ...example, cameras: 1.2 }), null);
  assert.equal(estimateEfficiency({ ...example, reductionPercent: 101 }), null);
  assert.equal(estimateEfficiency({ ...example, injuryCost: Infinity }), null);
});
