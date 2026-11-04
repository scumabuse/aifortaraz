import { strict as assert } from "node:assert";
import { test } from "node:test";
import {
  isPpeDetectable,
  undetectablePpe,
  zonesUndetectablePpe,
} from "../src/features/cameras/ppe.ts";

const sh17 = [
  "helmet",
  "vest",
  "glasses",
  "gloves",
  "mask",
  "boots",
  "earmuffs",
  "face_shield",
];
const zone = (required_ppe, enabled = true) => ({
  enabled,
  rules: { required_ppe },
});

test("unknown support (null or absent) marks nothing", () => {
  for (const supported of [null, undefined]) {
    assert.equal(isPpeDetectable("harness", supported), true);
    assert.deepEqual(undetectablePpe(["helmet", "harness"], supported), []);
    assert.deepEqual(zonesUndetectablePpe([zone(["harness"])], supported), []);
  }
});

test("each PPE option the model cannot detect is marked", () => {
  assert.equal(isPpeDetectable("helmet", sh17), true);
  assert.equal(isPpeDetectable("harness", sh17), false);
  assert.deepEqual(undetectablePpe(["helmet", "harness"], sh17), ["harness"]);
  assert.deepEqual(undetectablePpe(["helmet", "vest"], []), ["helmet", "vest"]);
});

test("the camera card lists unsupported PPE of enabled zones once", () => {
  const zones = [
    zone(["helmet", "harness"]),
    zone(["harness", "gloves"]),
    zone(["glasses"], false),
  ];
  assert.deepEqual(zonesUndetectablePpe(zones, sh17), ["harness"]);
  assert.deepEqual(zonesUndetectablePpe(zones, ["helmet"]), [
    "harness",
    "gloves",
  ]);
  assert.deepEqual(zonesUndetectablePpe([zone([])], sh17), []);
  assert.deepEqual(zonesUndetectablePpe([], []), []);
});
