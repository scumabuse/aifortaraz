import { strict as assert } from "node:assert";
import { createRequire } from "node:module";
import { test } from "node:test";
import {
  BOX_LABELS,
  OBJECT_COLORS,
  PPE_CODES,
  boxCaption,
  isVehicleLabel,
  objectColor,
  objectLabel,
  ppeMark,
  ppeState,
} from "../src/features/live/labels.ts";

const ru = createRequire(import.meta.url)("../src/locales/ru.json");
const names = ru.liveUi.objectTypes;

test("every contract box label has its own Russian name", () => {
  assert.equal(BOX_LABELS.length, 19);
  for (const label of BOX_LABELS) {
    assert.ok(Object.hasOwn(names, label), label);
    assert.match(names[label], /^[А-ЯЁ][а-яё ]+$/u, label);
    assert.notEqual(objectLabel(label, names), names.other, label);
  }
});

test("PPE object labels reuse the PPE names wording", () => {
  assert.deepEqual(Object.keys(ru.incidentUi.ppeNames), [...PPE_CODES]);
  for (const code of PPE_CODES)
    assert.equal(names[code].toLowerCase(), ru.incidentUi.ppeNames[code]);
});

test("vehicles, car and bare-part classes are translated", () => {
  assert.deepEqual(
    ["person", "vehicle", "forklift", "truck", "car"].map((label) =>
      objectLabel(label, names),
    ),
    ["Человек", "Техника", "Погрузчик", "Грузовик", "Автомобиль"],
  );
  assert.deepEqual(
    ["head", "face", "hands", "foot", "ear"].map((label) =>
      objectLabel(label, names),
    ),
    ["Голова", "Лицо", "Руки", "Ступня", "Ухо"],
  );
  assert.equal(objectLabel("harness", names), "Страховочная привязь");
});

test("unknown labels keep the Объект fallback", () => {
  for (const label of ["bicycle", "", "constructor", "toString", "Person"])
    assert.equal(objectLabel(label, names), "Объект", label);
});

test("car is drawn in the vehicle colour like forklift, vehicle and truck", () => {
  for (const label of ["car", "forklift", "vehicle", "truck"]) {
    assert.equal(isVehicleLabel(label), true, label);
    for (const state of [undefined, null, "ok", "violation", "danger"])
      assert.equal(objectColor(label, state), OBJECT_COLORS.vehicle, label);
  }
  for (const label of ["person", "helmet", "head"])
    assert.equal(isVehicleLabel(label), false, label);
});

test("non-vehicle boxes are coloured by their state", () => {
  assert.equal(objectColor("person", "danger"), OBJECT_COLORS.danger);
  assert.equal(objectColor("person", "violation"), OBJECT_COLORS.violation);
  assert.equal(objectColor("person", "ok"), OBJECT_COLORS.ok);
  assert.equal(objectColor("person", undefined), OBJECT_COLORS.ok);
  assert.notEqual(OBJECT_COLORS.ok, OBJECT_COLORS.vehicle);
});

test("ppe true is ✓, false is ✗ and null is ? (never ✗)", () => {
  assert.equal(ppeMark(true), "✓");
  assert.equal(ppeMark(false), "✗");
  assert.equal(ppeMark(null), "?");
  assert.equal(ppeMark(undefined), "?");
  assert.deepEqual([true, false, null].map(ppeState), [
    "present",
    "missing",
    "unknown",
  ]);
});

test("each ppe state has a Russian hint in the objects panel", () => {
  assert.deepEqual(
    [true, false, null].map((value) => ru.liveUi[ppeState(value)]),
    ["Есть", "Нет", "Не определено"],
  );
});

test("snapshot caption joins the track id and the Russian label", () => {
  const bbox = [0.41, 0.22, 0.12, 0.46];
  assert.equal(
    boxCaption({ track_id: 17, label: "person", bbox }, names),
    "#17 · Человек",
  );
  assert.equal(
    boxCaption({ track_id: 0, label: "head", bbox }, names),
    "#0 · Голова",
  );
  assert.equal(
    boxCaption({ track_id: null, label: "forklift", bbox }, names),
    "Погрузчик",
  );
  assert.equal(boxCaption({ label: "car", bbox }, names), "Автомобиль");
  assert.equal(boxCaption({ label: "drone", bbox }, names), "Объект");
});
