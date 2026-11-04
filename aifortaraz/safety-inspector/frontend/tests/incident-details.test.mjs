import { strict as assert } from "node:assert";
import { createRequire } from "node:module";
import { test } from "node:test";
import {
  DETAIL_FIELDS,
  INCIDENT_SOURCES,
  incidentDetailRows,
  normalizeSourceFilter,
  sourceBadgeKind,
} from "../src/features/incidents/details.ts";

const ru = createRequire(import.meta.url)("../src/locales/ru.json");
const t = ru.incidentUi;
const rows = (type_code, details) =>
  incidentDetailRows({ type_code, details }, t);
const boxes = [
  { track_id: 17, label: "person", bbox: [0.41, 0.22, 0.12, 0.46], conf: 0.9 },
];

test("ppe_violation shows missing and present PPE in Russian", () => {
  assert.deepEqual(
    rows("ppe_violation", {
      missing_ppe: ["helmet", "glasses"],
      present_ppe: ["vest"],
      boxes,
      frame_size: [1280, 720],
    }),
    [
      ["Отсутствуют СИЗ", "каска, очки"],
      ["Надеты СИЗ", "жилет"],
    ],
  );
  assert.deepEqual(
    rows("ppe_violation", { missing_ppe: ["harness"], present_ppe: [] }),
    [
      ["Отсутствуют СИЗ", "страховочная привязь"],
      ["Надеты СИЗ", "—"],
    ],
  );
});

test("every other incident type shows its contract fields", () => {
  assert.deepEqual(rows("zone_intrusion", { duration_sec: 14 }), [
    ["Время в зоне", "14 с"],
  ]);
  assert.deepEqual(
    rows("vehicle_proximity", {
      distance_m: 1.8,
      person_track_id: 17,
      vehicle_track_id: 3,
    }),
    [
      ["Расстояние до техники", "1,8 м"],
      ["Трек человека", "#17"],
      ["Трек техники", "#3"],
    ],
  );
  assert.deepEqual(rows("fall_detected", { pose_conf: 0.93 }), [
    ["Уверенность позы", "93%"],
  ]);
  assert.deepEqual(rows("person_immobile", { immobile_sec: 65.5 }), [
    ["Время без движения", "65,5 с"],
  ]);
  assert.deepEqual(rows("overcrowding", { count: 4, max_persons: 2 }), [
    ["Людей в зоне", "4"],
    ["Максимум людей", "2"],
  ]);
  assert.deepEqual(rows("custom", { title: "Учебное событие" }), [
    ["Название события", "Учебное событие"],
  ]);
});

test("every detail field has a Russian label", () => {
  for (const fields of Object.values(DETAIL_FIELDS))
    for (const field of fields)
      assert.match(t.detailFields[field], /^[А-ЯЁ]/u, field);
});

test("missing or malformed values are left out instead of shown wrong", () => {
  assert.deepEqual(rows("zone_intrusion", { boxes }), []);
  assert.deepEqual(
    rows("vehicle_proximity", {
      distance_m: "near",
      person_track_id: 1.5,
      vehicle_track_id: null,
    }),
    [],
  );
  assert.deepEqual(rows("fall_detected", { pose_conf: 1.7 }), []);
  assert.deepEqual(rows("overcrowding", { count: -1, max_persons: 2.5 }), []);
  assert.deepEqual(rows("custom", { title: "   " }), []);
  assert.deepEqual(rows("ppe_violation", { missing_ppe: "helmet" }), []);
  assert.deepEqual(rows("ppe_violation", { missing_ppe: ["drone_goggles"] }), [
    ["Отсутствуют СИЗ", "drone_goggles"],
  ]);
  assert.deepEqual(rows("constructor", { title: "x" }), []);
});

test("incident sources map to the AI, демо and вручную badges", () => {
  assert.deepEqual(
    INCIDENT_SOURCES.map((source) => t.sourceBadges[sourceBadgeKind(source)]),
    ["AI", "демо", "вручную"],
  );
  for (const source of ["", "AI", "camera", null, undefined, 1])
    assert.equal(sourceBadgeKind(source), null, String(source));
});

test("the source filter sends known codes only, comma-separated", () => {
  assert.equal(normalizeSourceFilter("ai"), "ai");
  assert.equal(normalizeSourceFilter("mock"), "mock");
  assert.equal(normalizeSourceFilter(" ai , mock,ai"), "ai,mock");
  assert.equal(normalizeSourceFilter("mock,bogus,manual"), "mock,manual");
  assert.equal(normalizeSourceFilter("bogus"), "");
  assert.equal(normalizeSourceFilter(""), "");
  assert.equal(normalizeSourceFilter(null), "");
  assert.equal(normalizeSourceFilter(undefined), "");
});
