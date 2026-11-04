# Scenario files

A scenario is a YAML file that describes a synthetic camera over time: the frame size and rate,
the camera's zone configuration (and live changes of it), persons and vehicles moving through
keyframes with the PPE evidence each person shows and the name of their pose, and the event
stream the logic layer must produce (spec 8.3). Scenarios bypass the detector: the same file
drives

* `safety_ai.capture.fake_source.FakeSource`: synthetic frames (`fake:<scenario.yaml>` as a camera
  source, spec 5.1);
* `safety_ai.inference.fake.FakeDetector` and `FakePose`: the detections and keypoints of those
  frames;
* the scenario runner: configs over time and the expected events and absences.

Everything is deterministic: detections and keypoints depend only on the frame id, and the
runner advances a fake clock by one frame period per frame, so a run never depends on machine
speed. `safety_ai.capture.scenario.load_scenario(path)` validates a file and reports every problem
as one readable line (see [Validation errors](#validation-errors)).

## Complete example

Every field of the format appears below. The expected timings are illustrative: derive the
expectations of a real scenario from the rules in `docs/ARCHITECTURE.md` 4.1-4.4 (zone
hysteresis 2 in / 3 out, PPE smoothing, `confirm_frames`, cooldowns).

```yaml
name: example_ppe_and_forklift
description: >
  A worker walks into the press area without a helmet, puts one on, is hidden behind a machine
  for one second, and a forklift passes along the loading aisle; at 24 s the site drops the
  helmet requirement.
frame_size: [1280, 720]
fps: 10
duration_s: 30
start_utc: "2026-11-03T09:00:00Z"   # a Tuesday, 14:00 local time in Asia/Yekaterinburg
camera_id: cam-1
timezone: Asia/Yekaterinburg
config:
  thresholds: {confirm_frames: 8, cooldown_sec: 60, min_confidence: 0.5}
  ppe_types: [helmet, vest, glasses, gloves, mask, boots, earmuffs, harness, face_shield]
  zones:
    - id: zone-press
      name: Press area
      type: ppe
      polygon: [[0.30, 0.45], [0.98, 0.45], [0.98, 0.99], [0.30, 0.99]]
      rules:
        required_ppe: [helmet, vest]
        schedule:
          - {days: [1, 2, 3, 4, 5], from: "08:00", to: "20:00"}
    - id: zone-aisle
      name: Loading aisle
      type: vehicle
      polygon: [[0.00, 0.45], [0.30, 0.45], [0.30, 0.99], [0.00, 0.99]]
      rules:
        min_vehicle_distance_m: 3.0
        confirm_frames: 4
config_updates:
  - at_s: 24
    config:
      thresholds: {confirm_frames: 8, cooldown_sec: 60, min_confidence: 0.5}
      zones:
        - id: zone-press
          name: Press area
          type: ppe
          polygon: [[0.30, 0.45], [0.98, 0.45], [0.98, 0.99], [0.30, 0.99]]
          rules:
            required_ppe: [vest]
tracks:
  - id: 1
    label: person
    conf: 0.92
    keyframes:
      - t: 0
        feet: [0.40, 0.92]
        height: 0.55
        pose: walking
        ppe: {helmet: missing, vest: present}
      - {t: 3, feet: [0.60, 0.92], pose: standing}
      - {t: 10, ppe: {helmet: present}}
      - {t: 15, visible: false}
      - {t: 16, visible: true}
  - id: 7
    label: forklift
    conf: 0.88
    keyframes:
      - {t: 18, box: [0.00, 0.50, 0.20, 0.30]}
      - {t: 26, box: [0.10, 0.55, 0.20, 0.30]}
expected_events:
  - type: ppe_violation
    zone: zone-press
    track: 1
    start_s: 0.8
    start_tol_s: 0.5
    end_s: 11.8
    end_tol_s: 1.0
    details: {missing_ppe: [helmet], present_ppe: [vest]}
expected_absent:
  - {type: vehicle_proximity}
  - {type: zone_intrusion, zone: zone-press}
  - {type: fall_detected, track: 1}
```

## Top-level fields

| field | required | default | meaning |
|---|---|---|---|
| `name` | yes | | Scenario name (reports, frame caption). |
| `description` | no | `""` | Free text. |
| `frame_size` | yes | | `[w, h]` in pixels (each 1..4096). |
| `fps` | yes | | Frames per second (0 < fps <= 120). |
| `duration_s` | yes | | Length in seconds (> 0). |
| `start_utc` | yes | | ISO 8601 time with a timezone (`2026-11-03T14:00:00Z`); the UTC time of t = 0. Zone schedules are evaluated in `timezone` at `start_utc + t`. |
| `camera_id` | no | `cam-1` | Camera id (`[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}`); also the configs' `camera.id`. |
| `timezone` | no | `UTC` | IANA timezone of the camera (zone schedules only). |
| `config` | yes | | The camera config in effect from t = 0 (see below). |
| `config_updates` | no | `[]` | `{at_s, config}` entries in increasing `at_s` order, each before `duration_s`. |
| `tracks` | yes | | Persons and vehicles (may be empty). |
| `expected_events` | no | `[]` | Events the logic layer must produce. |
| `expected_absent` | no | `[]` | Events that must not start. |

Unknown fields are errors everywhere in the file.

## Time

Scenario time `t` is in seconds from `start_utc`. Frame `k` (`frame_id = k`, counted from 0)
shows `t = k / fps`; a scenario has `n_frames = ceil(duration_s * fps)` frames covering
`0 <= t < duration_s` (30 s at 10 fps: frames 0..299, the last at t = 29.9). Times written in
the file (`t`, `at_s`, `start_s`, `end_s`) are compared with frame times with a tolerance of
1e-9 s, so `at_s: 2.5` applies from frame 25 at 10 fps.

## `config` and `config_updates`

`config` has the structure of the website's config response for one camera
(`docs/CONTRACT.md` 4.4): `camera`, `frame_size_hint`, `thresholds`, `ppe_types`, `zones` (with
`rules` and `schedule`), `version`, `updated_at`; zones also accept `enabled` (D-015). Each entry
of `config_updates` replaces the whole config from its `at_s` on (a full replacement, not a
patch: repeat everything that stays the same). In both:

* `camera` may be left out; `camera.id` is always `camera_id` (a different explicit id is an
  error) and `camera.name`, `stream_url`, `stream_type` are kept when given;
* `frame_size_hint` defaults to `frame_size`;
* `version` defaults to 1 for `config` and to the previous version + 1 for every update;
* unknown fields and unknown PPE codes are errors (the service itself ignores them in backend
  responses, but in a test a typo such as `entry_forbiden` must not silently disable a rule);
* schedule times may be written unquoted (`from: 08:00` stays the string `"08:00"`).

The runner applies an update on the first frame whose time is `>= at_s`.
`Scenario.config_at(t)` returns the config in effect at `t` (the same object for every `t` of one
period, so `config_at(t) is not previous` detects an update); `Scenario.config_update_times` lists
the `at_s` values.

## Tracks

| field | required | default | meaning |
|---|---|---|---|
| `id` | yes | | Track id reported by the fake detector (unique over persons and vehicles; >= 0). |
| `label` | no | `person` | `person`, `forklift`, `vehicle`, `truck` or `car`. |
| `conf` | no | `0.9` | Detector confidence of the track (0..1); its PPE boxes get the same confidence. Persons below the camera's `min_confidence` are ignored by the rules (D-017). |
| `keyframes` | yes | | At least one keyframe, strictly increasing `t`, each `t <= duration_s`. |

### Keyframes

| field | meaning |
|---|---|
| `t` | Scenario time of the keyframe (>= 0). |
| `box` | `[x, y, w, h]`: normalized top-left corner and size, inside the frame (`x + w <= 1`, `y + h <= 1`). Vehicles always use `box`. |
| `feet` | Persons only: `[x, y]`, the normalized bottom-centre of the person box (the point zones test). |
| `height` | With `feet`: the person's standing height as a fraction of the frame height (0 < height <= 1). Required on the first `feet` keyframe; later ones keep the previous height when they leave it out. |
| `pose` | Persons only: `standing`, `walking`, `bending`, `kneeling`, `sitting`, `falling` or `lying`. |
| `ppe` | Persons only: `{code: present \| missing \| unknown}` for the PPE codes `helmet`, `vest`, `glasses`, `gloves`, `mask`, `boots`, `earmuffs`, `harness`, `face_shield`. |
| `visible` | `false`: the track is not detected (occlusion, track loss); `true`: detected again. |

Semantics:

* **Appearance.** A track does not exist before its first keyframe. After its last keyframe the
  last state holds until the end of the scenario (use `visible: false` to make it leave).
* **Positions interpolate.** The position (box, or feet point and height) interpolates linearly
  between the keyframes that set one; keyframes without a position only change the stepped
  values. The first keyframe must set a position, and all positioned keyframes of a track use
  the same kind (`box` or `feet`).
* **Everything else steps.** `pose` (default `standing`), `visible` (default `true`) and each
  PPE code (default `unknown`) take the value of the last keyframe at or before `t` that sets
  them; `ppe` changes only the codes it lists (`{t: 10, ppe: {helmet: present}}` keeps the vest).
* **Feet positions.** The person box comes from `logic.poses.box_for_pose(pose, feet,
  person_height_px = height * frame_h)`: the box of the *current* pose standing on the feet point,
  clamped to the frame. A person who switches to `lying` keeps the feet point and gets a wide,
  low box (aspect about 3.9). Nominal box sizes in units of the standing height (w x h):
  standing 0.25 x 1.0, walking 0.65 x 1.01, bending 0.52 x 0.70, kneeling 0.68 x 0.77, sitting
  0.47 x 0.77, falling 0.90 x 0.85, lying 1.16 x 0.30.
* **Box positions.** The box is used as written (interpolated); the pose still drives the
  keypoints, which are fitted into that box.
* **Not detected.** An invisible track, and a person whose box lies entirely outside the frame,
  produce no detection, no PPE boxes and no keypoints.
* **Walking** animates its gait with phase `t / 1 s` (one cycle of two steps per second).

## PPE evidence

The fake detector turns each person's PPE state at `t` into item and bare-part boxes:

| state | detections |
|---|---|
| `present` | the item's box(es) in the item's association region |
| `missing` | the bare counterpart's box(es) in its region, no item |
| `unknown` | nothing |

| code | item placement | bare counterpart (missing) |
|---|---|---|
| `helmet` | above the head centre | `head` |
| `glasses` | on the eyes | `face` |
| `mask` | below the nose | `face` |
| `face_shield` | over the face | `face` |
| `earmuffs` | on the ears | `ear` |
| `vest` | on the torso | none |
| `harness` | on the torso | none |
| `gloves` | one box on each wrist | `hands` (one per wrist) |
| `boots` | one box on each ankle | `foot` (one per ankle) |

Each box is anchored at the synthetic keypoints of the person's pose and pushed inside the
item's region (head: top 25 % of the person box, torso: 20-70 %, feet: bottom 15 %, hands: the
box) so its assignment score is 1, far above the thresholds (0.5; helmet and vest 0.6). Glove
and bare-hand boxes stay inside the 0.12 h wrist circles as well, so the hands evidence does
not depend on whether the logic layer has keypoints for the person. The logic layer therefore
sees exactly the written state as PRESENT / MISSING / UNKNOWN, with two consequences of the
association rules (spec 5.5):

1. **Vest and harness** have no bare counterpart: `missing` and `unknown` both render nothing,
   and the logic decides from the torso: MISSING when the person box is at least 120 px high and
   the torso region lies inside the frame (more than 1 px from every border), UNKNOWN otherwise.
   An `unknown` vest needs a small person or a person cut by the frame border.
2. **Glasses, mask and face shield share the bare face.** While one of them is `missing`, the
   bare face box makes every other face item that is not `present` read MISSING as well.

Keep persons whose PPE matters from overlapping: an item that lies fully inside the regions of
two persons goes to the lower track id (association tie rule).

## Poses and keypoints

`FakePose` returns `logic.poses.synthesize_pose(pose, box, phase=...)` for every requested track
that is a visible person at the frame's time: COCO-17 keypoints with confidence 0.9 inside the
person's scenario box (the box the fake detector reported). Torso angles from vertical: standing
2°, walking 5°, bending 70°, kneeling 8°, sitting 4°, falling 52°, lying 90°. A fall is written
as pose steps, for example `standing` -> `falling` (0.3-0.5 s) -> `lying`: the box aspect jumps
from 0.25 over 1.06 to 3.9 at the same feet point.

## Expected events

| field | required | meaning |
|---|---|---|
| `type` | yes | `ppe_violation`, `zone_intrusion`, `vehicle_proximity`, `fall_detected`, `person_immobile`, `overcrowding` or `custom`. |
| `zone` | yes | The zone id of the event, or `null` (fall/immobile outside every zone). Must exist in `config` or an update. |
| `track` | yes | The event's `track_id`: a person track (overcrowding: the lowest id inside; proximity: the person). |
| `start_s` | yes | Scenario time of the frame on which the event became ACTIVE. |
| `start_tol_s` | yes | Allowed deviation of the start (seconds, >= 0). |
| `end_s` | yes | Scenario time of the event's `ended_at` (the confirming frame; for track loss the last time the track was seen, D-028), or `null`: still active at the end, ended by shutdown. |
| `end_tol_s` | with `end_s` | Allowed deviation of the end (required when `end_s` is a number). |
| `details` | no | A subset of the event's `details` fields to compare (names are checked against the event type: `missing_ppe`, `present_ppe`, `duration_sec`, `distance_m`, `person_track_id`, `vehicle_track_id`, `pose_conf`, `immobile_sec`, `count`, `max_persons`, `title`, `boxes`, `frame_size`). Numbers match within 1e-6, lists of strings (PPE codes) as sets, everything else exactly. |

`ExpectedEvent.matches(type, zone_id, track_id)`, `start_ok(start_s)`, `end_ok(end_s | None)`
and `details_mismatches(details)` implement these comparisons for the runner.

## Expected absences

`expected_absent` entries are `{type, zone?, track?}`: no event of `type` may start during the
scenario. A given `zone` narrows the match to that zone (`zone: null` to events without a zone; a
left-out `zone` matches every zone); a given `track` narrows it to that person. Zones and tracks
must exist, as for expected events.

## Running a scenario

The runner drives a `FakeClock` itself and asks the source for each frame synchronously:

```python
from safety_ai.capture.fake_source import FakeSource
from safety_ai.capture.scenario import load_scenario
from safety_ai.clock import FakeClock
from safety_ai.inference.fake import FakeDetector, FakePose

scenario = load_scenario(path)
clock = FakeClock(start_utc=scenario.start_utc)
source = FakeSource(scenario.camera_id, path, clock, realtime=False)
detector, pose = FakeDetector(scenario), FakePose(scenario)
config = scenario.config_at(0.0)  # LogicEngine(..., config, timezone=scenario.timezone)
for index in range(scenario.n_frames):
    if index:
        clock.advance(scenario.frame_period_s)
    t = scenario.time_of(index)
    if scenario.config_at(t) is not config:
        config = scenario.config_at(t)  # engine.update_config(config, mono, utc)
    frame = source.frame_at(index)  # frame_id=index, captured_at=start_utc + index/fps
    batch = detector.detect(frame)  # pose.estimate(frame, {track: box}) on request
    ...  # feed the pipeline / logic engine
# finally engine.shutdown(clock.monotonic(), clock.now_utc()) ends the events with end_s: null
```

`frame_at` never starts the capture thread and does not change the source's state or counters.
`Scenario.seconds_at(utc)` converts an event's `ended_at` back to scenario seconds.

As a live camera source (`source: fake:tests/scenarios/<file>.yaml`, or
`FakeSource(..., realtime=True)`) the capture thread plays the scenario at its fps in wall time
and ends after the last frame; with `loop=True` it starts over while `frame_id` keeps increasing
(frame `k` shows `t = (k mod n_frames) / fps`; build `FakeDetector` and `FakePose` with
`loop=True` too). `FakeSource(..., realtime=False)` makes the next frame as soon as the previous
one was read (no pacing, no drops). The frames show a plain background, faint zone outlines, a
filled rectangle with a label per visible track and a caption with the scenario name and `t`.

## Validation errors

Every problem is reported as one line naming the file and the dotted path, all problems of a
file at once where possible:

```text
tests/scenarios/x.yaml: fps: Field required
tests/scenarios/x.yaml: tracks.0.keyframes.2.t: must be greater than the previous keyframe's t (4)
tests/scenarios/x.yaml: tracks.1.keyframes.0.pose: is allowed for persons only
tests/scenarios/x.yaml: config.zones.0.rules.entry_forbiden: unknown field
tests/scenarios/x.yaml: config.zones.0.rules.required_ppe.1: unknown PPE code 'helmett'; expected one of: helmet, vest, glasses, gloves, mask, boots, earmuffs, harness, face_shield
tests/scenarios/x.yaml: expected_events.0.zone: unknown zone 'zone-a' (zones: zone-aisle, zone-press)
tests/scenarios/x.yaml: expected_events.0.end_tol_s: is required when end_s is set
```

Cross-checks between sections (duplicate track ids, unknown zones and tracks in expectations,
times after the end) run once every section is valid on its own. Opening an invalid file as a
camera source (`fake:<file>`) raises the same `ScenarioError`, a `SourceError`.
