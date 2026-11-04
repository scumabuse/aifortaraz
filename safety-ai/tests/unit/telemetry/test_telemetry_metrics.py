"""Tests of safety_ai.telemetry.metrics: names and labels, rendering and the delta round trip."""

from __future__ import annotations

import logging
import math
import pickle
import threading

import pytest
from prometheus_client.parser import text_string_to_metric_families

from safety_ai.telemetry.metrics import (
    MAX_PENDING_OBSERVATIONS,
    METRIC_SPECS,
    Metrics,
    MetricsDelta,
)

EXPECTED_FAMILIES = {
    # family name as parsed -> (type, label names); counters are exposed with a _total suffix
    "safety_ai_frames_received": ("counter", ("camera",)),
    "safety_ai_frames_processed": ("counter", ("camera",)),
    "safety_ai_frames_dropped": ("counter", ("camera",)),
    "safety_ai_detections": ("counter", ("camera", "label")),
    "safety_ai_events": ("counter", ("camera", "state", "type")),
    "safety_ai_confirmation_latency_seconds": ("histogram", ()),
    "safety_ai_backend_request_seconds": ("histogram", ("endpoint",)),
    "safety_ai_backend_failures": ("counter", ("endpoint", "reason")),
    "safety_ai_reconnects": ("counter", ("camera", "kind")),
    "safety_ai_spool_depth": ("gauge", ("camera",)),
    "safety_ai_processed_fps": ("gauge", ("camera",)),
    "safety_ai_active_events": ("gauge", ("camera",)),
}


def exercise(metrics: Metrics) -> None:
    metrics.frame_received("cam-1")
    metrics.frame_received("cam-1")
    metrics.frames_processed("cam-1", 2)
    metrics.frames_dropped("cam-1", 3)
    metrics.detection("cam-1", "person", 4)
    metrics.detection("cam-1", "helmet")
    metrics.event("cam-1", "ppe_violation", "started")
    metrics.event("cam-1", "ppe_violation", "ended")
    metrics.observe_confirmation_latency(0.3)
    metrics.observe_confirmation_latency(1.2)
    metrics.observe_backend_request("events", 0.02)
    metrics.backend_failure("events", "timeout")
    metrics.reconnect("cam-1", "stream")
    metrics.reconnect("", "ws")
    metrics.set_spool_depth("cam-1", 5)
    metrics.set_fps("cam-1", 11.5)
    metrics.set_active_events("cam-1", 1)


def samples(metrics: Metrics) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    text = metrics.render().decode("utf-8")
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
    }


def value(metrics: Metrics, name: str, **labels: str) -> float:
    return samples(metrics)[(name, tuple(sorted(labels.items())))]


# ------------------------------------------------------------------------------------ names


def test_render_has_exactly_the_architecture_families_and_labels():
    metrics = Metrics()
    exercise(metrics)
    text = metrics.render().decode("utf-8")

    families = {f.name: f for f in text_string_to_metric_families(text)}

    assert set(families) == set(EXPECTED_FAMILIES)
    for name, (kind, labels) in EXPECTED_FAMILIES.items():
        family = families[name]
        assert family.type == kind
        label_sets = {tuple(sorted(k for k in s.labels if k != "le")) for s in family.samples}
        assert label_sets == {labels}, name
    assert "_created" not in text
    assert "# TYPE safety_ai_frames_received_total counter" in text
    assert len(METRIC_SPECS) == 12


def test_rendered_values():
    metrics = Metrics()
    exercise(metrics)

    assert value(metrics, "safety_ai_frames_received_total", camera="cam-1") == 2
    assert value(metrics, "safety_ai_frames_processed_total", camera="cam-1") == 2
    assert value(metrics, "safety_ai_frames_dropped_total", camera="cam-1") == 3
    assert value(metrics, "safety_ai_detections_total", camera="cam-1", label="person") == 4
    assert (
        value(
            metrics, "safety_ai_events_total", camera="cam-1", type="ppe_violation", state="ended"
        )
        == 1
    )
    assert value(metrics, "safety_ai_confirmation_latency_seconds_count") == 2
    assert value(metrics, "safety_ai_confirmation_latency_seconds_sum") == pytest.approx(1.5)
    assert value(metrics, "safety_ai_confirmation_latency_seconds_bucket", le="1.0") == 1
    assert value(metrics, "safety_ai_confirmation_latency_seconds_bucket", le="1.5") == 2
    assert value(metrics, "safety_ai_backend_request_seconds_count", endpoint="events") == 1
    assert (
        value(metrics, "safety_ai_backend_failures_total", endpoint="events", reason="timeout") == 1
    )
    assert value(metrics, "safety_ai_reconnects_total", camera="", kind="ws") == 1
    assert value(metrics, "safety_ai_spool_depth", camera="cam-1") == 5
    assert value(metrics, "safety_ai_processed_fps", camera="cam-1") == 11.5
    assert value(metrics, "safety_ai_active_events", camera="cam-1") == 1


def test_register_camera_creates_zero_series():
    metrics = Metrics()
    metrics.register_camera("cam-7")

    assert value(metrics, "safety_ai_frames_received_total", camera="cam-7") == 0
    assert value(metrics, "safety_ai_reconnects_total", camera="cam-7", kind="stream") == 0
    assert value(metrics, "safety_ai_processed_fps", camera="cam-7") == 0
    assert metrics.collect_delta().empty


def test_instances_have_private_registries():
    first, second = Metrics(), Metrics()
    first.frame_received("cam-1")

    assert value(first, "safety_ai_frames_received_total", camera="cam-1") == 1
    assert ("safety_ai_frames_received_total", (("camera", "cam-1"),)) not in samples(second)
    assert first.registry is not second.registry


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.event("cam-1", "nonsense", "started"),
        lambda m: m.event("cam-1", "ppe_violation", "paused"),
        lambda m: m.reconnect("cam-1", "wifi"),
        lambda m: m.frames_dropped("cam-1", -1),
        lambda m: m.observe_backend_request("events", -0.1),
        lambda m: m.observe_confirmation_latency(math.nan),
    ],
)
def test_invalid_values_are_rejected(call):
    with pytest.raises(ValueError):
        call(Metrics())


# ------------------------------------------------------------------------------------ deltas


def test_delta_round_trip_through_pickle():
    worker = Metrics()
    exercise(worker)
    delta = pickle.loads(pickle.dumps(worker.collect_delta()))

    main = Metrics(track_deltas=False)
    main.apply(delta)

    expected = {key: v for key, v in samples(worker).items() if v != 0}
    actual = {key: v for key, v in samples(main).items() if v != 0}
    assert actual == expected
    assert main.collect_delta().empty


def test_collect_delta_resets_and_accumulates_per_series():
    metrics = Metrics()
    metrics.frame_received("cam-1")
    metrics.frame_received("cam-1")
    metrics.set_fps("cam-1", 5.0)
    metrics.set_fps("cam-1", 7.0)
    metrics.observe_backend_request("heartbeat", 0.1)
    metrics.observe_backend_request("heartbeat", 0.2)

    delta = metrics.collect_delta()

    assert delta.counters == (("frames_received", ("cam-1",), 2.0),)
    assert delta.gauges == (("processed_fps", ("cam-1",), 7.0),)
    assert delta.observations == (("backend_request", ("heartbeat",), (0.1, 0.2)),)
    assert metrics.collect_delta().empty
    metrics.frames_dropped("cam-1", 0)  # zero increments create the series, not a delta
    assert metrics.collect_delta().empty


def test_apply_accumulates_across_deltas_and_workers():
    main = Metrics(track_deltas=False)
    for camera in ("cam-1", "cam-2"):
        worker = Metrics()
        worker.frames_processed(camera, 10)
        main.apply(worker.collect_delta())
        worker.frames_processed(camera, 5)
        main.apply(worker.collect_delta())

    assert value(main, "safety_ai_frames_processed_total", camera="cam-1") == 15
    assert value(main, "safety_ai_frames_processed_total", camera="cam-2") == 15


def test_merge_combines_deltas_like_applying_both():
    worker = Metrics()
    worker.frame_received("cam-1")
    worker.set_spool_depth("cam-1", 3)
    worker.observe_confirmation_latency(0.5)
    first = worker.collect_delta()
    worker.frame_received("cam-1")
    worker.set_spool_depth("cam-1", 1)
    worker.observe_confirmation_latency(0.7)
    second = worker.collect_delta()

    merged = first.merge(second)
    separately, together = Metrics(track_deltas=False), Metrics(track_deltas=False)
    separately.apply(first)
    separately.apply(second)
    together.apply(merged)

    assert samples(together) == samples(separately)
    assert merged.counters == (("frames_received", ("cam-1",), 2.0),)
    assert merged.gauges == (("spool_depth", ("cam-1",), 1.0),)
    assert merged.observations == (("confirmation_latency", (), (0.5, 0.7)),)
    assert MetricsDelta().merge(MetricsDelta()).empty


def test_apply_skips_malformed_entries_with_a_warning(caplog):
    delta = MetricsDelta(
        counters=(("unknown", ("x",), 1.0), ("frames_received", ("a", "b"), 1.0)),
        gauges=(("frames_received", ("cam-1",), 1.0),),
        observations=(("confirmation_latency", (), (0.1,)),),
    )
    metrics = Metrics()

    with caplog.at_level(logging.WARNING):
        metrics.apply(delta)

    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 3
    assert value(metrics, "safety_ai_confirmation_latency_seconds_count") == 1


def test_track_deltas_false_keeps_no_pending_changes():
    metrics = Metrics(track_deltas=False)
    exercise(metrics)
    assert metrics.collect_delta().empty


def test_pending_observations_are_bounded():
    metrics = Metrics()
    for index in range(MAX_PENDING_OBSERVATIONS + 10):
        metrics.observe_confirmation_latency(float(index))

    ((_, _, values),) = metrics.collect_delta().observations

    assert len(values) == MAX_PENDING_OBSERVATIONS
    assert values[0] == 10.0  # the oldest observations were discarded


def test_helpers_are_thread_safe():
    metrics = Metrics()

    def work() -> None:
        for _ in range(1000):
            metrics.frame_received("cam-1")

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert value(metrics, "safety_ai_frames_received_total", camera="cam-1") == 4000
    assert metrics.collect_delta().counters == (("frames_received", ("cam-1",), 4000.0),)
