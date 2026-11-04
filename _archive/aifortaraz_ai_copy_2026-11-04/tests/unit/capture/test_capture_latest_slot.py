"""LatestSlot: newest-item semantics, at-most-once delivery, drop counting, close and threads."""

from __future__ import annotations

import random
import threading
import time

import pytest

from safety_ai.capture.latest_slot import LatestSlot, SlotClosedError


def test_get_returns_the_item_once() -> None:
    slot: LatestSlot[int] = LatestSlot()
    assert slot.put(1) is False
    assert slot.pending is True
    assert slot.get(0.0) == 1
    assert slot.pending is False
    assert slot.get(0.0) is None
    assert slot.dropped == 0


def test_put_replaces_unread_item_and_counts_the_drop() -> None:
    slot: LatestSlot[str] = LatestSlot()
    assert slot.put("a") is False
    assert slot.put("b") is True
    assert slot.put("c") is True
    assert slot.dropped == 2
    assert slot.get(0.0) == "c"
    assert slot.put("d") is False
    assert slot.dropped == 2


def test_get_times_out_with_none() -> None:
    slot: LatestSlot[int] = LatestSlot()
    started = time.monotonic()
    assert slot.get(0.02) is None
    assert time.monotonic() - started >= 0.015


@pytest.mark.parametrize("timeout", [0.0, -1.0])
def test_non_positive_timeout_does_not_wait(timeout: float) -> None:
    slot: LatestSlot[int] = LatestSlot()
    started = time.monotonic()
    assert slot.get(timeout) is None
    assert time.monotonic() - started < 0.05


def test_none_items_are_delivered() -> None:
    slot: LatestSlot[None] = LatestSlot()
    slot.put(None)
    assert slot.pending is True
    assert slot.get(0.0) is None
    assert slot.pending is False


def test_get_wakes_up_on_put_from_another_thread() -> None:
    slot: LatestSlot[int] = LatestSlot()
    timer = threading.Timer(0.02, slot.put, args=(7,))
    timer.start()
    try:
        assert slot.get(None) == 7
    finally:
        timer.cancel()
        timer.join()


def test_close_wakes_waiting_readers() -> None:
    slot: LatestSlot[int] = LatestSlot()
    results: list[int | None] = []
    readers = [
        threading.Thread(target=lambda: results.append(slot.get(5.0)), daemon=True)
        for _ in range(3)
    ]
    for reader in readers:
        reader.start()
    time.sleep(0.02)
    started = time.monotonic()
    slot.close()
    for reader in readers:
        reader.join(2.0)
    assert time.monotonic() - started < 1.0
    assert results == [None, None, None]
    assert slot.closed is True


def test_close_keeps_the_last_unread_item_and_rejects_puts() -> None:
    slot: LatestSlot[int] = LatestSlot()
    slot.put(1)
    slot.close()
    slot.close()
    with pytest.raises(SlotClosedError):
        slot.put(2)
    assert slot.get(1.0) == 1
    started = time.monotonic()
    assert slot.get(1.0) is None
    assert slot.get(None) is None
    assert time.monotonic() - started < 0.05


def test_random_operation_sequences_match_a_model() -> None:
    rng = random.Random(20261104)
    for _ in range(200):
        slot: LatestSlot[int] = LatestSlot()
        model: int | None = None
        dropped = 0
        for step in range(rng.randint(1, 40)):
            if rng.random() < 0.6:
                replaced = slot.put(step)
                assert replaced is (model is not None)
                dropped += int(model is not None)
                model = step
            else:
                assert slot.get(0.0) == model
                model = None
            assert slot.dropped == dropped
            assert slot.pending is (model is not None)


def test_concurrent_producer_and_consumer() -> None:
    slot: LatestSlot[int] = LatestSlot()
    total = 5000
    received: list[int] = []
    done = threading.Event()

    def produce() -> None:
        for item in range(total):
            slot.put(item)
        done.set()

    def consume() -> None:
        while True:
            item = slot.get(0.05)
            if item is not None:
                received.append(item)
                if item == total - 1:
                    return
            elif done.is_set() and not slot.pending:
                return

    consumer = threading.Thread(target=consume, daemon=True)
    producer = threading.Thread(target=produce, daemon=True)
    consumer.start()
    producer.start()
    producer.join(10.0)
    consumer.join(10.0)
    assert not producer.is_alive() and not consumer.is_alive()
    assert received == sorted(set(received)), "items must arrive at most once and in order"
    assert received[-1] == total - 1, "the newest item is never lost"
    assert len(received) + slot.dropped == total


def test_concurrent_readers_never_share_an_item() -> None:
    slot: LatestSlot[int] = LatestSlot()
    total = 2000
    seen: list[int] = []
    lock = threading.Lock()

    def consume() -> None:
        while True:
            item = slot.get(None)
            if item is None:
                return
            with lock:
                seen.append(item)

    readers = [threading.Thread(target=consume, daemon=True) for _ in range(4)]
    for reader in readers:
        reader.start()
    for item in range(total):
        slot.put(item)
    slot.close()
    for reader in readers:
        reader.join(5.0)
    assert not any(reader.is_alive() for reader in readers)
    assert len(seen) == len(set(seen))
    assert len(seen) + slot.dropped == total
