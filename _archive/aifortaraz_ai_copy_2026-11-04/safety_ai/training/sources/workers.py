"""A bounded thread-pool map used by the downloaders (never more than a few tasks queued)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TypeVar

__all__ = ["run_bounded"]

_T = TypeVar("_T")
_R = TypeVar("_R")


def run_bounded(
    func: Callable[[_T], _R],
    items: Iterable[_T],
    *,
    workers: int,
    on_result: Callable[[_T, _R | None, BaseException | None], None],
    max_in_flight: int | None = None,
) -> None:
    """Run ``func`` over ``items`` on a thread pool with a bounded number of pending tasks.

    ``on_result(item, result, error)`` is called in the calling thread for every item, in
    completion order; at most ``max_in_flight`` (default ``2 * workers``) tasks are queued, so a
    long item iterator never turns into an unbounded queue.
    """
    limit = max_in_flight or 2 * max(1, workers)
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="src") as pool:
        pending: dict[Future[_R], _T] = {}
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < limit:
                try:
                    item = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending[pool.submit(func, item)] = item
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                item = pending.pop(future)
                error = future.exception()
                on_result(item, None if error is not None else future.result(), error)
