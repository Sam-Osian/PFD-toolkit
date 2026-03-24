"""Server-Sent Events helpers for long-running AI workflows.

Provides a thread-safe bridge that lets a background worker thread push
structured progress events to a Django ``StreamingHttpResponse``.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Callable, Generator, Optional

from django.http import StreamingHttpResponse


_SENTINEL = object()

# Keepalive interval (seconds).  SSE comments are invisible to the
# EventSource API but prevent reverse-proxies from closing idle
# connections.
_KEEPALIVE_INTERVAL = 15


def _format_sse(event: str, data: Any) -> str:
    """Format a single SSE frame."""
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


class SSEProgressBridge:
    """Thread-safe bridge between a worker thread and an SSE generator.

    The worker calls :meth:`stage`, :meth:`progress`, :meth:`complete`, or
    :meth:`error` to emit events.  The SSE generator (running in the
    request thread) reads from the internal queue and yields formatted
    SSE frames.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[str | object] = queue.Queue()
        self._closed = threading.Event()

    # ------------------------------------------------------------------
    # Worker-side API
    # ------------------------------------------------------------------

    def stage(self, stage: str, message: str) -> None:
        """Signal a new high-level stage (e.g. 'Summarising reports')."""
        self._put("stage", {"stage": stage, "message": message})

    def progress(self, completed: int, total: int, description: str) -> None:
        """Report item-level progress within a stage."""
        percent = round(completed / total * 100) if total else 0
        self._put(
            "progress",
            {
                "completed": completed,
                "total": total,
                "description": description,
                "percent": percent,
            },
        )

    def progress_callback(self, completed: int, total: int, description: str) -> None:
        """Callback compatible with ``LLM.generate(progress_callback=...)``."""
        self.progress(completed, total, description)

    def complete(self, *, redirect: str, message: str = "") -> None:
        """Signal successful completion."""
        self._put("complete", {"redirect": redirect, "message": message})
        self.close()

    def error(self, message: str) -> None:
        """Signal an error."""
        self._put("error", {"message": message})
        self.close()

    def close(self) -> None:
        """Put the sentinel so the generator stops."""
        if not self._closed.is_set():
            self._closed.set()
            self._queue.put(_SENTINEL)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _put(self, event: str, data: Any) -> None:
        if not self._closed.is_set():
            self._queue.put(_format_sse(event, data))


def _sse_event_stream(bridge: SSEProgressBridge) -> Generator[str, None, None]:
    """Yield SSE frames from *bridge* until it is closed."""
    while True:
        try:
            item = bridge._queue.get(timeout=_KEEPALIVE_INTERVAL)
        except queue.Empty:
            # Send a keepalive comment to prevent proxy timeout.
            yield ": keepalive\n\n"
            continue

        if item is _SENTINEL:
            return
        yield item  # type: ignore[misc]


def sse_response(bridge: SSEProgressBridge) -> StreamingHttpResponse:
    """Return a ``StreamingHttpResponse`` wired to *bridge*."""
    response = StreamingHttpResponse(
        _sse_event_stream(bridge),
        content_type="text/event-stream",
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


def run_in_background(
    bridge: SSEProgressBridge,
    target: Callable[..., None],
    *,
    args: tuple = (),
    kwargs: Optional[dict[str, Any]] = None,
) -> threading.Thread:
    """Run *target* in a daemon thread; close *bridge* on exit."""

    def _wrapper() -> None:
        try:
            target(*args, **(kwargs or {}))
        except Exception as exc:
            bridge.error(str(exc))
        finally:
            bridge.close()

    thread = threading.Thread(target=_wrapper, daemon=True)
    thread.start()
    return thread
