"""Shared pytest fixtures and test doubles for the BTC Stochastic DCA Monitor suite.

This module provides:

* :class:`FakeWebSocketApp` -- a hand-rolled stand-in for
  ``websocket.WebSocketApp`` (from the ``websocket-client`` package). It is
  queue-driven: tests enqueue raw payloads, and :meth:`FakeWebSocketApp.run_forever`
  drains the queue, invoking the registered ``on_message`` callback for each
  payload. It also supports injecting an ``on_close(code, reason)`` event so the
  reconnection-loop tests (Property 8 / REQ-8) can simulate abnormal disconnects.
* the ``frozen_time`` fixture, a thin wrapper over ``freezegun`` for deterministic
  time in reconnect-delay and timestamp tests.
* a registered ``hypothesis`` profile (``max_examples=100, deadline=None``) loaded
  for every test session so property tests run at the required iteration count
  without per-test boilerplate.

See the design document's "Testing Strategy" section for the full mapping of
fixtures to correctness properties.
"""

from __future__ import annotations

import collections
from typing import Any, Callable, Deque, Optional

import pytest


# ---------------------------------------------------------------------------
# Hypothesis profile registration (REQ minimum 100 iterations per property test)
# ---------------------------------------------------------------------------
# Registered at import time so it is available before any property test runs.
# Guarded so the suite can still be collected in environments where Hypothesis
# is not installed (the property tests themselves import it directly).
try:
    from hypothesis import settings as _hypothesis_settings

    _hypothesis_settings.register_profile(
        "btc_monitor",
        max_examples=100,
        deadline=None,
    )
    _hypothesis_settings.load_profile("btc_monitor")
except Exception:  # pragma: no cover - only hit when Hypothesis is absent
    # Hypothesis is a dev-only dependency (requirements-dev.txt). If it is not
    # installed, property tests will fail to import on their own; we avoid
    # breaking collection of the non-property tests here.
    pass


# ---------------------------------------------------------------------------
# Fake WebSocketApp (REQ-3, REQ-8) -- mimics websocket-client's WebSocketApp
# ---------------------------------------------------------------------------
class FakeWebSocketApp:
    """A queue-driven test double for ``websocket.WebSocketApp``.

    The real ``WebSocketApp`` opens a socket, calls ``on_open`` once, delivers
    inbound frames to ``on_message`` until the connection closes, and finally
    calls ``on_close(ws, close_status_code, close_msg)``. ``run_forever`` blocks
    for the lifetime of that session.

    This fake reproduces that contract deterministically:

    * Construct with the same keyword callbacks the Monitor registers
      (``on_open``, ``on_message``, ``on_close``). Extra keyword arguments
      accepted by the real class (``on_error``, ``header``, ``on_ping`` ...)
      are tolerated and ignored.
    * Tests enqueue raw payloads with :meth:`queue_message` (or pass an initial
      list of messages to the constructor).
    * :meth:`run_forever` invokes ``on_open`` once, drains every queued message
      through ``on_message``, then fires ``on_close`` with the injected close
      code and reason. It returns when the queue is exhausted (one "session").
    * :meth:`inject_close` sets the ``(code, reason)`` delivered to ``on_close``
      so reconnect tests can simulate abnormal disconnects (e.g. code 1006).

    Because the Monitor stores the candle buffer and ``is_oversold`` flag outside
    the session lifecycle, a single ``FakeWebSocketApp`` instance can be
    ``run_forever``-ed repeatedly to model successive reconnects, and
    ``run_forever_calls`` records how many sessions were opened.
    """

    def __init__(
        self,
        url: str | None = None,
        on_open: Optional[Callable[["FakeWebSocketApp"], Any]] = None,
        on_message: Optional[Callable[["FakeWebSocketApp", str], Any]] = None,
        on_close: Optional[
            Callable[["FakeWebSocketApp", Optional[int], Optional[str]], Any]
        ] = None,
        *,
        messages: Optional[list[str]] = None,
        **_ignored: Any,
    ) -> None:
        self.url = url
        self.on_open = on_open
        self.on_message = on_message
        self.on_close = on_close

        # Queue of raw payloads to deliver to ``on_message`` (REQ-3.3).
        self._queue: Deque[str] = collections.deque(messages or [])

        # Injectable close event delivered to ``on_close`` (REQ-8.6, REQ-8.7).
        self._close_code: Optional[int] = None
        self._close_reason: Optional[str] = None

        # Bookkeeping useful for reconnect assertions (Property 8).
        self.run_forever_calls: int = 0
        self.keep_running: bool = False

    # -- test-side controls -------------------------------------------------
    def queue_message(self, raw: str) -> None:
        """Enqueue a raw payload for delivery on the next ``run_forever`` drain."""
        self._queue.append(raw)

    def queue_messages(self, raws: list[str]) -> None:
        """Enqueue several raw payloads in order."""
        self._queue.extend(raws)

    def inject_close(self, code: Optional[int] = None, reason: Optional[str] = None) -> None:
        """Set the ``(code, reason)`` passed to ``on_close`` when the session ends.

        Lets reconnect tests simulate an abnormal disconnect such as
        ``inject_close(1006, "abnormal")`` (REQ-8.6).
        """
        self._close_code = code
        self._close_reason = reason

    # -- WebSocketApp-compatible surface ------------------------------------
    def run_forever(self, *_args: Any, **_kwargs: Any) -> None:
        """Run one fake session: open, drain queued messages, then close.

        Mirrors ``WebSocketApp.run_forever``: calls ``on_open`` once, forwards
        each queued payload to ``on_message``, and finally calls ``on_close``
        with the injected close code/reason. Returns when the queue empties,
        modelling a single connection lifetime.
        """
        self.run_forever_calls += 1
        self.keep_running = True

        if self.on_open is not None:
            self.on_open(self)

        # Drain the queue snapshot taken at session start so messages enqueued
        # by callbacks belong to a subsequent session, matching real framing.
        while self._queue:
            raw = self._queue.popleft()
            if self.on_message is not None:
                self.on_message(self, raw)

        self.keep_running = False
        if self.on_close is not None:
            self.on_close(self, self._close_code, self._close_reason)

    def close(self, *_args: Any, **_kwargs: Any) -> None:
        """No-op stop hook mirroring ``WebSocketApp.close`` for API parity."""
        self.keep_running = False


@pytest.fixture
def fake_ws_class() -> type[FakeWebSocketApp]:
    """Return the :class:`FakeWebSocketApp` type for patching ``WebSocketApp``."""
    return FakeWebSocketApp


@pytest.fixture
def fake_ws() -> FakeWebSocketApp:
    """Return a fresh, empty :class:`FakeWebSocketApp` instance."""
    return FakeWebSocketApp()


# ---------------------------------------------------------------------------
# Time control fixture (REQ-8.3 reconnect delay, REQ-11.5 UTC timestamps)
# ---------------------------------------------------------------------------
@pytest.fixture
def frozen_time():
    """Freeze wall-clock time for deterministic timestamp / delay assertions.

    Yields the ``freezegun`` ``frozen_time`` handle so a test can advance time
    with ``frozen_time.tick(...)``. ``freezegun`` is imported lazily so the
    fixture only requires the dependency when a test actually uses it.

    Example::

        def test_uses_time(frozen_time):
            frozen_time.tick(delta=datetime.timedelta(seconds=5))
    """
    from freezegun import freeze_time

    with freeze_time("2024-01-01T00:00:00Z") as frozen:
        yield frozen
