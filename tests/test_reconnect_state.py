"""Property-based and unit tests for the resilient reconnection loop.

``btc_stochastic_monitor.run_forever(state)`` is the persistent ``while True``
reconnection loop. Each iteration:

1. Computes ``now_ms = int(time.time() * 1000)`` and calls
   ``check_buffer_freshness(state, now_ms)``. When the buffer is *stale* it
   discards ``state.buffer`` (``[]``), resets ``state.is_oversold = False``,
   logs a RECONNECT WARNING, and re-runs ``bootstrap_history(state)`` before
   opening a new session (REQ-8.5).
2. Constructs a module-level ``WebSocketApp`` bound to ``state`` and calls
   ``ws.run_forever()`` inside a ``try/except`` (REQ-8.1, REQ-8.2).
3. On any close / exception, increments a reconnect counter, logs a RECONNECT
   record, and ``time.sleep(RECONNECT_DELAY_S)`` before looping (REQ-8.3).

This file hosts:

* **Property 8 (task 12.3): State preservation across reconnects with
  stale-buffer recovery.** **Validates: Requirements 8.3, 8.4, 8.5.**

Test strategy (documented per the task notes)
---------------------------------------------
``run_forever`` is an INFINITE loop, so the loop is stopped deterministically
using **strategy (A)**: ``time.sleep`` is patched with a recorder that appends
each ``seconds`` argument and, after the configured number of disconnects,
raises a :class:`StopLoop` sentinel (a ``BaseException`` subclass, so the
loop's inner ``except Exception`` cannot swallow it, and the ``sleep`` call sits
outside that ``try`` block anyway). The test catches ``StopLoop`` around the
single ``run_forever(state)`` call.

Because ``run_forever`` constructs a *new* ``WebSocketApp`` every iteration, a
fake factory (:class:`_FakeSessionFactory`) hands back one fake session per
iteration. Each fake session models **one session + one disconnect**: at the
moment ``run_forever`` begins it snapshots ``state`` ("immediately after the new
session opens"), invokes the real (``functools.partial``-bound) ``on_open``,
drains that session's queued **closed-candle** payloads through the real
``on_message`` -> ``process_data`` (mutating the buffer / ``is_oversold``),
snapshots ``state`` again at the end of the session, advances the controlled
clock for the next iteration, and finally fires ``on_close(ws, 1006,
"abnormal")``.

Freshness is made deterministic by controlling ``btc_stochastic_monitor.time``:
``time.time`` returns a value the fake session advances at each session
boundary, chosen per iteration to be either clearly *fresh* (within one day of
the newest candle) or clearly *stale* (five days past it). ``bootstrap_history``
is patched to install a known "freshly-bootstrapped" buffer without any network
I/O, so the stale-recovery branch is observable.

The invariant verified for every iteration ``i`` (using an *independent* inline
re-derivation of the freshness predicate as the oracle):

* **fresh** -> the buffer and ``is_oversold`` captured when session ``i`` opens
  equal the values captured at the end of session ``i-1`` (preserved across the
  reconnect, REQ-8.4); for the first iteration they equal the initial state.
* **stale** -> the buffer equals the freshly-bootstrapped buffer the patched
  ``bootstrap_history`` installs and ``is_oversold`` equals ``False`` (REQ-8.5).

Plus: ``time.sleep`` is called exactly once per disconnect, each with
``RECONNECT_DELAY_S`` (== 5) (REQ-8.3).
"""

from __future__ import annotations

import json
import logging
from unittest import mock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule

import btc_stochastic_monitor as bsm
from btc_stochastic_monitor import Candle, MonitorState


# ---------------------------------------------------------------------------
# Loop-stopping sentinel (strategy A)
# ---------------------------------------------------------------------------
class StopLoop(BaseException):
    """Sentinel raised by the patched ``time.sleep`` to break the infinite loop.

    Subclasses ``BaseException`` (not ``Exception``) so the reconnection loop's
    inner ``except Exception`` around ``WebSocketApp.run_forever`` can never
    accidentally swallow it. In ``run_forever`` the ``time.sleep`` call is also
    outside that ``try`` block, so the sentinel propagates straight out of
    ``run_forever`` to the test.
    """


# ---------------------------------------------------------------------------
# Deterministic time anchors (all multiples of 1000 ms so that
# ``int(time.time() * 1000)`` round-trips exactly, with time.time() == ms/1000)
# ---------------------------------------------------------------------------
T0 = 1_700_000_000_000  # base epoch ms for all candle open times
_MIN_MS = 60_000  # one minute in ms; candle spacing
THRESHOLD = bsm.STALE_BUFFER_THRESHOLD_MS  # 86_400_000 (1 day)

# A "fresh" wall clock: one hour past T0. The newest candle is always within a
# few minutes of T0, so ``now - newest_open`` stays well under one day -> fresh.
NOW_FRESH = T0 + 3_600_000
# A "stale" wall clock: five days past T0 -> ``now - newest_open`` exceeds one
# day for every candle in play -> stale (forces re-bootstrap, REQ-8.5).
NOW_STALE = T0 + 5 * THRESHOLD


def _mk_candle(open_time_ms: int, volume: float) -> Candle:
    """Build a Candle with a distinguishing ``volume`` marker."""
    return Candle(
        open_time_ms=open_time_ms,
        open=100.0,
        high=110.0,
        low=90.0,
        close=105.0,
        volume=volume,
    )


# Initial buffer the Monitor starts with (volume marker 100.0). Newest open == T0.
INITIAL_BUFFER = [_mk_candle(T0 - (11 - i) * _MIN_MS, 100.0) for i in range(12)]

# The buffer the patched ``bootstrap_history`` installs on stale recovery
# (volume marker 200.0, so it is unequal to INITIAL_BUFFER and the stale branch
# is observable). Newest open == T0.
FRESH_BOOTSTRAP_BUFFER = [_mk_candle(T0 - (9 - i) * _MIN_MS, 200.0) for i in range(10)]


# ---------------------------------------------------------------------------
# Logging suppression (keeps the controlled clock free of logging's time.time()
# calls and keeps stderr quiet across many Hypothesis examples)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _suppress_monitor_logging():
    """Silence the Monitor's Logger for the duration of each test.

    Raises the named logger's level above CRITICAL so ``logger.info/warning/
    error`` short-circuit before constructing a ``LogRecord`` (and therefore
    never call ``time.time()`` on the patched clock), and attaches a
    ``NullHandler`` so the "no handlers" last-resort path stays quiet.
    """
    logger = logging.getLogger(bsm.LOGGER_NAME)
    old_level = logger.level
    old_propagate = logger.propagate
    logger.setLevel(logging.CRITICAL + 10)
    logger.propagate = False
    null_handler = logging.NullHandler()
    logger.addHandler(null_handler)
    try:
        yield
    finally:
        logger.removeHandler(null_handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate


# ---------------------------------------------------------------------------
# Closed-candle payload builder (Binance kline frame with "x": true)
# ---------------------------------------------------------------------------
def _closed_kline_payload(
    open_time_ms: int, o: float, h: float, low: float, c: float, v: float
) -> str:
    """Serialize a closed (``"x": true``) Binance kline frame for ``on_message``."""
    return json.dumps(
        {
            "e": "kline",
            "k": {
                "t": open_time_ms,
                "o": o,
                "h": h,
                "l": low,
                "c": c,
                "v": v,
                "x": True,
            },
        }
    )


# ---------------------------------------------------------------------------
# Scenario execution harness
# ---------------------------------------------------------------------------
# A "session spec" is (fresh: bool, ohlcv: list[(o, h, l, c, v)]):
#   * fresh -> the wall clock at the TOP of that iteration is NOW_FRESH
#     (preserve branch), otherwise NOW_STALE (re-bootstrap branch).
#   * ohlcv -> the closed candles pushed through that session, each appended
#     with a globally strictly-increasing open time so update_buffer appends it.


def _execute_scenario(sessions, *, initial_oversold: bool = False):
    """Drive ``run_forever`` once over the given session specs; collect snapshots.

    Returns a dict with the captured ``open_snaps`` / ``close_snaps`` (each a
    list of ``(buffer_list, is_oversold)`` tuples), the recorded ``sleep_args``,
    the per-iteration ``now_for_iter`` values, and the ``initial_buffer`` /
    ``initial_oversold`` the loop started from. Performs no property assertions
    itself (the caller does), so unit tests can inspect branch-specific outcomes.
    """
    num = len(sessions)
    assert num >= 1

    # Wall clock per iteration (ms): fresh or stale.
    now_for_iter = [NOW_FRESH if fresh else NOW_STALE for (fresh, _ohlcv) in sessions]

    # Assign globally strictly-increasing open times so every pushed candle is
    # newer than the buffer's current max and therefore appended (REQ-4.1).
    payloads_per_session: list[list[str]] = []
    global_idx = 0
    for _fresh, ohlcv in sessions:
        msgs: list[str] = []
        for (o, h, low, c, v) in ohlcv:
            open_time_ms = T0 + _MIN_MS * (global_idx + 1)
            global_idx += 1
            msgs.append(_closed_kline_payload(open_time_ms, o, h, low, c, v))
        payloads_per_session.append(msgs)

    # The single shared MonitorState the loop mutates across sessions.
    state = MonitorState(buffer=list(INITIAL_BUFFER), is_oversold=initial_oversold)

    # Controlled clock: time.time() returns the current value WITHOUT consuming
    # it (so incidental calls are harmless); the fake session advances it at
    # each session boundary to the next iteration's value.
    clock = {"now_ms": now_for_iter[0]}

    def fake_time() -> float:
        return clock["now_ms"] / 1000.0

    # Recording sleep that stops the infinite loop after ``num`` disconnects.
    sleep_args: list[float] = []

    def fake_sleep(seconds) -> None:
        sleep_args.append(seconds)
        if len(sleep_args) >= num:
            raise StopLoop

    # Patched bootstrap installs a known buffer without network I/O (REQ-8.5).
    def fake_bootstrap(st_: MonitorState) -> None:
        st_.buffer = list(FRESH_BOOTSTRAP_BUFFER)

    open_snaps: list[tuple[list[Candle], bool]] = []
    close_snaps: list[tuple[list[Candle], bool]] = []
    session_counter = {"i": 0}

    class _FakeSession:
        """One fake WebSocket session = one open + drain + abnormal close."""

        def __init__(self, url=None, on_open=None, on_message=None, on_close=None, **_kw):
            self.url = url
            self.on_open = on_open
            self.on_message = on_message
            self.on_close = on_close

        def run_forever(self, *_a, **_k):
            i = session_counter["i"]

            # "Immediately after the new session opens": snapshot BEFORE any
            # message mutates the buffer. list(...) copies the list; Candle is
            # frozen so the contents are effectively immutable.
            open_snaps.append((list(state.buffer), state.is_oversold))

            # Real partial-bound on_open(state, ws) -- logging only.
            if self.on_open is not None:
                self.on_open(self)

            # Drain this session's closed candles through the real on_message ->
            # process_data path, mutating state.buffer / state.is_oversold.
            for raw in payloads_per_session[i]:
                if self.on_message is not None:
                    self.on_message(self, raw)

            # End-of-session ("immediately before the disconnect") snapshot.
            close_snaps.append((list(state.buffer), state.is_oversold))

            # Advance the controlled clock for the NEXT iteration's freshness
            # check before the disconnect fires.
            if i + 1 < num:
                clock["now_ms"] = now_for_iter[i + 1]
            session_counter["i"] += 1

            # Force the abnormal close (code 1006) the task requires.
            if self.on_close is not None:
                self.on_close(self, 1006, "abnormal")

        def close(self, *_a, **_k):  # pragma: no cover - API parity only
            pass

    with mock.patch.object(bsm, "WebSocketApp", _FakeSession), mock.patch.object(
        bsm.time, "sleep", fake_sleep
    ), mock.patch.object(bsm.time, "time", fake_time), mock.patch.object(
        bsm, "bootstrap_history", fake_bootstrap
    ), mock.patch.object(
        bsm, "send_telegram_notification", lambda _msg: True
    ):
        try:
            bsm.run_forever(state)
        except StopLoop:
            pass

    return {
        "state": state,
        "open_snaps": open_snaps,
        "close_snaps": close_snaps,
        "sleep_args": sleep_args,
        "now_for_iter": now_for_iter,
        "initial_buffer": list(INITIAL_BUFFER),
        "initial_oversold": initial_oversold,
        "num": num,
    }


def _assert_property8(result) -> None:
    """Assert the full Property 8 invariant over an executed scenario."""
    num = result["num"]
    open_snaps = result["open_snaps"]
    close_snaps = result["close_snaps"]
    sleep_args = result["sleep_args"]
    now_for_iter = result["now_for_iter"]

    # --- REQ-8.3: exactly one sleep(5) per disconnect -----------------------
    assert len(sleep_args) == num, (
        f"expected {num} reconnect sleeps (one per disconnect), got {len(sleep_args)}"
    )
    assert all(s == bsm.RECONNECT_DELAY_S for s in sleep_args), (
        f"every reconnect sleep must be RECONNECT_DELAY_S; got {sleep_args}"
    )
    assert bsm.RECONNECT_DELAY_S == 5  # pin the configured 5-second delay

    # One session opened and one disconnect observed per iteration.
    assert len(open_snaps) == num
    assert len(close_snaps) == num

    # --- REQ-8.4 / REQ-8.5: per-iteration state handling --------------------
    for i in range(num):
        # State at the TOP of iteration i (before the freshness check):
        #   i == 0  -> the initial state the loop started with;
        #   i >= 1  -> the state at the end of the previous session (unchanged
        #              through the reconnect sleep).
        if i == 0:
            buf_top = result["initial_buffer"]
            ov_top = result["initial_oversold"]
        else:
            buf_top, ov_top = close_snaps[i - 1]

        now_ms = now_for_iter[i]

        # Independent oracle: re-derive the freshness predicate inline rather
        # than calling check_buffer_freshness, so the test does not assume the
        # function under orchestration is correct.
        expected_fresh = bool(buf_top) and (
            now_ms - buf_top[-1].open_time_ms
        ) <= THRESHOLD

        open_buf, open_ov = open_snaps[i]

        if expected_fresh:
            # Preserved across the reconnect (REQ-8.4).
            assert open_buf == buf_top, (
                f"iteration {i}: fresh buffer must be preserved across reconnect"
            )
            assert open_ov == ov_top, (
                f"iteration {i}: fresh is_oversold must be preserved across reconnect"
            )
        else:
            # Discarded + re-bootstrapped, cool-down reset (REQ-8.5).
            assert open_buf == FRESH_BOOTSTRAP_BUFFER, (
                f"iteration {i}: stale buffer must be replaced by the "
                f"freshly-bootstrapped buffer"
            )
            assert open_ov is False, (
                f"iteration {i}: stale recovery must reset is_oversold to False"
            )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
_price = st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)
_ohlcv = st.tuples(_price, _price, _price, _price, _price)
# A session: (fresh?, up to 3 closed candles).
_session = st.tuples(st.booleans(), st.lists(_ohlcv, max_size=3))


# ---------------------------------------------------------------------------
# Property 8 (primary): RuleBasedStateMachine over N disconnect cycles
# Validates: Requirements 8.3, 8.4, 8.5
# ---------------------------------------------------------------------------
class ReconnectStateMachine(RuleBasedStateMachine):
    """**Property 8: State preservation across reconnects with stale-buffer recovery**

    **Validates: Requirements 8.3, 8.4, 8.5**

    Each ``add_session`` rule appends one disconnect cycle (a freshness choice
    plus zero or more closed candles to push) to the scenario plan. On
    ``teardown`` the accumulated plan is executed against ``run_forever`` once
    via the fake-WebSocket factory described in the module docstring, forcing an
    ``on_close(code=1006, reason="abnormal")`` per cycle, and the full Property 8
    invariant is asserted:

    * ``time.sleep(5)`` is invoked exactly once per disconnect (REQ-8.3);
    * when the buffer is fresh, the buffer and ``is_oversold`` immediately after
      the new session opens equal their pre-disconnect values (REQ-8.4);
    * when the buffer is stale, the buffer equals the freshly-bootstrapped
      buffer and ``is_oversold`` equals ``False`` (REQ-8.5).
    """

    def __init__(self) -> None:
        super().__init__()
        self.plan: list[tuple[bool, list]] = []

    @rule(session=_session)
    def add_session(self, session) -> None:
        # Cap the plan so each example stays bounded and fast.
        if len(self.plan) < 6:
            self.plan.append(session)

    def teardown(self) -> None:
        if not self.plan:
            return
        result = _execute_scenario(self.plan)
        _assert_property8(result)


ReconnectStateMachine.TestCase.settings = settings(max_examples=100, deadline=None)
TestReconnectStateMachine = ReconnectStateMachine.TestCase


# ---------------------------------------------------------------------------
# Property 8 (auxiliary): @given form over a generated scenario
# Validates: Requirements 8.3, 8.4, 8.5
# ---------------------------------------------------------------------------
@given(sessions=st.lists(_session, min_size=1, max_size=5))
@settings(max_examples=100, deadline=None)
def test_property8_state_preservation_across_reconnects(sessions) -> None:
    """**Property 8: State preservation across reconnects with stale-buffer recovery**

    **Validates: Requirements 8.3, 8.4, 8.5**

    For any sequence of ``N >= 1`` disconnect cycles (each pushing closed
    candles then closing abnormally with code 1006), the reconnection loop
    sleeps ``RECONNECT_DELAY_S`` exactly ``N`` times and, immediately after each
    new session opens, either preserves the buffer / ``is_oversold`` (fresh) or
    installs the freshly-bootstrapped buffer with ``is_oversold == False``
    (stale).
    """
    result = _execute_scenario(sessions)
    _assert_property8(result)


# ---------------------------------------------------------------------------
# Targeted unit tests for the fresh and stale branches
# ---------------------------------------------------------------------------
def test_fresh_buffer_preserved_across_reconnect() -> None:
    """Fresh buffer + is_oversold survive a 1006 disconnect unchanged (REQ-8.4).

    Two fresh sessions: session 0 pushes closed candles (mutating the buffer);
    session 1 (also fresh) must open with exactly the buffer and ``is_oversold``
    that session 0 ended with -- i.e. the reconnect preserved them.
    """
    sessions = [
        (True, [(50.0, 60.0, 40.0, 45.0, 10.0), (52.0, 62.0, 42.0, 47.0, 11.0)]),
        (True, [(53.0, 63.0, 43.0, 48.0, 12.0)]),
    ]
    result = _execute_scenario(sessions)
    _assert_property8(result)

    open_snaps = result["open_snaps"]
    close_snaps = result["close_snaps"]

    # The new session (1) opened with exactly the prior session's end state.
    assert open_snaps[1][0] == close_snaps[0][0], "buffer not preserved across reconnect"
    assert open_snaps[1][1] == close_snaps[0][1], "is_oversold not preserved across reconnect"
    # And the preserved buffer genuinely reflects session 0's pushed candles
    # (i.e. preservation is non-trivial, not an empty/bootstrap buffer).
    assert open_snaps[1][0] != FRESH_BOOTSTRAP_BUFFER
    assert len(open_snaps[1][0]) == len(INITIAL_BUFFER) + 2

    # Exactly two disconnects -> two 5-second sleeps.
    assert result["sleep_args"] == [5, 5]


def test_stale_buffer_triggers_rebootstrap_and_reset() -> None:
    """Stale buffer is discarded, re-bootstrapped, and is_oversold reset (REQ-8.5).

    The loop starts with ``is_oversold = True`` and a stale wall clock, so the
    first iteration must discard the buffer, reset the cool-down lock, and
    re-bootstrap before opening the session.
    """
    sessions = [(False, [])]  # one stale session, no candles
    result = _execute_scenario(sessions, initial_oversold=True)
    _assert_property8(result)

    open_buf, open_ov = result["open_snaps"][0]
    # Buffer replaced by the freshly-bootstrapped buffer, cool-down reset.
    assert open_buf == FRESH_BOOTSTRAP_BUFFER
    assert open_buf != INITIAL_BUFFER
    assert open_ov is False
    # Single disconnect -> single 5-second sleep.
    assert result["sleep_args"] == [5]


def test_one_sleep_of_five_per_disconnect() -> None:
    """Exactly one ``time.sleep(5)`` is issued per disconnect across N cycles (REQ-8.3)."""
    for n in (1, 2, 3, 5):
        sessions = [(True, []) for _ in range(n)]
        result = _execute_scenario(sessions)
        assert result["sleep_args"] == [5] * n, (
            f"{n} disconnects must yield {n} sleeps of 5; got {result['sleep_args']}"
        )


def test_fresh_then_stale_then_fresh_sequence() -> None:
    """A fresh -> stale -> fresh trajectory exercises both branches in one run.

    Iteration 0 (fresh): preserve the initial buffer.
    Iteration 1 (stale): discard + re-bootstrap, reset is_oversold.
    Iteration 2 (fresh): preserve the (bootstrapped + drained) buffer from
    iteration 1.
    """
    sessions = [
        (True, [(50.0, 60.0, 40.0, 45.0, 10.0)]),
        (False, [(51.0, 61.0, 41.0, 46.0, 11.0)]),
        (True, [(52.0, 62.0, 42.0, 47.0, 12.0)]),
    ]
    result = _execute_scenario(sessions)
    _assert_property8(result)

    open_snaps = result["open_snaps"]
    close_snaps = result["close_snaps"]

    # Iteration 0 fresh: opened with the initial buffer.
    assert open_snaps[0][0] == INITIAL_BUFFER
    # Iteration 1 stale: opened with the freshly-bootstrapped buffer.
    assert open_snaps[1][0] == FRESH_BOOTSTRAP_BUFFER
    # Iteration 2 fresh: opened with exactly what session 1 ended with.
    assert open_snaps[2][0] == close_snaps[1][0]
    assert result["sleep_args"] == [5, 5, 5]
