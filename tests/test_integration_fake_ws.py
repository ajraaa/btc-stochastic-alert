"""End-to-end integration tests for the BTC Stochastic DCA Monitor (task 14.1).

These tests wire the Monitor's real components together -- the centralized
Logger, the historical Bootstrapper, the WebSocket callbacks, the buffer /
indicator / trigger pipeline, the Telegram Notifier, and the resilient
reconnection loop -- with only the two true I/O boundaries faked:

* ``requests.get`` (Binance REST klines) and ``requests.post`` (Telegram
  ``sendMessage``) are mocked with the ``responses`` library so no network call
  leaves the process and every Telegram dispatch is captured.
* ``websocket.WebSocketApp`` is replaced by the queue-driven
  :class:`~tests.conftest.FakeWebSocketApp` (Test A) or a small fake session
  factory (Test B), so a *recorded* sequence of Binance kline frames can be
  streamed deterministically and an abnormal disconnect injected on demand.

Two integration scenarios are exercised (per the task's recommended structure):

**Test A -- end-to-end oversold episode + log-file categories.**
Validates: Requirements 2, 3, 4, 5, 6, 7, 10.4, 10.5, 10.6.
Startup emits a STARTUP record; ``bootstrap_history`` seeds 50 historical
candles; a recorded payload sequence (interleaving non-closed ``x=false`` and
closed ``x=true`` frames) is streamed through the real ``on_open`` /
``on_message`` / ``on_close`` callbacks via the fake WebSocket. The sequence
drives the smoothed %K strictly below 20 (one closed candle) and later strictly
above 20, producing exactly ONE oversold episode -> exactly ONE Telegram POST.
The captured message carries the symbol, interval, and two-decimal %K / %D; the
candle buffer stays bounded at 50; and the backup log FILE contains records
categorized ``STARTUP``, ``INDICATOR``, and ``NOTIFY``.

**Test B -- reconnect preserves state.**
Validates: Requirements 8.3, 8.4 (and the REQ-8.5 fresh branch).
A fresh state (recent candle open times) is driven through ``run_forever`` with
``WebSocketApp`` patched by a fake factory that, per session, snapshots the
state at session-open, drains a couple of closed candles, and then injects
``on_close(ws, 1006, "abnormal")``. ``time.sleep`` is patched to record each
delay and raise a sentinel after the requested number of disconnects, breaking
the otherwise-infinite loop. The test asserts the loop sleeps ``5`` seconds once
per disconnect and that ``state.buffer`` and ``state.is_oversold`` immediately
after each new session opens equal their pre-disconnect values (preserved), with
``bootstrap_history`` never invoked while the buffer is fresh.

Indicator math note (native pandas STOCH(5,3,3))
------------------------------------------------
``fast_k = 100 * (close - rolling_min_5(low)) / (rolling_max_5(high) -
rolling_min_5(low))`` and ``%K = SMA_3(fast_k)``. With every candle pinned to
``high=110, low=90`` the denominator is a constant 20, so ``fast_k = 5*(close -
90)``. Seeding 50 candles at ``close=100`` gives ``fast_k=50`` and a smoothed
``%K=50``. Streaming closed candles at ``close=91`` (``fast_k=5``) walks the
3-period SMA down 50 -> 35 -> 20 -> 5: the THIRD oversold close is the first to
drive ``%K`` strictly below 20 (FIRE). A subsequent ``close=109``
(``fast_k=95``) lifts the SMA back to 35 (> 20), releasing the cool-down lock.
"""

from __future__ import annotations

import functools
import json
import logging
import re
from unittest import mock

import pytest
import responses

import btc_stochastic_monitor as bsm
from btc_stochastic_monitor import Candle, MonitorState

from .conftest import FakeWebSocketApp

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
VALID_TOKEN = "123456:test-bot-token"
VALID_CHAT_ID = "987654321"
TELEGRAM_URL = f"https://api.telegram.org/bot{VALID_TOKEN}/sendMessage"

ONE_DAY_MS = 86_400_000
T0 = 1_700_000_000_000  # base epoch ms for the seeded historical candles

# Pinned OHLC band: high/low constant so the STOCH denominator is a constant 20
# and the close alone steers fast_k = 5 * (close - 90).
HIGH = 110.0
LOW = 90.0
SEED_CLOSE = 100.0  # fast_k = 50  -> smoothed %K = 50 after warm-up
OVERSOLD_CLOSE = 91.0  # fast_k = 5   -> walks the SMA toward 5
RECOVER_CLOSE = 109.0  # fast_k = 95  -> lifts the SMA back above 20


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------
def _kline_frame(open_time_ms: int, close: float, *, closed: bool) -> str:
    """Serialize a Binance kline WS frame with the given ``x`` (is-closed) flag."""
    return json.dumps(
        {
            "e": "kline",
            "E": open_time_ms,
            "s": "BTCUSDT",
            "k": {
                "t": open_time_ms,
                "o": SEED_CLOSE,
                "h": HIGH,
                "l": LOW,
                "c": close,
                "v": 1.0,
                "x": closed,
            },
        }
    )


def _seed_rows(count: int = 50) -> list[list]:
    """Build ``count`` Binance-style REST kline rows (12 elements each).

    ``bootstrap_history`` reads only indices 0..5 (openTime, OHLCV) and ignores
    the rest; values are strings to mirror Binance's real payload shape.
    """
    rows: list[list] = []
    for i in range(count):
        open_time = T0 + i * ONE_DAY_MS
        rows.append(
            [
                open_time,
                str(SEED_CLOSE),  # open
                str(HIGH),  # high
                str(LOW),  # low
                str(SEED_CLOSE),  # close
                "1.0",  # volume
                open_time + ONE_DAY_MS - 1,  # closeTime
                "0",
                0,
                "0",
                "0",
                "0",
            ]
        )
    return rows


def _flush_and_close_handlers(logger: logging.Logger) -> None:
    """Flush and detach all handlers so the backup log file is readable/closable.

    On Windows an open ``FileHandler`` keeps the temp log file locked, so the
    handlers are explicitly flushed, closed, and removed before the file is read
    and the surrounding ``tmp_path`` is torn down.
    """
    for handler in list(logger.handlers):
        try:
            handler.flush()
        finally:
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover - defensive cleanup only
                pass


# ===========================================================================
# Test A -- end-to-end oversold episode + log-file categories
# Validates: Requirements 2, 3, 4, 5, 6, 7, 10.4, 10.5, 10.6
# ===========================================================================
@responses.activate
def test_end_to_end_oversold_episode_and_log_categories(monkeypatch, tmp_path):
    """**Validates: Requirements 2, 3, 4, 5, 6, 7, 10.4, 10.5, 10.6**

    A full startup -> bootstrap -> live-stream -> notify -> log flow produces
    exactly one Telegram dispatch for a single oversold episode, with the
    expected two-decimal %K / %D message fields, a buffer bounded at 50, and a
    backup log file carrying STARTUP / INDICATOR / NOTIFY records.
    """
    # --- Valid credentials + a real, temporary backup log FILE --------------
    monkeypatch.setattr(bsm, "TELEGRAM_BOT_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(bsm, "TELEGRAM_CHAT_ID", VALID_CHAT_ID)
    log_path = str(tmp_path / "monitor.log")
    monkeypatch.setattr(bsm, "BACKUP_LOG_FILE", log_path)

    # --- Mock the two I/O boundaries: Binance REST GET + Telegram POST ------
    responses.add(responses.GET, bsm.BINANCE_REST_URL, json=_seed_rows(50), status=200)
    responses.add(responses.POST, TELEGRAM_URL, json={"ok": True}, status=200)

    # --- Startup: init the real Logger and emit the STARTUP record (as main) -
    state = MonitorState(buffer=[])
    state.logger = bsm.init_logger(log_path)
    try:
        state.logger.info(
            "Monitor startup complete; beginning bootstrap for %s @ %s.",
            bsm.SYMBOL,
            bsm.INTERVAL,
            extra={"category": "STARTUP"},
        )

        # --- Bootstrap seeds exactly 50 historical candles (REQ-2) ----------
        bsm.bootstrap_history(state)
        assert len(state.buffer) == 50, "bootstrap should seed the full 50-candle buffer"
        assert len(state.buffer) <= bsm.HISTORY_LIMIT
        # Chronological order preserved (REQ-2.2/REQ-4.3).
        opens = [c.open_time_ms for c in state.buffer]
        assert opens == sorted(opens)

        # --- Independent oracle for the FIRE candle's expected %K / %D ------
        # Mirror the buffer through the first three oversold closes (the third
        # is the FIRE) and recompute the reading from scratch.
        shadow = list(state.buffer)
        fire_open_time = 0
        for j in range(3):
            fire_open_time = T0 + (50 + j) * ONE_DAY_MS
            bsm.update_buffer(
                shadow,
                Candle(fire_open_time, SEED_CLOSE, HIGH, LOW, OVERSOLD_CLOSE, 1.0),
            )
        expected = bsm.compute_stochastic(shadow)
        assert expected is not None
        assert expected.k < bsm.OVERSOLD_THRESHOLD, "3rd oversold close must drive %K < 20"

        # --- Recorded payload sequence: interleave x=false and x=true frames -
        # Open times continue strictly after the seeded history so every closed
        # candle is appended (REQ-4.1). Non-closed (x=false) frames MUST be
        # ignored (REQ-3.4) and never reach the buffer / indicator.
        closed_closes = [
            OVERSOLD_CLOSE,  # A1: %K -> 35  (QUIET)
            OVERSOLD_CLOSE,  # A2: %K -> 20  (QUIET; boundary, strict < not met)
            OVERSOLD_CLOSE,  # A3: %K ->  5  (FIRE; first strictly < 20)
            OVERSOLD_CLOSE,  # A4: %K ->  5  (SUPPRESS; cool-down engaged)
            RECOVER_CLOSE,   # R1: %K -> 35  (RESET; cool-down released)
        ]
        payloads: list[str] = []
        for idx, close in enumerate(closed_closes):
            open_time = T0 + (50 + idx) * ONE_DAY_MS
            # A non-closed (x=false) frame precedes each closed candle and uses
            # the SAME open time to mimic mid-candle ticks; it must be ignored.
            payloads.append(_kline_frame(open_time, close, closed=False))
            payloads.append(_kline_frame(open_time, close, closed=True))

        # --- Stream the sequence through the real callbacks via the fake WS --
        ws = FakeWebSocketApp(
            bsm.BINANCE_WS_URL,
            on_open=functools.partial(bsm.on_open, state),
            on_message=functools.partial(bsm.on_message, state),
            on_close=functools.partial(bsm.on_close, state),
            messages=payloads,
        )
        ws.run_forever()  # open -> drain all frames -> close

        # --- REQ-7 / REQ-6: exactly ONE Telegram POST for the one episode ----
        telegram_posts = [
            call
            for call in responses.calls
            if call.request.method == "POST" and call.request.url == TELEGRAM_URL
        ]
        assert len(telegram_posts) == 1, (
            f"exactly one Telegram dispatch expected for a single oversold "
            f"episode, got {len(telegram_posts)}"
        )

        # --- REQ-7.2: message carries symbol, interval, 2-decimal %K / %D ----
        body = json.loads(telegram_posts[0].request.body)
        text = body["text"]
        assert body["chat_id"] == VALID_CHAT_ID
        assert bsm.SYMBOL in text
        assert bsm.INTERVAL in text
        # Two-decimal formatting present for both %K and %D.
        assert re.search(r"%K=\d+\.\d{2}\b", text), f"two-decimal %K missing: {text!r}"
        assert re.search(r"%D=\d+\.\d{2}\b", text), f"two-decimal %D missing: {text!r}"
        # And the exact recomputed values (independent oracle).
        assert f"%K={expected.k:.2f}" in text
        assert f"%D={expected.d:.2f}" in text
        assert f"close={OVERSOLD_CLOSE}" in text

        # --- REQ-4.2 / REQ-1.6: buffer remains bounded at 50 ----------------
        assert len(state.buffer) == 50, "buffer must stay bounded at HISTORY_LIMIT (50)"
        # The five closed candles are the newest entries (non-closed frames did
        # not mutate the buffer, REQ-3.4).
        newest = state.buffer[-1]
        assert newest.open_time_ms == T0 + (50 + len(closed_closes) - 1) * ONE_DAY_MS
        assert newest.close == RECOVER_CLOSE
    finally:
        _flush_and_close_handlers(state.logger)

    # --- REQ-11.3 / REQ-10.4..10.6: the backup log FILE has the categories ---
    with open(log_path, encoding="utf-8") as handle:
        log_text = handle.read()

    assert "[STARTUP]" in log_text, "log file must contain a STARTUP record"
    assert "[INDICATOR]" in log_text, "log file must contain INDICATOR records"
    assert "[NOTIFY]" in log_text, "log file must contain a NOTIFY record"

    # Exactly one INDICATOR record per CLOSED candle (5), proving non-closed
    # frames were filtered out before the indicator (REQ-3.4, REQ-10.4).
    indicator_lines = [line for line in log_text.splitlines() if "[INDICATOR]" in line]
    assert len(indicator_lines) == len(closed_closes), (
        f"expected one INDICATOR record per closed candle ({len(closed_closes)}), "
        f"got {len(indicator_lines)}"
    )
    # Exactly one NOTIFY record (the single successful dispatch, REQ-10.6).
    notify_lines = [line for line in log_text.splitlines() if "[NOTIFY]" in line]
    assert len(notify_lines) == 1
    assert "success" in notify_lines[0].lower()


# ===========================================================================
# Test B -- reconnect preserves state (REQ-8.3, REQ-8.4)
# ===========================================================================
class _StopLoop(BaseException):
    """Sentinel raised from the patched ``time.sleep`` to break the infinite loop.

    Subclasses ``BaseException`` so ``run_forever``'s inner ``except Exception``
    around the session cannot swallow it; the ``time.sleep`` call also sits
    outside that ``try`` block, so the sentinel propagates straight to the test.
    """


def _recent_oversold_buffer(now_ms: int, count: int = 12) -> list[Candle]:
    """Build ``count`` recent candles whose STOCH %K computes to 0 (oversold).

    Each candle uses ``high=100, low=0, close=0`` so over any window
    ``high_max=100``/``low_min=0`` and ``fast_k=0`` -> %K = 0 (< 20). Open times
    land within the last few minutes of ``now_ms`` so ``check_buffer_freshness``
    returns ``True`` and ``run_forever`` keeps (rather than re-bootstraps) the
    buffer (REQ-8.5 fresh branch). With ``is_oversold`` already engaged, these
    finite readings produce SUPPRESS and never reset the lock, so a non-default
    ``is_oversold == True`` is preserved meaningfully across the reconnect.
    """
    return [
        Candle(
            open_time_ms=now_ms - (count - i) * 60_000,
            open=0.0,
            high=100.0,
            low=0.0,
            close=0.0,
            volume=10.0,
        )
        for i in range(count)
    ]


def _oversold_frame(open_time_ms: int) -> str:
    """Serialize a closed kline in the SAME band as ``_recent_oversold_buffer``.

    Uses ``high=100, low=0, close=0`` (matching the seeded buffer) so over every
    5-bar window ``high_max=100`` / ``low_min=0`` and ``fast_k=0`` -> %K stays 0
    (< 20). Streaming these keeps the cool-down lock engaged (SUPPRESS only) and
    never resets ``is_oversold``, so its preserved ``True`` value is meaningful
    across the reconnect. A mismatched band would let a transition window's
    ``low_min`` straddle 0 and 90 and spuriously lift %K above 20 (a RESET).
    """
    return json.dumps(
        {
            "e": "kline",
            "E": open_time_ms,
            "s": "BTCUSDT",
            "k": {
                "t": open_time_ms,
                "o": 0.0,
                "h": 100.0,
                "l": 0.0,
                "c": 0.0,
                "v": 10.0,
                "x": True,
            },
        }
    )


def test_reconnect_preserves_buffer_and_oversold_state(monkeypatch):
    """**Validates: Requirements 8.3, 8.4**

    Injecting ``on_close(code=1006, reason="abnormal")`` per session, the
    reconnection loop sleeps 5 seconds once per disconnect and re-opens with the
    candle buffer and ``is_oversold`` flag preserved from the prior session.
    ``bootstrap_history`` is never called because the buffer stays fresh.
    """
    # Silence the shared Monitor logger so this test produces no stray output
    # and does not touch any FileHandler left over from other tests.
    logger = logging.getLogger(bsm.LOGGER_NAME)
    null_handler = logging.NullHandler()
    old_level, old_propagate = logger.level, logger.propagate
    logger.setLevel(logging.CRITICAL + 10)
    logger.propagate = False
    logger.addHandler(null_handler)

    num_disconnects = 2
    now_ms = 2_000_000_000_000  # fixed wall clock (ms), multiple of 1000

    # Fresh, oversold buffer; engage the cool-down lock so its True value is a
    # meaningful (non-default) thing to preserve across the reconnect.
    state = MonitorState(buffer=_recent_oversold_buffer(now_ms), is_oversold=True)

    # Each session drains a couple of closed candles (newer than the current
    # buffer max so they append) -- buffer growth makes preservation non-trivial.
    base = now_ms - 30_000
    messages_per_session = [
        [
            _oversold_frame(base + 5_000),
            _oversold_frame(base + 10_000),
        ],
        [
            _oversold_frame(base + 15_000),
        ],
    ]

    # Controlled, fixed clock so the buffer is always fresh (REQ-8.5 fresh path).
    def fake_time() -> float:
        return now_ms / 1000.0

    # Recording sleep that breaks the infinite loop after N disconnects (REQ-8.3).
    sleep_args: list[float] = []

    def fake_sleep(seconds) -> None:
        sleep_args.append(seconds)
        if len(sleep_args) >= num_disconnects:
            raise _StopLoop

    # bootstrap_history must NOT run while the buffer is fresh; flag it if it does.
    bootstrap_calls = {"count": 0}

    def fake_bootstrap(_state: MonitorState) -> None:
        bootstrap_calls["count"] += 1

    open_snaps: list[tuple[list[Candle], bool]] = []
    close_snaps: list[tuple[list[Candle], bool]] = []
    session_counter = {"i": 0}

    class _FakeSession:
        """One fake WebSocket session: open -> drain closed candles -> 1006 close."""

        def __init__(self, url=None, on_open=None, on_message=None, on_close=None, **_kw):
            self.url = url
            self.on_open = on_open
            self.on_message = on_message
            self.on_close = on_close

        def run_forever(self, *_a, **_k):
            i = session_counter["i"]
            # "Immediately after the new session opens": snapshot BEFORE any
            # message mutates the buffer (REQ-8.4).
            open_snaps.append((list(state.buffer), state.is_oversold))

            if self.on_open is not None:
                self.on_open(self)

            for raw in messages_per_session[i] if i < len(messages_per_session) else []:
                if self.on_message is not None:
                    self.on_message(self, raw)

            # "Immediately before the disconnect" snapshot.
            close_snaps.append((list(state.buffer), state.is_oversold))
            session_counter["i"] += 1

            # Inject the abnormal disconnect the task requires (REQ-8.6).
            if self.on_close is not None:
                self.on_close(self, 1006, "abnormal")

        def close(self, *_a, **_k):  # pragma: no cover - API parity only
            pass

    try:
        with mock.patch.object(bsm, "WebSocketApp", _FakeSession), mock.patch.object(
            bsm.time, "sleep", fake_sleep
        ), mock.patch.object(bsm.time, "time", fake_time), mock.patch.object(
            bsm, "bootstrap_history", fake_bootstrap
        ):
            try:
                bsm.run_forever(state)
            except _StopLoop:
                pass
    finally:
        logger.removeHandler(null_handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate

    # --- REQ-8.3: exactly one 5-second reconnect sleep per disconnect --------
    assert sleep_args == [bsm.RECONNECT_DELAY_S] * num_disconnects
    assert bsm.RECONNECT_DELAY_S == 5

    # One session opened and one disconnect observed per iteration.
    assert len(open_snaps) == num_disconnects
    assert len(close_snaps) == num_disconnects

    # --- REQ-8.5 fresh branch: the buffer never went stale, so no re-bootstrap.
    assert bootstrap_calls["count"] == 0, "fresh buffer must not trigger re-bootstrap"

    # --- REQ-8.4: state preserved across each reconnect ----------------------
    # Session 0 opens with the initial state.
    initial_buffer = _recent_oversold_buffer(now_ms)
    assert open_snaps[0][0] == initial_buffer
    assert open_snaps[0][1] is True
    # Each later session opens with EXACTLY the prior session's end state.
    for i in range(1, num_disconnects):
        assert open_snaps[i][0] == close_snaps[i - 1][0], (
            f"iteration {i}: buffer not preserved across the reconnect"
        )
        assert open_snaps[i][1] == close_snaps[i - 1][1], (
            f"iteration {i}: is_oversold not preserved across the reconnect"
        )
    # Preservation is non-trivial: session 0 grew the buffer by two candles, and
    # the cool-down lock survived as True the whole way through.
    assert len(close_snaps[0][0]) == len(initial_buffer) + 2
    assert open_snaps[1][0] != initial_buffer
    assert close_snaps[-1][1] is True
