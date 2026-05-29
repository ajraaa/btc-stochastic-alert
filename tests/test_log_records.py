"""Property-based and unit tests for the Monitor's required log records.

This file hosts the log-record audit correctness property for the BTC
Stochastic DCA Monitor:

* **Property 10 (task 13.1): Required log records emitted with required fields.**
  **Validates: Requirements 10.4, 10.5, 10.6.**

What the implementation does (verified against ``btc_stochastic_monitor.py``)
-----------------------------------------------------------------------------
* ``process_data(state, candle)`` emits exactly ONE ``INDICATOR``-category INFO
  record per closed candle (REQ-10.4). The record is one of two shapes:

  - *skipped* (when ``compute_stochastic`` returns ``None`` -- warm-up / NaN):
    ``"Indicator evaluation skipped for closed candle (open_time_ms=...): no
    finite Stochastic reading (warm-up or NaN)."`` -- carries no %K/%D.
  - *finite* (a real reading): ``"Indicator evaluation complete:
    close_time_ms=<t> %K=<k.4f> %D=<d.4f> decision=<repr>"`` and, on a ``FIRE``
    decision, an additional ``dispatched=<bool>`` (REQ-10.5). ``decision`` is the
    string value of one of the four :class:`~btc_stochastic_monitor.TriggerDecision`
    members: ``"alert dispatched"``, ``"suppressed by cool-down"``,
    ``"cool-down released"``, or ``"no oversold condition"``.

* ``send_telegram_notification(message)`` emits exactly one record describing the
  dispatch outcome (REQ-10.6): a ``NOTIFY`` INFO ("dispatched successfully (HTTP
  ...)") on a 2xx, a ``NOTIFY`` ERROR including the HTTP status code on a non-2xx,
  a ``NOTIFY`` ERROR naming the exception type on a ``requests.RequestException``,
  or a ``CONFIG`` ERROR when credentials are missing.

* ``run_forever`` emits one ``RECONNECT`` INFO record carrying a reconnect-attempt
  counter per disconnect, and ``main`` emits one ``STARTUP`` INFO record on
  startup completion (REQ-10.4).

Test design
-----------
The property part (Part 1) drives generated sequences of closed candles directly
through ``process_data`` and asserts the COUNT invariant (one ``INDICATOR`` record
per call, REQ-10.4) plus, per record, the field-presence / field-value invariant
(REQ-10.5). A from-scratch shadow buffer is updated in lock-step with the Monitor's
buffer so the test knows independently whether each candle yields a finite reading
(and, if so, the exact ``%K`` / ``%D`` / ``close_time_ms`` to expect).

Part 2 drives at least one ``FIRE`` per case with the Telegram endpoint mocked via
``responses`` and asserts exactly one ``NOTIFY`` record per dispatch, that success
is indicated on a 2xx, and that a failure reason (HTTP status / exception type) is
included on failure (REQ-10.6).

Part 3 adds focused checks that a single ``STARTUP`` record is emitted on startup
and one ``RECONNECT`` record is emitted per reconnect attempt (REQ-10.4).
"""

from __future__ import annotations

import logging
from unittest import mock

import pytest
import requests
import responses
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import btc_stochastic_monitor as bsm
from btc_stochastic_monitor import Candle, MonitorState, TriggerDecision

# Valid credentials used wherever a real dispatch must be exercised. The mocked
# Telegram endpoint URL is built from the same token the implementation uses.
VALID_TOKEN = "123456:test-bot-token"
VALID_CHAT_ID = "987654321"
TELEGRAM_URL = f"https://api.telegram.org/bot{VALID_TOKEN}/sendMessage"

# The four trigger-decision strings that a finite INDICATOR record must contain
# exactly one of (REQ-10.5).
DECISION_STRINGS = [member.value for member in TriggerDecision]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _records_for(records, category, *, level=None):
    """Filter captured records by event category (and optionally level)."""
    out = []
    for record in records:
        if getattr(record, "category", None) != category:
            continue
        if level is not None and record.levelno != level:
            continue
        out.append(record)
    return out


class _RecordCollector(logging.Handler):
    """A logging handler that simply accumulates every record it receives.

    Used by the property test (Part 1) instead of the function-scoped ``caplog``
    fixture so the capture is self-contained per Hypothesis example (avoiding the
    fixture-not-reset-between-examples pitfall).
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _oversold_seed_buffer(n: int = 12) -> list[Candle]:
    """Build a buffer of ``n`` candles whose Stochastic %K computes to 0.

    Every candle has ``high == 100``, ``low == 0``, ``close == 0`` so that over
    any 5-bar window ``high_max == 100`` and ``low_min == 0`` (denominator 100,
    not a flat 0/0 window) and ``fast_k == 0`` -> smoothed %K == 0 -> %D == 0.
    With ``n >= 9`` the last row is finite, so pushing one more such candle drives
    a finite reading with ``k == 0 < OVERSOLD_THRESHOLD``: from a not-oversold
    state this fires exactly one alert (REQ-6.2).
    """
    return [
        Candle(open_time_ms=i, open=0.0, high=100.0, low=0.0, close=0.0, volume=10.0)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Strategy: a sequence of well-formed closed candles
# ---------------------------------------------------------------------------
@st.composite
def candle_specs(draw):
    """Generate 1..20 OHLCV specs with a strictly positive high/low spread.

    Each spec is ``(low, spread, open_frac, close_frac, volume)``; the test
    materialises a :class:`Candle` with ``high = low + spread`` (so
    ``high > low``) and ``open`` / ``close`` inside ``[low, high]``. Open times are
    assigned strictly ascending by the test (index order) so every candle is
    appended by ``update_buffer``. Starting from an empty buffer, the first
    candles are warm-up (``None`` reading -> "skipped" record) and later candles
    yield finite readings, so both INDICATOR record shapes are exercised.
    """
    n = draw(st.integers(min_value=1, max_value=20))
    specs = []
    for _ in range(n):
        low = draw(
            st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)
        )
        spread = draw(
            st.floats(min_value=0.5, max_value=10_000.0, allow_nan=False, allow_infinity=False)
        )
        open_frac = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
        close_frac = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
        volume = draw(
            st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)
        )
        specs.append((low, spread, open_frac, close_frac, volume))
    return specs


# ---------------------------------------------------------------------------
# Property 10 (Part 1): one INDICATOR record per closed candle, with required
# fields on finite readings.
# Validates: Requirements 10.4, 10.5
# ---------------------------------------------------------------------------
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(specs=candle_specs())
def test_property10_one_indicator_record_per_candle_with_fields(specs) -> None:
    """**Property 10: Required log records emitted with required fields** (INDICATOR)

    **Validates: Requirements 10.4, 10.5**

    For ANY generated sequence of closed candles fed one-by-one to
    ``process_data``, exactly one ``INDICATOR`` record is emitted per call
    (REQ-10.4). For each record that corresponds to a finite reading (detected by
    the ``"decision="`` marker), the message contains the candle ``close_time_ms``,
    the latest ``%K`` and ``%D`` (formatted to four decimals), and exactly one of
    the four trigger-decision strings (REQ-10.5). For the warm-up / NaN case the
    single record notes that evaluation was "skipped".
    """
    logger = logging.getLogger(bsm.LOGGER_NAME)
    collector = _RecordCollector()
    prev_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(collector)

    try:
        # Stub the Notifier so a FIRE decision performs no real network I/O and
        # emits no NOTIFY record; this part isolates the INDICATOR invariants.
        with mock.patch.object(bsm, "send_telegram_notification", lambda _msg: True):
            state = MonitorState(buffer=[])
            # Shadow buffer mirrors state.buffer exactly (same candles, same
            # order) so an independent compute_stochastic tells us whether each
            # candle yields a finite reading and, if so, its exact %K / %D.
            shadow: list[Candle] = []
            indicator_seen = 0

            for index, (low, spread, open_frac, close_frac, volume) in enumerate(specs):
                high = low + spread
                candle = Candle(
                    open_time_ms=index,  # strictly ascending -> always appended
                    open=low + open_frac * spread,
                    high=high,
                    low=low,
                    close=low + close_frac * spread,
                    volume=volume,
                )

                # Independent oracle: mirror the buffer update and recompute.
                bsm.update_buffer(shadow, candle)
                expected = bsm.compute_stochastic(shadow)

                bsm.process_data(state, candle)

                indicator_records = _records_for(collector.records, "INDICATOR")
                new_records = indicator_records[indicator_seen:]

                # REQ-10.4: exactly one INDICATOR record per closed candle.
                assert len(new_records) == 1, (
                    f"candle #{index}: expected exactly one INDICATOR record, "
                    f"got {len(new_records)}"
                )
                indicator_seen = len(indicator_records)

                message = new_records[0].getMessage()

                if expected is None:
                    # Warm-up / NaN: the record still exists and notes the skip.
                    assert "skipped" in message.lower(), (
                        f"candle #{index}: None-reading record must note 'skipped': "
                        f"{message!r}"
                    )
                    assert "decision=" not in message
                else:
                    # Finite reading -> REQ-10.5 required fields.
                    assert "decision=" in message, (
                        f"candle #{index}: finite-reading record must carry a "
                        f"decision field: {message!r}"
                    )
                    assert f"close_time_ms={expected.close_time_ms}" in message, (
                        f"candle #{index}: record must include close_time_ms: {message!r}"
                    )
                    assert f"%K={expected.k:.4f}" in message, (
                        f"candle #{index}: record must include the latest %K "
                        f"(formatted): {message!r}"
                    )
                    assert f"%D={expected.d:.4f}" in message, (
                        f"candle #{index}: record must include the latest %D "
                        f"(formatted): {message!r}"
                    )
                    matched = [value for value in DECISION_STRINGS if value in message]
                    assert len(matched) == 1, (
                        f"candle #{index}: record must contain exactly one of the four "
                        f"decision strings; matched {matched} in {message!r}"
                    )

            # Total invariant: one INDICATOR record per process_data call.
            assert indicator_seen == len(specs)
    finally:
        logger.removeHandler(collector)
        logger.setLevel(prev_level)


# ---------------------------------------------------------------------------
# Property 10 (Part 2): one NOTIFY record per dispatch with success/failure info.
# Validates: Requirements 10.6
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "scenario",
    ["success", "http_failure", "transport_failure"],
)
@responses.activate
def test_property10_one_notify_record_per_dispatch(scenario, monkeypatch, caplog):
    """**Property 10: Required log records emitted with required fields** (NOTIFY)

    **Validates: Requirements 10.6**

    Driving a single ``FIRE`` (a not-oversold state plus a candle whose %K is 0)
    with the Telegram endpoint mocked produces exactly one ``NOTIFY`` record per
    dispatch. On success the record indicates success; on failure (non-2xx or a
    transport exception) it includes a failure reason (the HTTP status code or the
    exception type). The candle's single ``INDICATOR`` record records the FIRE
    decision and the dispatch outcome.
    """
    monkeypatch.setattr(bsm, "TELEGRAM_BOT_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(bsm, "TELEGRAM_CHAT_ID", VALID_CHAT_ID)

    if scenario == "success":
        responses.add(responses.POST, TELEGRAM_URL, json={"ok": True}, status=200)
        expect_success = True
    elif scenario == "http_failure":
        responses.add(responses.POST, TELEGRAM_URL, status=500)
        expect_success = False
    else:  # transport_failure
        responses.add(responses.POST, TELEGRAM_URL, body=requests.exceptions.Timeout())
        expect_success = False

    # Not oversold + an oversold-driving candle -> exactly one FIRE / dispatch.
    state = MonitorState(buffer=_oversold_seed_buffer(12), is_oversold=False)
    fire_candle = Candle(open_time_ms=12, open=0.0, high=100.0, low=0.0, close=0.0, volume=10.0)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=bsm.LOGGER_NAME):
        bsm.process_data(state, fire_candle)

    # --- REQ-10.6: exactly one NOTIFY record describing the dispatch ---------
    notify_records = _records_for(caplog.records, "NOTIFY")
    assert len(notify_records) == 1, (
        f"expected exactly one NOTIFY record per dispatch, got {len(notify_records)}"
    )
    notify_message = notify_records[0].getMessage()

    if expect_success:
        assert notify_records[0].levelno == logging.INFO
        assert "success" in notify_message.lower(), (
            f"successful dispatch record must indicate success: {notify_message!r}"
        )
        assert "200" in notify_message  # HTTP 200 success indicator
    else:
        assert notify_records[0].levelno == logging.ERROR
        if scenario == "http_failure":
            assert "500" in notify_message, (
                f"failed dispatch record must include the HTTP status code: "
                f"{notify_message!r}"
            )
        else:
            assert "Timeout" in notify_message, (
                f"failed dispatch record must include the exception type as the "
                f"failure reason: {notify_message!r}"
            )

    # --- The FIRE candle's single INDICATOR record carries the outcome -------
    indicator_records = _records_for(caplog.records, "INDICATOR", level=logging.INFO)
    assert len(indicator_records) == 1
    indicator_message = indicator_records[0].getMessage()
    assert TriggerDecision.FIRE.value in indicator_message  # "alert dispatched"
    assert f"dispatched={expect_success}" in indicator_message


@responses.activate
def test_property10_notify_records_count_matches_dispatches(monkeypatch, caplog):
    """Across several FIRE/RESET episodes, NOTIFY records equal the dispatch count.

    Drives two distinct oversold episodes separated by a recovery (%K back above
    the threshold). Exactly two FIRE dispatches occur, so exactly two NOTIFY
    records are emitted -- one per dispatch (REQ-10.6).
    """
    monkeypatch.setattr(bsm, "TELEGRAM_BOT_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(bsm, "TELEGRAM_CHAT_ID", VALID_CHAT_ID)
    responses.add(responses.POST, TELEGRAM_URL, json={"ok": True}, status=200)

    state = MonitorState(buffer=_oversold_seed_buffer(12), is_oversold=False)

    # Helper candle builders keyed on whether they push %K oversold or recovered.
    def oversold_candle(t):
        return Candle(open_time_ms=t, open=0.0, high=100.0, low=0.0, close=0.0, volume=10.0)

    def recovered_candle(t):
        # close at the top of the range -> %K rises to 100 (strictly > 20),
        # releasing the cool-down lock so the next oversold candle can fire again.
        return Candle(open_time_ms=t, open=100.0, high=100.0, low=0.0, close=100.0, volume=10.0)

    # NOTE: %K is a 3-period SMA of fast %K, so it reacts with a lag: a single
    # oversold/recovered candle does not flip the smoothed %K across 20 -- it
    # takes a few consecutive candles to flush the smoothing window. Episode 2
    # therefore pushes several oversold candles; only the one that finally drives
    # the smoothed %K strictly below 20 (from a released lock) FIREs.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=bsm.LOGGER_NAME):
        bsm.process_data(state, oversold_candle(12))   # episode 1: FIRE (seed all oversold)
        bsm.process_data(state, oversold_candle(13))   # SUPPRESS (still locked)
        # Push several recovered candles so the smoothed %K rises above 20 -> RESET.
        for t in range(14, 20):
            bsm.process_data(state, recovered_candle(t))
        # Push several oversold candles so the smoothed %K falls back below 20;
        # the lock is released, so the crossing FIREs exactly one more dispatch.
        for t in range(20, 24):
            bsm.process_data(state, oversold_candle(t))  # episode 2: one FIRE

    notify_records = _records_for(caplog.records, "NOTIFY")
    # Two FIRE dispatches -> two NOTIFY records (REQ-10.6). No more, no fewer.
    assert len(notify_records) == 2, (
        f"expected one NOTIFY record per dispatch (2 dispatches), got "
        f"{len(notify_records)}"
    )
    assert all(record.levelno == logging.INFO for record in notify_records)
    assert all("success" in record.getMessage().lower() for record in notify_records)


# ---------------------------------------------------------------------------
# Property 10 (Part 3): STARTUP and RECONNECT records (REQ-10.4)
# ---------------------------------------------------------------------------
def test_property10_startup_record_emitted_once(monkeypatch, tmp_path, caplog):
    """``main`` emits exactly one STARTUP record on startup completion (REQ-10.4).

    ``validate_config`` is satisfied with valid credentials; ``bootstrap_history``
    and ``run_forever`` are stubbed so ``main`` returns immediately after logging
    startup. The backup log path is redirected to a temp file.
    """
    monkeypatch.setattr(bsm, "TELEGRAM_BOT_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(bsm, "TELEGRAM_CHAT_ID", VALID_CHAT_ID)
    monkeypatch.setattr(bsm, "BACKUP_LOG_FILE", str(tmp_path / "monitor.log"))
    monkeypatch.setattr(bsm, "bootstrap_history", lambda state: None)
    monkeypatch.setattr(bsm, "run_forever", lambda state: None)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=bsm.LOGGER_NAME):
        bsm.main()

    startup_records = _records_for(caplog.records, "STARTUP", level=logging.INFO)
    assert len(startup_records) == 1, (
        f"expected exactly one STARTUP record, got {len(startup_records)}"
    )
    assert "startup" in startup_records[0].getMessage().lower()


@pytest.mark.parametrize("disconnects", [1, 3])
def test_property10_one_reconnect_record_per_attempt(disconnects, monkeypatch, caplog):
    """``run_forever`` emits one RECONNECT record per reconnect attempt (REQ-10.4).

    A fake ``WebSocketApp`` opens then closes abnormally each iteration; ``time.sleep``
    is patched to break the infinite loop after ``disconnects`` disconnects. The
    buffer is seeded fresh (newest candle == now) so no stale-recovery RECONNECT
    warning is emitted, isolating the per-attempt RECONNECT INFO records.
    """
    import time as _time

    class _StopLoop(BaseException):
        """Breaks the infinite reconnection loop from the patched ``time.sleep``."""

    now_ms = int(_time.time() * 1000)
    # Fresh buffer: newest candle's open time is "now", so check_buffer_freshness
    # returns True and run_forever skips re-bootstrap (no RECONNECT WARNING).
    state = MonitorState(
        buffer=[Candle(open_time_ms=now_ms, open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0)],
        is_oversold=False,
    )

    sleep_calls: list[float] = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) >= disconnects:
            raise _StopLoop

    class _FakeWS:
        def __init__(self, url=None, on_open=None, on_message=None, on_close=None, **_kw):
            self.on_open = on_open
            self.on_close = on_close

        def run_forever(self, *_a, **_k):
            if self.on_open is not None:
                self.on_open(self)
            if self.on_close is not None:
                self.on_close(self, 1006, "abnormal")

        def close(self, *_a, **_k):  # pragma: no cover - API parity only
            pass

    monkeypatch.setattr(bsm, "WebSocketApp", _FakeWS)
    monkeypatch.setattr(bsm.time, "sleep", fake_sleep)
    monkeypatch.setattr(bsm, "bootstrap_history", lambda state: None)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=bsm.LOGGER_NAME):
        try:
            bsm.run_forever(state)
        except _StopLoop:
            pass

    reconnect_records = _records_for(caplog.records, "RECONNECT", level=logging.INFO)
    assert len(reconnect_records) == disconnects, (
        f"expected one RECONNECT record per disconnect ({disconnects}), got "
        f"{len(reconnect_records)}"
    )
    assert all("reconnect attempt" in record.getMessage().lower() for record in reconnect_records)
    # One reconnect delay per disconnect (REQ-8.3 corollary of REQ-10.4 counting).
    assert sleep_calls == [bsm.RECONNECT_DELAY_S] * disconnects
