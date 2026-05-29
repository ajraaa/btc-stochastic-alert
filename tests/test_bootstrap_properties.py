"""Property-based and unit tests for the historical Bootstrapper.

``btc_stochastic_monitor.bootstrap_history(state)`` issues a single HTTPS GET to
the Binance REST ``/api/v3/klines`` endpoint and, on an HTTP 200 response whose
body is a JSON array of at least 10 rows, parses each row
``[openTime, open, high, low, close, volume, ...]`` into a :class:`Candle`, sorts
the candles ascending by ``open_time_ms``, and assigns them to ``state.buffer``.

This file hosts:

* **Property 2 (task 9.2):** Bootstrap preserves rows in chronological order.
  **Validates: Requirements 2.2, 2.3.** A Hypothesis strategy generates
  well-formed responses of 10..50 rows with strictly-ascending open times and
  finite OHLCV; the endpoint returns the rows in a *shuffled* order so the
  ascending-sort guarantee is exercised meaningfully. After ``bootstrap_history``
  runs, the buffer must have the same length, be sorted strictly ascending by
  ``open_time_ms``, and have every ``Candle`` field equal to the corresponding
  source-row field.
* A deterministic complement unit test that feeds an explicitly out-of-order
  response and asserts the ascending sort plus 1:1 field mapping.

Task 9.3 appends its retry/failure-mode unit tests to this same file, so the
shared imports, constants, helpers, and strategies live at module top and every
test is a standalone top-level function to avoid write conflicts.
"""

from __future__ import annotations

from unittest import mock

import responses
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import btc_stochastic_monitor as bsm
from btc_stochastic_monitor import Candle, MonitorState

# ---------------------------------------------------------------------------
# Shared constants and strategies
# ---------------------------------------------------------------------------

# One 1-day interval in milliseconds; used to derive realistic closeTime values
# and to space generated open times by up to a full day.
ONE_DAY_MS = 86_400_000

# Finite OHLCV component. ``bootstrap_history`` only calls ``float(...)`` on
# these fields, so any finite value is valid input; bounds keep the numbers in a
# sane range while still spanning negatives, zero, and large magnitudes. Values
# round-trip exactly through JSON (json uses the shortest round-tripping repr),
# so exact equality assertions below are sound.
finite_floats = st.floats(
    allow_nan=False,
    allow_infinity=False,
    min_value=-1e9,
    max_value=1e9,
)


@st.composite
def kline_responses(draw):
    """Generate a well-formed Binance kline response of 10..50 rows.

    Returns a ``(source_rows, response_rows)`` tuple where:

    * ``source_rows`` is the canonical list of Binance-style kline rows ordered
      strictly ascending by open time. Each row is a 12-element list mirroring
      Binance's real payload shape ``[openTime(int ms), open, high, low, close,
      volume, closeTime, ...]``; ``bootstrap_history`` reads only indices 0..5
      and ignores the rest.
    * ``response_rows`` is the same rows in a Hypothesis-chosen permuted order.
      Returning them shuffled forces the implementation's ascending sort to do
      real work, so the "chronological order" claim of Property 2 is tested
      rather than trivially satisfied by pre-sorted input.

    Open times are strictly ascending (built from a random start plus strictly
    positive cumulative increments), guaranteeing unique open times so the
    buffer ordering after sorting is unambiguous.
    """
    n = draw(st.integers(min_value=10, max_value=50))

    # Strictly-ascending open times: a random start plus positive increments.
    start = draw(st.integers(min_value=0, max_value=2_000_000_000_000))
    increments = draw(
        st.lists(
            st.integers(min_value=1, max_value=ONE_DAY_MS),
            min_size=n,
            max_size=n,
        )
    )
    open_times: list[int] = []
    current = start
    for inc in increments:
        current += inc
        open_times.append(current)

    # Build canonical ascending rows with finite OHLCV per row.
    source_rows: list[list] = []
    for open_time in open_times:
        o = draw(finite_floats)
        h = draw(finite_floats)
        low = draw(finite_floats)
        c = draw(finite_floats)
        v = draw(finite_floats)
        close_time = open_time + ONE_DAY_MS - 1
        # 12-element row mirroring Binance; only indices 0..5 are parsed.
        source_rows.append(
            [open_time, o, h, low, c, v, close_time, 0.0, 0, 0.0, 0.0, "0"]
        )

    # Permute the response order so the implementation's sort is exercised.
    order = draw(st.permutations(range(n)))
    response_rows = [source_rows[i] for i in order]

    return source_rows, response_rows


def _raise_on_sleep(*_args, **_kwargs):
    """Fail fast if ``bootstrap_history`` retries.

    A valid HTTP 200 response with >= 10 rows must populate the buffer and
    return on the first attempt (REQ-2.6). The only place ``bootstrap_history``
    sleeps is between retries, so any call to ``time.sleep`` means an unexpected
    retry occurred. Raising here surfaces that as an immediate test failure
    instead of an infinite loop, and doubly satisfies the "do not actually
    sleep" defensive requirement.
    """
    raise AssertionError(
        "bootstrap_history called time.sleep, indicating an unexpected retry; "
        "a valid 200 response with >= 10 rows must populate the buffer without "
        "retrying."
    )


# ---------------------------------------------------------------------------
# Property 2: Bootstrap preserves rows in chronological order
# Validates: Requirements 2.2, 2.3
# ---------------------------------------------------------------------------
@given(payload=kline_responses())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property2_bootstrap_preserves_rows_in_chronological_order(payload):
    """**Property 2: Bootstrap preserves rows in chronological order**

    **Validates: Requirements 2.2, 2.3**

    For any well-formed REST kline response (10..50 rows, finite OHLCV, unique
    open times), after ``bootstrap_history`` the buffer (a) has the same length
    as the response, (b) is sorted strictly ascending by ``open_time_ms``, and
    (c) has each ``Candle``'s six fields equal to the corresponding source-row
    fields. The endpoint returns the rows shuffled, so passing this property
    also confirms the implementation actually sorts.
    """
    source_rows, response_rows = payload
    n = len(source_rows)

    state = MonitorState(buffer=[], is_oversold=False, logger=None)

    # ``responses.RequestsMock`` is created per example so registrations never
    # accumulate across Hypothesis iterations. ``time.sleep`` is patched to fail
    # fast on any unexpected retry rather than hang the test.
    with mock.patch.object(bsm.time, "sleep", _raise_on_sleep):
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                bsm.BINANCE_REST_URL,
                json=response_rows,
                status=200,
            )
            bsm.bootstrap_history(state)

    buffer = state.buffer

    # (a) Same length as the source response (REQ-2.2).
    assert len(buffer) == n, f"expected {n} candles, got {len(buffer)}"

    # (b) Strictly ascending by open_time_ms -> chronological order (REQ-2.2).
    open_times = [c.open_time_ms for c in buffer]
    assert open_times == sorted(open_times), f"buffer not ascending: {open_times}"
    assert all(
        open_times[i] < open_times[i + 1] for i in range(len(open_times) - 1)
    ), f"buffer open times not strictly ascending: {open_times}"

    # (c) Every Candle field equals the corresponding source-row field
    # (REQ-2.3). The buffer is ascending and ``source_rows`` is the canonical
    # ascending list, so a positional zip matches like-for-like; the open-time
    # assertion inside the loop also guards against any misalignment.
    for candle, row in zip(buffer, source_rows):
        assert candle.open_time_ms == int(row[0])
        assert candle.open == float(row[1])
        assert candle.high == float(row[2])
        assert candle.low == float(row[3])
        assert candle.close == float(row[4])
        assert candle.volume == float(row[5])


# ---------------------------------------------------------------------------
# Deterministic complement: explicitly out-of-order input is sorted ascending
# Validates: Requirements 2.2, 2.3
# ---------------------------------------------------------------------------
@responses.activate
def test_bootstrap_sorts_descending_response_into_ascending_buffer():
    """A response delivered in descending order is sorted ascending with fields intact.

    **Validates: Requirements 2.2, 2.3**

    Deterministic counterpart to Property 2: 12 rows are returned newest-first,
    and ``bootstrap_history`` must produce an ascending buffer whose candles map
    1:1 (by open time) to the source rows.
    """
    # 12 Binance-style rows in DESCENDING open-time order (12 days .. 1 day).
    rows = []
    for i in range(12):
        open_time = (12 - i) * ONE_DAY_MS
        rows.append(
            [
                open_time,
                float(i),          # open
                float(i + 1),      # high
                float(i - 1),      # low
                float(i) + 0.5,    # close
                float(i) * 10.0,   # volume
                open_time + ONE_DAY_MS - 1,  # closeTime (ignored)
            ]
        )

    responses.add(responses.GET, bsm.BINANCE_REST_URL, json=rows, status=200)

    state = MonitorState(buffer=[], is_oversold=False, logger=None)
    with mock.patch.object(bsm.time, "sleep", _raise_on_sleep):
        bsm.bootstrap_history(state)

    # Length preserved and ordering is now ascending (REQ-2.2).
    assert len(state.buffer) == 12
    open_times = [c.open_time_ms for c in state.buffer]
    assert open_times == sorted(open_times)
    assert open_times[0] == 1 * ONE_DAY_MS
    assert open_times[-1] == 12 * ONE_DAY_MS

    # Field equality matched by open time (REQ-2.3).
    by_open_time = {int(row[0]): row for row in rows}
    for candle in state.buffer:
        row = by_open_time[candle.open_time_ms]
        assert candle.open == float(row[1])
        assert candle.high == float(row[2])
        assert candle.low == float(row[3])
        assert candle.close == float(row[4])
        assert candle.volume == float(row[5])
