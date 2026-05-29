"""Property-based tests for the Stochastic indicator (``compute_stochastic``).

This module hosts the indicator correctness properties for the BTC Stochastic
DCA Monitor:

* Property 3 (this file): for any candle buffer whose last row yields finite
  output, ``compute_stochastic`` returns a ``StochasticReading`` whose ``k`` and
  ``d`` equal the last-row values of an **independent** STOCH(5, 3, 3)
  recomputation, read from columns named exactly ``STOCHk_5_3_3`` /
  ``STOCHd_5_3_3`` (REQ-1.5, REQ-5.1, REQ-5.2, REQ-5.3).
* Property 4 (task 6.3): appended to this same file later -- the NaN / missing
  column "skip" case (REQ-5.4).

Architectural note
------------------
``pandas-ta`` has been deprecated in this project: the runtime targets Python
3.14+, where ``numba`` (a ``pandas-ta`` build dependency) refuses to build, so
``compute_stochastic`` now computes STOCH(5, 3, 3) with **native vanilla pandas
rolling windows** (no ``pandas_ta``, no ``numba``). Property 3 is therefore
validated by an *independent* reference computation built directly with pandas
from the generated candles -- not by spying on ``pandas_ta`` -- while still
asserting the configured parameters (length=5, k_smooth=3, d_smooth=3) and the
exact configured column names are what the implementation uses.

The strategies and helpers below are placed at module scope so the Property 4
test (task 6.3) can reuse the same candle / buffer generators.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import btc_stochastic_monitor as bsm
from btc_stochastic_monitor import Candle, compute_stochastic


# ---------------------------------------------------------------------------
# Shared strategies (reusable by Property 4 / task 6.3)
# ---------------------------------------------------------------------------

# Price prices are kept in a comfortably positive, finite range so that the
# division ``(close - low_min) / (high_max - low_min)`` stays well-conditioned
# and never overflows; the exact magnitudes are irrelevant to the property.
_PRICE_MIN = 1.0
_PRICE_MAX = 1_000_000.0

# A strictly-positive spread between low and high. Forcing ``high > low`` for
# every candle guarantees ``high_max > low_min`` over any fully-populated 5-bar
# window (the argmin-of-low candle still has ``high > low``), so the only source
# of NaN in the last row is the rolling warm-up -- which the test skips via
# ``assume`` -- rather than a degenerate ``0 / 0`` flat-window division.
_SPREAD_MIN = 0.01
_SPREAD_MAX = 100_000.0


@st.composite
def candle_strategy(draw, open_time_ms: int = 0):
    """Generate a single realistic OHLCV :class:`Candle`.

    Invariants enforced (realistic OHLC):
        * ``high > low`` (strictly positive spread),
        * ``low <= open <= high`` and ``low <= close <= high`` so that
          ``high >= max(open, close)`` and ``low <= min(open, close)``.

    ``open_time_ms`` is supplied by the caller (the buffer strategy assigns
    strictly-ascending times) so the buffer is chronologically ordered like the
    real candle buffer.
    """
    low = draw(
        st.floats(
            min_value=_PRICE_MIN,
            max_value=_PRICE_MAX,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    spread = draw(
        st.floats(
            min_value=_SPREAD_MIN,
            max_value=_SPREAD_MAX,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    high = low + spread
    # open / close fall anywhere within [low, high] (fractions of the spread).
    open_frac = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    close_frac = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    open_price = low + open_frac * spread
    close_price = low + close_frac * spread
    volume = draw(
        st.floats(
            min_value=0.0,
            max_value=_PRICE_MAX,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return Candle(
        open_time_ms=open_time_ms,
        open=open_price,
        high=high,
        low=low,
        close=close_price,
        volume=volume,
    )


@st.composite
def buffer_strategy(draw, min_size: int = 8, max_size: int = 50):
    """Generate a chronological candle buffer of ``min_size..max_size`` candles.

    Open times are strictly ascending and unique (built from a random start
    plus strictly-positive increments), mirroring the real buffer invariant.
    Default size range is 8..50 per Property 3; STOCH(5, 3, 3) needs 9 candles
    before the last row's %D is non-NaN, so the smallest buffers are skipped by
    the caller's ``assume`` (the value-equality property does not apply during
    warm-up; that None case is covered by Property 4 / task 6.3).
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    start = draw(st.integers(min_value=0, max_value=1_000_000))
    increments = draw(
        st.lists(
            st.integers(min_value=1, max_value=86_400_000),
            min_size=n,
            max_size=n,
        )
    )
    buffer: list[Candle] = []
    t = start
    for inc in increments:
        buffer.append(draw(candle_strategy(open_time_ms=t)))
        t += inc
    return buffer


# ---------------------------------------------------------------------------
# Independent reference computation (NOT calling compute_stochastic)
# ---------------------------------------------------------------------------
def reference_last_row_stochastic(buffer: list[Candle]) -> tuple[float, float]:
    """Independently recompute STOCH(5, 3, 3) and return the last-row (%K, %D).

    This is a from-scratch reimplementation using vanilla pandas rolling windows
    and the **configured** parameters (length=5, k_smooth=3, d_smooth=3). The
    smoothed series are attached to a DataFrame under the column names named
    *exactly* ``STOCHk_5_3_3`` / ``STOCHd_5_3_3`` (REQ-1.5) and the final values
    are read from the last row (REQ-5.2, REQ-5.3), so the property compares two
    independent derivations rather than the implementation against itself.
    """
    length = bsm.STOCH_K_LENGTH  # 5
    k_smooth = bsm.STOCH_K_SMOOTH  # 3
    d_smooth = bsm.STOCH_D_SMOOTH  # 3

    df = pd.DataFrame(
        {
            "high": [c.high for c in buffer],
            "low": [c.low for c in buffer],
            "close": [c.close for c in buffer],
        }
    )
    low_min = df["low"].rolling(window=length).min()
    high_max = df["high"].rolling(window=length).max()
    fast_k = 100 * ((df["close"] - low_min) / (high_max - low_min))
    stoch_k_series = fast_k.rolling(window=k_smooth).mean()
    stoch_d_series = stoch_k_series.rolling(window=d_smooth).mean()

    # Attach under the exact configured column names and read the last row.
    df[bsm.STOCH_K_COL] = stoch_k_series
    df[bsm.STOCH_D_COL] = stoch_d_series
    return float(df[bsm.STOCH_K_COL].iloc[-1]), float(df[bsm.STOCH_D_COL].iloc[-1])


# ---------------------------------------------------------------------------
# Property 3: Stochastic computation uses configured columns and parameters
# Validates: Requirements 1.5, 5.1, 5.2, 5.3
# ---------------------------------------------------------------------------
@settings(max_examples=100, deadline=None)
@given(buffer=buffer_strategy())
def test_property3_stochastic_uses_configured_columns_and_parameters(
    buffer: list[Candle],
) -> None:
    """**Property 3: Stochastic computation uses configured columns and parameters**

    **Validates: Requirements 1.5, 5.1, 5.2, 5.3**

    For any candle buffer (size 8..50) whose independent STOCH(5, 3, 3) last-row
    %K and %D are finite (non-NaN), ``compute_stochastic`` returns a
    ``StochasticReading`` whose ``k`` / ``d`` equal those independently-computed
    last-row values of the columns named exactly ``STOCHk_5_3_3`` /
    ``STOCHd_5_3_3``. The reading's ``close`` and ``close_time_ms`` are also
    derived from the most recent candle.
    """
    # REQ-1.5: the implementation reads the configured, exactly-named columns.
    assert bsm.STOCH_K_COL == "STOCHk_5_3_3"
    assert bsm.STOCH_D_COL == "STOCHd_5_3_3"

    # Independent reference using the configured length=5, k=3, d=3 (REQ-5.1).
    expected_k, expected_d = reference_last_row_stochastic(buffer)

    # The value-equality property only applies once warm-up has elapsed. NaN
    # last-row values (short / flat windows) are the "skip" case covered by
    # Property 4 (task 6.3), so discard those examples here.
    assume(not math.isnan(expected_k))
    assume(not math.isnan(expected_d))

    reading = compute_stochastic(buffer)

    # A finite reference last row must yield a numeric reading (not the None
    # warm-up/skip path).
    assert reading is not None, (
        "compute_stochastic returned None even though the independent reference "
        f"last row was finite: %K={expected_k}, %D={expected_d}"
    )

    # %K / %D equal the last-row values of STOCHk_5_3_3 / STOCHd_5_3_3
    # (REQ-5.2, REQ-5.3).
    assert reading.k == pytest.approx(expected_k)
    assert reading.d == pytest.approx(expected_d)

    # close / close_time_ms are taken from the most recent candle (REQ-5.6).
    assert reading.close == buffer[-1].close
    assert reading.close_time_ms == buffer[-1].open_time_ms + bsm.STALE_BUFFER_THRESHOLD_MS


# ---------------------------------------------------------------------------
# Property 4: NaN or missing columns suppress trigger evaluation
# Validates: Requirements 5.4
# ---------------------------------------------------------------------------
#
# REQ-5.4: "IF the latest %K value or the latest %D value is NaN, OR the column
# STOCHk_5_3_3 or the column STOCHd_5_3_3 is missing from the output, THEN THE
# Indicator_Engine SHALL skip trigger evaluation for the current Closed_Candle."
#
# In the NATIVE (vanilla pandas) implementation the columns are always created,
# so the realistic "skip" trigger is a NaN in the last row. Two distinct ways to
# force a NaN last row are exercised below:
#
#   1. Warm-up: STOCH(5, 3, 3) needs 5 + (3-1) + (3-1) = 9 candles before the
#      last-row %D is non-NaN. Any buffer of 1..8 finite candles yields a NaN
#      last row -> compute_stochastic returns None.
#   2. Flat window: a buffer of >= 9 candles whose last 5 candles share the same
#      high == low makes high_max == low_min over the %K window, so the
#      (close - low_min) / (high_max - low_min) division is 0/0 -> NaN -> None.
#
# The "skip trigger evaluation" contract is then demonstrated with the documented
# gating pattern: a mocked ``evaluate_trigger`` stand-in is invoked only when
# ``compute_stochastic`` returns a non-None reading, and is asserted to be never
# called when the reading is None.


# A buffer that is too short for STOCH(5, 3, 3) to produce a non-NaN last row.
# 9 candles are required (5 + 2 + 2); sizes 1..8 are all still in warm-up.
_WARMUP_MAX_SIZE = 8


@settings(max_examples=100, deadline=None)
@given(buffer=buffer_strategy(min_size=1, max_size=_WARMUP_MAX_SIZE))
def test_property4_warmup_short_buffer_returns_none(buffer: list[Candle]) -> None:
    """**Property 4: NaN or missing columns suppress trigger evaluation** (warm-up)

    **Validates: Requirements 5.4**

    For any candle buffer too short for STOCH(5, 3, 3) warm-up (1..8 finite
    candles), the last-row smoothed %K / %D are NaN, so ``compute_stochastic``
    returns ``None`` and trigger evaluation is skipped for the candle.
    """
    # Sanity: these buffers are genuinely below the 9-candle warm-up boundary.
    assert len(buffer) <= _WARMUP_MAX_SIZE

    reading = compute_stochastic(buffer)

    assert reading is None, (
        "compute_stochastic must return None for a warm-up-length buffer "
        f"(len={len(buffer)}) whose last-row %K/%D are NaN (REQ-5.4)"
    )

    # Demonstrate the gate (REQ-5.4): a mocked evaluate_trigger stand-in is only
    # invoked when the reading is non-None. With reading is None it must NOT be
    # called, i.e. trigger evaluation is skipped for the candle.
    mock_evaluate_trigger = MagicMock()
    if reading is not None:
        mock_evaluate_trigger(reading)
    mock_evaluate_trigger.assert_not_called()


def _flat_last_window_buffer(total: int = 12, window: int = bsm.STOCH_K_LENGTH) -> list[Candle]:
    """Build a buffer of ``total`` candles whose final ``window`` candles are flat.

    The last ``window`` candles all share an identical ``high == low`` price, so
    over the trailing %K window ``high_max == low_min`` and the fast %K division
    is ``0 / 0`` -> NaN. With ``total >= 9`` the buffer is past warm-up, isolating
    the flat-window NaN (rather than the warm-up NaN) as the cause of the skip.
    """
    candles: list[Candle] = []
    # Leading, non-degenerate candles (varied highs/lows) so the buffer is well
    # past the 9-candle warm-up boundary.
    for i in range(total - window):
        base = 100.0 + i
        candles.append(
            Candle(
                open_time_ms=i,
                open=base,
                high=base + 5.0,
                low=base - 5.0,
                close=base + 1.0,
                volume=10.0,
            )
        )
    # Trailing flat window: identical high == low == close makes the rolling
    # high_max == low_min over the last %K window, forcing a 0/0 NaN last row.
    flat_price = 500.0
    for j in range(window):
        idx = (total - window) + j
        candles.append(
            Candle(
                open_time_ms=idx,
                open=flat_price,
                high=flat_price,
                low=flat_price,
                close=flat_price,
                volume=10.0,
            )
        )
    return candles


def test_property4_flat_window_returns_none() -> None:
    """**Property 4: NaN or missing columns suppress trigger evaluation** (flat window)

    **Validates: Requirements 5.4**

    For a buffer past warm-up (>= 9 candles) whose final %K-length window is flat
    (``high == low`` for every candle in the window), the fast-%K denominator
    ``high_max - low_min`` is 0, so the last-row %K / %D are NaN and
    ``compute_stochastic`` returns ``None``.
    """
    buffer = _flat_last_window_buffer(total=12)
    # Past warm-up: the None here is due to the flat window, not too few candles.
    assert len(buffer) >= 9

    reading = compute_stochastic(buffer)

    assert reading is None, (
        "compute_stochastic must return None when the trailing %K window is flat "
        "(high_max == low_min -> 0/0 NaN), suppressing trigger evaluation (REQ-5.4)"
    )

    # Gate demonstration: reading is None, so evaluate_trigger is never invoked.
    mock_evaluate_trigger = MagicMock()
    if reading is not None:
        mock_evaluate_trigger(reading)
    mock_evaluate_trigger.assert_not_called()


def test_property4_missing_column_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """**Property 4: NaN or missing columns suppress trigger evaluation** (missing column)

    **Validates: Requirements 5.4**

    Exercises the "column missing from output" skip branch of
    ``compute_stochastic``. The native implementation always assigns the
    configured columns, so to reach the ``STOCH_K_COL not in df.columns`` guard
    we patch ``bsm.pd.DataFrame`` with a thin subclass whose ``__setitem__``
    silently drops any attempt to create the ``STOCH_K_COL`` column. The
    indicator's computed DataFrame therefore lacks the expected %K column,
    ``compute_stochastic`` returns ``None``, and trigger evaluation is skipped.
    """

    class _DropKColumnDataFrame(pd.DataFrame):
        """DataFrame that refuses to store the configured %K column.

        ``__setitem__`` ignores writes whose key equals the (current) configured
        :data:`STOCH_K_COL`, so after ``compute_stochastic`` assigns its results
        the %K column is absent from the frame, driving the missing-column guard.
        """

        # Required so pandas operations that construct new frames return the
        # base class rather than this test-only subclass.
        @property
        def _constructor(self):  # type: ignore[override]
            return pd.DataFrame

        def __setitem__(self, key, value):  # type: ignore[override]
            if key == bsm.STOCH_K_COL:
                return  # drop the %K column write so the column stays missing
            super().__setitem__(key, value)

    monkeypatch.setattr(bsm.pd, "DataFrame", _DropKColumnDataFrame)

    # A buffer comfortably past warm-up so the only reason for None is the
    # missing %K column, not a NaN warm-up row.
    buffer = [
        Candle(
            open_time_ms=i,
            open=100.0 + i,
            high=110.0 + i,
            low=90.0 + i,
            close=100.0 + (i % 3),
            volume=10.0,
        )
        for i in range(15)
    ]

    reading = compute_stochastic(buffer)

    assert reading is None, (
        "compute_stochastic must return None when the configured %K column is "
        "absent from the computed output (REQ-5.4)"
    )

    # Gate demonstration: reading is None, so evaluate_trigger is never invoked.
    mock_evaluate_trigger = MagicMock()
    if reading is not None:
        mock_evaluate_trigger(reading)
    mock_evaluate_trigger.assert_not_called()


def test_property4_gate_invokes_trigger_when_reading_present() -> None:
    """Control: when ``compute_stochastic`` DOES return a reading, the gate fires.

    Confirms the gating pattern used in the Property 4 assertions is meaningful:
    given a healthy, past-warm-up buffer with non-degenerate prices,
    ``compute_stochastic`` returns a non-``None`` reading and the mocked
    ``evaluate_trigger`` stand-in IS invoked exactly once with that reading. Without
    this control the ``assert_not_called`` checks could pass vacuously.
    """
    # A varied, past-warm-up buffer that yields finite %K / %D.
    buffer = [
        Candle(
            open_time_ms=i,
            open=100.0 + (i % 7),
            high=110.0 + (i % 5),
            low=90.0 - (i % 3),
            close=95.0 + (i % 11),
            volume=10.0,
        )
        for i in range(20)
    ]

    reading = compute_stochastic(buffer)
    assert reading is not None, "control buffer should yield a finite reading"

    mock_evaluate_trigger = MagicMock()
    if reading is not None:
        mock_evaluate_trigger(reading)
    mock_evaluate_trigger.assert_called_once_with(reading)
