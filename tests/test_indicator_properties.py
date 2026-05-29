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
