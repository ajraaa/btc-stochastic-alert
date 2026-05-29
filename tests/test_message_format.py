"""Property-based test for the Telegram alert message formatter (``format_alert``).

This module hosts **Property 7** for the BTC Stochastic DCA Monitor: every alert
message built by ``format_alert`` surfaces the fields a trader needs to act on an
oversold signal -- the trading symbol, the kline interval, the latest %K and %D
each rendered with exactly two decimal places, and the closing price rendered as
a numeric value (REQ-7.2).

The module under test is ``btc_stochastic_monitor`` which exposes::

    format_alert(symbol: str, interval: str, reading: StochasticReading) -> str

Its concrete rendering (verified against the implementation) is::

    "BTC Stochastic DCA alert: {symbol} @ {interval} is oversold. "
    "%K={reading.k:.2f} %D={reading.d:.2f} close={reading.close}"

so ``k`` and ``d`` use the two-decimal ``"{:.2f}"`` pattern while ``close`` is
embedded via default float-to-string conversion (equivalently ``f"{close}"`` /
``str(close)``). The substring assertions below derive their expected fragments
using those *same* formatting rules rather than hard-coded literals, so the test
stays robust to floating-point rounding (e.g. Python banker's rounding on
``19.995``) and to scientific-notation rendering of ``close``.
"""

from __future__ import annotations

from hypothesis import example, given, settings
from hypothesis import strategies as st

from btc_stochastic_monitor import (
    INTERVAL,
    SYMBOL,
    StochasticReading,
    format_alert,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
# Finite floats bounded to a reasonable price/oscillator range. The bound keeps
# generated values in everyday magnitudes; correctness does NOT depend on it,
# because every expected substring below is derived with the same formatting the
# implementation uses, so even scientific-notation ``close`` values match.
_finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)

_reading_strategy = st.builds(
    StochasticReading,
    k=_finite,
    d=_finite,
    close=_finite,
    close_time_ms=st.integers(min_value=0, max_value=2_000_000_000_000),
)


# ---------------------------------------------------------------------------
# Property 7: Alert message contains required fields with two-decimal formatting
# Validates: Requirements 7.2
# ---------------------------------------------------------------------------
@settings(max_examples=100, deadline=None)
@given(reading=_reading_strategy)
# Edge case: k = 19.995 -- the nearest double is 19.99499999..., so
# ``f"{19.995:.2f}"`` rounds to "19.99" (banker's rounding on the stored value).
# The assertion compares against ``f"{reading.k:.2f}"`` computed the same way, so
# it is correct whichever way the rounding lands -- never a hard-coded "20.00".
@example(reading=StochasticReading(k=19.995, d=55.5, close=42000.0, close_time_ms=0))
# Edge case: d = 0.0 -- must render "0.00".
@example(reading=StochasticReading(k=12.34, d=0.0, close=30000.5, close_time_ms=0))
def test_property7_alert_contains_required_fields(reading: StochasticReading) -> None:
    """**Property 7: Alert message contains required fields with two-decimal formatting**

    **Validates: Requirements 7.2**

    For any ``StochasticReading`` with finite floats, ``format_alert`` SHALL
    produce a message containing (a) the symbol, (b) the interval, (c) ``k`` with
    exactly two decimal digits, (d) ``d`` with exactly two decimal digits, and
    (e) ``close`` rendered as a numeric value.
    """
    message = format_alert(SYMBOL, INTERVAL, reading)

    # (a) symbol and (b) interval appear verbatim.
    assert SYMBOL in message
    assert INTERVAL in message

    # (c) %K and (d) %D rendered with exactly two decimals. Expected substrings
    # are computed with the implementation's own "{:.2f}" rule so the assertion
    # is robust to floating-point rounding at boundaries like 19.995.
    assert f"{reading.k:.2f}" in message
    assert f"{reading.d:.2f}" in message

    # (e) close rendered as a numeric value -- matched against the implementation's
    # default float rendering (``f"{close}"`` == ``str(close)``).
    assert f"{reading.close}" in message


# ---------------------------------------------------------------------------
# Focused unit examples (complement the property with concrete expectations)
# ---------------------------------------------------------------------------
def test_two_decimal_rendering_exact() -> None:
    """A canonical reading renders %K and %D with exactly two decimals."""
    reading = StochasticReading(k=12.3, d=45.678, close=42000.0, close_time_ms=0)
    message = format_alert(SYMBOL, INTERVAL, reading)

    assert "%K=12.30" in message  # 12.3 -> two decimals
    assert "%D=45.68" in message  # 45.678 -> rounded to two decimals
    assert "btcusdt" in message
    assert "1d" in message
    assert "close=42000.0" in message


def test_zero_d_renders_two_decimals() -> None:
    """``d = 0.0`` renders as "0.00" (edge case from the spec)."""
    reading = StochasticReading(k=5.0, d=0.0, close=10.0, close_time_ms=0)
    message = format_alert(SYMBOL, INTERVAL, reading)

    assert "%D=0.00" in message
    assert "%K=5.00" in message
