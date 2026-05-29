"""Property-based tests for the candle buffer (``update_buffer``).

This module hosts Property 1 for the BTC Stochastic DCA Monitor.

* Property 1 (this file): for any initial candle buffer and any finite sequence
  of ``update_buffer(candle)`` operations, after each operation the buffer
  satisfies the bounded-length, strict-ascending, unique-open-time invariants,
  and the operation's effect matches its case (append / replace / discard).

The property is exercised with a Hypothesis ``RuleBasedStateMachine``: one rule
generates a ``Candle`` -- with ``open_time_ms`` drawn from a small integer domain
so collisions (replace) and older-than-latest times (discard) occur frequently
alongside strictly-newer times (append) -- and applies ``update_buffer`` to a
buffer accumulated across rule invocations. After each step the test compares
the real buffer against an independently-derived oracle and re-checks the
buffer invariants, matching "after each operation" in the property statement.

A handful of focused unit tests cover the three classification cases and the
trim-to-``HISTORY_LIMIT`` boundary explicitly, complementing the property test.
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

import btc_stochastic_monitor as bsm
from btc_stochastic_monitor import Candle, HISTORY_LIMIT, update_buffer

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Finite OHLCV component. The buffer logic keys solely on ``open_time_ms``, so
# the price/volume fields only need to be finite, distinguishable values; bounds
# keep them comfortably away from NaN/inf without constraining the search space
# in a way that matters for this property.
finite_floats = st.floats(
    allow_nan=False,
    allow_infinity=False,
    min_value=-1e9,
    max_value=1e9,
)

# ``open_time_ms`` is drawn from a small integer domain (0..30). A narrow domain
# is deliberate: it makes same-open-time collisions (replace) and
# earlier-than-latest arrivals (discard) common rather than astronomically rare,
# so a single run exercises all three operation cases many times.
open_time_strategy = st.integers(min_value=0, max_value=30)

candle_strategy = st.builds(
    Candle,
    open_time_ms=open_time_strategy,
    open=finite_floats,
    high=finite_floats,
    low=finite_floats,
    close=finite_floats,
    volume=finite_floats,
)


# ---------------------------------------------------------------------------
# Independent oracle (reference model)
# ---------------------------------------------------------------------------
def classify_and_expect(
    buffer: list[Candle], candle: Candle
) -> tuple[str, list[Candle]]:
    """Independently classify ``candle`` against ``buffer`` and derive the result.

    This is a from-scratch reimplementation of the buffer semantics described by
    REQ-4.1, REQ-4.2, REQ-4.4, REQ-4.5 -- intentionally NOT calling
    ``update_buffer`` -- so the property test compares two independent
    derivations rather than the implementation against itself.

    Returns a ``(classification, expected_buffer)`` tuple where ``classification``
    is one of ``"replace"``, ``"append"`` or ``"discard"`` and ``expected_buffer``
    is a fresh list (the input ``buffer`` is never mutated).
    """
    open_times = [c.open_time_ms for c in buffer]

    # Replace (REQ-4.4): a matching open time exists -> overwrite in place; the
    # length and ordering are unchanged. Checked first because a matching open
    # time takes precedence over the "earlier than latest" discard rule.
    if candle.open_time_ms in open_times:
        idx = open_times.index(candle.open_time_ms)
        expected = list(buffer)
        expected[idx] = candle
        return "replace", expected

    # Append (REQ-4.1): strictly newer than every existing candle (vacuously
    # true for an empty buffer). Append then drop leading (oldest) entries until
    # the bounded-length invariant holds (REQ-4.2).
    if not open_times or candle.open_time_ms > max(open_times):
        expected = list(buffer) + [candle]
        while len(expected) > HISTORY_LIMIT:
            expected = expected[1:]
        return "append", expected

    # Discard (REQ-4.5): earlier than the latest open time with no matching
    # entry -> buffer unchanged.
    return "discard", list(buffer)


def _assert_buffer_invariants(buffer: list[Candle]) -> None:
    """Assert the three structural invariants that must hold after every op."""
    times = [c.open_time_ms for c in buffer]

    # (a) Bounded length (REQ-1.6, REQ-4.2).
    assert len(buffer) <= HISTORY_LIMIT, f"buffer exceeded HISTORY_LIMIT: {len(buffer)}"

    # (b) Strict ascending order by open_time_ms (REQ-4.3). Strict ascending
    # also implies (c) uniqueness, but uniqueness is asserted separately for a
    # clearer failure message.
    assert all(
        times[i] < times[i + 1] for i in range(len(times) - 1)
    ), f"open times not strictly ascending: {times}"

    # (c) Unique open times (REQ-4.4 replacement semantics).
    assert len(set(times)) == len(times), f"duplicate open times present: {times}"


# ---------------------------------------------------------------------------
# Property 1: Buffer invariants under any update sequence
# Validates: Requirements 1.6, 4.1, 4.2, 4.3, 4.4, 4.5
# ---------------------------------------------------------------------------
class BufferStateMachine(RuleBasedStateMachine):
    """Drive ``update_buffer`` with a generated sequence of candles.

    **Property 1: Buffer invariants under any update sequence**

    **Validates: Requirements 1.6, 4.1, 4.2, 4.3, 4.4, 4.5**

    A single rule generates a ``Candle`` and applies ``update_buffer`` to the
    buffer accumulated so far. After each application the test asserts (1) the
    real buffer equals the independently-derived oracle buffer, and (2) the
    operation's effect matches its classification (append / replace / discard).
    The structural invariants are re-checked after every step via the
    ``@invariant`` method, satisfying the "after each operation" quantifier.
    """

    def __init__(self) -> None:
        super().__init__()
        # The buffer under test, accumulated across rule invocations. Starts
        # empty; the very first append exercises the empty-buffer branch.
        self.buffer: list[Candle] = []

    @rule(candle=candle_strategy)
    def apply_update(self, candle: Candle) -> None:
        # Snapshot the pre-update buffer so the oracle classifies against the
        # exact state the implementation sees (update_buffer mutates in place).
        before = list(self.buffer)
        classification, expected = classify_and_expect(before, candle)

        result = update_buffer(self.buffer, candle)

        # update_buffer mutates and returns the same list object.
        assert result is self.buffer, "update_buffer must return the buffer it mutated"

        # (d) The operation matches its expected case: the real buffer equals
        # the buffer derived independently by the oracle.
        assert result == expected, (
            f"{classification} mismatch\n"
            f"  before={before}\n  candle={candle}\n"
            f"  expected={expected}\n  actual={result}"
        )

        # Per-case structural effects (REQ-4.1, REQ-4.2, REQ-4.4, REQ-4.5).
        if classification == "replace":
            # Length unchanged; same set of open times; the slot now holds candle.
            assert len(result) == len(before)
            assert {c.open_time_ms for c in result} == {c.open_time_ms for c in before}
            assert candle in result
        elif classification == "append":
            # The newest candle lands at the end; length grows by one until the
            # bound is hit, then stays pinned at HISTORY_LIMIT.
            assert result[-1] == candle
            assert len(result) == min(len(before) + 1, HISTORY_LIMIT)
        else:  # discard
            # Buffer untouched.
            assert result == before

    @invariant()
    def structural_invariants_hold(self) -> None:
        # Re-checked after every executed rule (REQ-1.6, REQ-4.2, REQ-4.3, REQ-4.4).
        _assert_buffer_invariants(self.buffer)


# Bind the state machine to a pytest-collected TestCase and apply the required
# settings (>= 100 examples, no deadline). The conftest also loads the
# ``btc_monitor`` profile; the explicit settings here make the contract local
# and independent of profile-load ordering.
BufferStateMachine.TestCase.settings = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
TestProperty1BufferInvariants = BufferStateMachine.TestCase


# ---------------------------------------------------------------------------
# Focused unit tests for each classification case (complement the property)
# ---------------------------------------------------------------------------
def _candle(open_time_ms: int, close: float = 1.0) -> Candle:
    """Build a Candle with a given open time; OHLCV values are placeholders."""
    return Candle(
        open_time_ms=open_time_ms,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def test_append_when_strictly_newer():
    buffer = [_candle(1), _candle(2)]
    result = update_buffer(buffer, _candle(3))
    assert [c.open_time_ms for c in result] == [1, 2, 3]


def test_append_to_empty_buffer():
    result = update_buffer([], _candle(10))
    assert [c.open_time_ms for c in result] == [10]


def test_replace_keeps_length_and_overwrites_slot():
    buffer = [_candle(1, close=1.0), _candle(2, close=2.0)]
    replacement = _candle(2, close=99.0)
    result = update_buffer(buffer, replacement)
    assert len(result) == 2
    assert [c.open_time_ms for c in result] == [1, 2]
    assert result[1].close == 99.0


def test_discard_when_older_with_no_match():
    buffer = [_candle(5), _candle(6), _candle(7)]
    before = list(buffer)
    result = update_buffer(buffer, _candle(3))
    assert result == before


def test_trim_drops_oldest_when_exceeding_history_limit():
    # Fill exactly to the limit, then append one more; the oldest leading entry
    # is dropped and the buffer stays pinned at HISTORY_LIMIT (REQ-4.2).
    buffer = [_candle(i) for i in range(HISTORY_LIMIT)]
    result = update_buffer(buffer, _candle(HISTORY_LIMIT))
    assert len(result) == HISTORY_LIMIT
    assert result[0].open_time_ms == 1  # original oldest (0) was dropped
    assert result[-1].open_time_ms == HISTORY_LIMIT
