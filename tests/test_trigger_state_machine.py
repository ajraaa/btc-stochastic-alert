"""Property-based tests for the cool-down trigger state machine (``evaluate_trigger``).

This module hosts Property 5 for the BTC Stochastic DCA Monitor: the cool-down
state machine fires *exactly once per oversold episode* and re-arms only after
%K rises strictly above the oversold threshold.

The module under test is ``btc_stochastic_monitor`` which exposes
``evaluate_trigger(state, reading) -> TriggerDecision`` with the (pure, except
for the ``state.is_oversold`` mutation) semantics:

* not is_oversold and k < 20  -> is_oversold=True, FIRE
* is_oversold and k <= 20      -> SUPPRESS (no change)
* is_oversold and k > 20       -> is_oversold=False, RESET
* otherwise                    -> QUIET (no change)

Two complementary tests are provided:

* :class:`CoolDownStateMachine` -- a Hypothesis ``RuleBasedStateMachine`` whose
  rule pushes a generated ``(k, d, close)`` reading and accumulates the
  resulting decisions, asserting the three episode invariants (Property 5).
* :func:`test_boundary_transitions_direct` -- a focused, non-stateful
  ``@given``/``@example`` test that pins the exact boundary behaviour at
  ``19.999``, ``20.0``, and ``20.001`` from both the engaged and disengaged
  cool-down states.

A handful of plain unit tests pin a canonical multi-episode decision sequence.
"""

from __future__ import annotations

from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from btc_stochastic_monitor import (
    OVERSOLD_THRESHOLD,
    MonitorState,
    StochasticReading,
    TriggerDecision,
    evaluate_trigger,
)


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------
# %K is drawn from a band that straddles the oversold threshold (20) from both
# sides, plus the exact boundary values 19.999 / 20.0 / 20.001 so the generated
# domain always includes the cases that distinguish strict ``<`` engagement from
# ``<=`` suppression and strict ``>`` release.
_k_strategy = st.one_of(
    st.floats(min_value=0.0, max_value=40.0, allow_nan=False, allow_infinity=False),
    st.sampled_from([19.999, 20.0, 20.001]),
)

# ``d`` and ``close`` do not influence the decision; any finite float is valid.
_finite_floats = st.floats(allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property 5: Cool-down state machine fires exactly once per oversold episode
# Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5
# ---------------------------------------------------------------------------
class CoolDownStateMachine(RuleBasedStateMachine):
    """**Property 5: Cool-down state machine fires exactly once per oversold episode**

    **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**

    A single rule pushes a generated ``(k, d, close)`` reading through
    ``evaluate_trigger`` and accumulates the resulting decisions. The machine
    maintains an independent reference model of ``is_oversold`` and the decision
    history to assert:

    * Invariant 1: between any two ``FIRE`` decisions there is at least one
      ``RESET`` (a decision following a reading with ``k > OVERSOLD_THRESHOLD``).
    * Invariant 2: a contiguous run of readings with ``k <= 20`` produces at most
      one ``FIRE`` (enforced by the ``fired_since_reset`` latch below).
    * Invariant 3: a reading with ``k == 20`` never produces ``FIRE`` and never
      produces ``RESET``.

    The boundary inputs ``19.999`` / ``20.0`` / ``20.001`` are guaranteed to be
    in the generated domain (see ``_k_strategy``).
    """

    def __init__(self) -> None:
        super().__init__()
        # System under test: an empty buffer is fine -- evaluate_trigger never
        # reads it, only state.is_oversold which starts False (REQ-6.1).
        self.state = MonitorState(buffer=[], is_oversold=False)
        # Independent reference model of the cool-down lock.
        self.ref_oversold = False
        # Latch enforcing "at most one FIRE per oversold episode" (Invariants 1+2):
        # set on FIRE, cleared on RESET; a second FIRE before a RESET fails.
        self.fired_since_reset = False
        # Full (k, decision) history for the independent cross-check invariant.
        self.decisions: list[tuple[float, TriggerDecision]] = []

    @rule(k=_k_strategy, d=_finite_floats, close=_finite_floats)
    def push_reading(self, k: float, d: float, close: float) -> None:
        reading = StochasticReading(k=k, d=d, close=close, close_time_ms=0)

        # --- Independent reference state machine over (ref_oversold, k) -------
        if not self.ref_oversold and k < OVERSOLD_THRESHOLD:
            expected = TriggerDecision.FIRE
            expected_oversold = True
        elif self.ref_oversold and k <= OVERSOLD_THRESHOLD:
            expected = TriggerDecision.SUPPRESS
            expected_oversold = True
        elif self.ref_oversold and k > OVERSOLD_THRESHOLD:
            expected = TriggerDecision.RESET
            expected_oversold = False
        else:
            expected = TriggerDecision.QUIET
            expected_oversold = False  # only reachable when ref_oversold is False

        oversold_before = self.ref_oversold
        decision = evaluate_trigger(self.state, reading)

        # The decision and the state mutation match the reference model
        # (REQ-6.2, REQ-6.3, REQ-6.4).
        assert decision == expected, (
            f"k={k!r} is_oversold(before)={oversold_before}: "
            f"expected {expected} but got {decision}"
        )
        assert self.state.is_oversold == expected_oversold
        self.ref_oversold = expected_oversold

        # --- Invariant 3: the exact boundary k == 20 never fires/resets ------
        if k == OVERSOLD_THRESHOLD:
            assert decision is not TriggerDecision.FIRE, (
                "k == 20 must not FIRE (engagement uses strict <)"
            )
            assert decision is not TriggerDecision.RESET, (
                "k == 20 must not RESET (release uses strict >)"
            )

        # --- Invariants 1 + 2: one FIRE per episode, RESET re-arms -----------
        if decision is TriggerDecision.FIRE:
            assert not self.fired_since_reset, (
                "two FIRE decisions without an intervening RESET (k > 20)"
            )
            self.fired_since_reset = True
        elif decision is TriggerDecision.RESET:
            self.fired_since_reset = False

        self.decisions.append((k, decision))

    @invariant()
    def fire_requires_intervening_reset(self) -> None:
        """Independent cross-check of Invariants 1 + 2 from the full history.

        Re-derives, from the accumulated decision sequence, that no two ``FIRE``
        decisions occur without a ``RESET`` between them -- equivalently, every
        contiguous oversold episode yields at most one ``FIRE``.
        """
        seen_fire = False
        for _k, decision in self.decisions:
            if decision is TriggerDecision.FIRE:
                assert not seen_fire, (
                    "history contains two FIRE decisions with no RESET between them"
                )
                seen_fire = True
            elif decision is TriggerDecision.RESET:
                seen_fire = False


# Run the state machine at the required iteration count with no deadline so
# slow CI machines don't flake (mirrors the ``btc_monitor`` Hypothesis profile).
CoolDownStateMachine.TestCase.settings = settings(max_examples=100, deadline=None)
TestCoolDownStateMachine = CoolDownStateMachine.TestCase


# ---------------------------------------------------------------------------
# Focused boundary test (non-stateful) with explicit @example coverage
# Validates: Requirements 6.2, 6.3, 6.4
# ---------------------------------------------------------------------------
@settings(max_examples=100, deadline=None)
@given(k=st.floats(min_value=0.0, max_value=40.0, allow_nan=False, allow_infinity=False))
@example(k=19.999)
@example(k=20.0)
@example(k=20.001)
def test_boundary_transitions_direct(k: float) -> None:
    """**Property 5 (boundary): strict ``<`` engagement and strict ``>`` release**

    **Validates: Requirements 6.2, 6.3, 6.4**

    Asserts, for any %K (with the three exact boundary values pinned via
    ``@example``), the decision and resulting lock state from BOTH the
    disengaged (``is_oversold=False``) and engaged (``is_oversold=True``)
    starting states. In particular this pins:

    * ``k = 19.999`` from not-oversold -> ``FIRE``
    * ``k = 20.0`` from not-oversold   -> ``QUIET`` (never ``FIRE``)
    * ``k = 20.0`` from oversold       -> ``SUPPRESS`` (never ``RESET``)
    * ``k = 20.001`` from oversold     -> ``RESET``
    * ``k = 20.001`` from not-oversold -> ``QUIET``
    """
    reading = StochasticReading(k=k, d=0.0, close=0.0, close_time_ms=0)

    # --- From a disengaged cool-down lock (is_oversold = False) --------------
    state_not = MonitorState(buffer=[], is_oversold=False)
    decision_from_not = evaluate_trigger(state_not, reading)
    if k < OVERSOLD_THRESHOLD:
        # Engagement uses strict <: k strictly below 20 fires and locks.
        assert decision_from_not is TriggerDecision.FIRE
        assert state_not.is_oversold is True
    else:
        # k >= 20 from a disengaged lock is no oversold condition.
        assert decision_from_not is TriggerDecision.QUIET
        assert state_not.is_oversold is False
    # A disengaged lock can only FIRE or stay QUIET -- never SUPPRESS/RESET.
    assert decision_from_not in (TriggerDecision.FIRE, TriggerDecision.QUIET)

    # --- From an engaged cool-down lock (is_oversold = True) -----------------
    state_over = MonitorState(buffer=[], is_oversold=True)
    decision_from_over = evaluate_trigger(state_over, reading)
    if k <= OVERSOLD_THRESHOLD:
        # Suppression uses <=: at or below 20 stays silent and locked.
        assert decision_from_over is TriggerDecision.SUPPRESS
        assert state_over.is_oversold is True
    else:
        # Release uses strict >: strictly above 20 re-arms the lock.
        assert decision_from_over is TriggerDecision.RESET
        assert state_over.is_oversold is False
    # An engaged lock can only SUPPRESS or RESET -- never FIRE again.
    assert decision_from_over in (TriggerDecision.SUPPRESS, TriggerDecision.RESET)

    # --- Invariant 3: the exact boundary k == 20 ----------------------------
    if k == OVERSOLD_THRESHOLD:
        assert decision_from_not is TriggerDecision.QUIET  # never FIRE at k == 20
        assert decision_from_over is TriggerDecision.SUPPRESS  # never RESET at k == 20


# ---------------------------------------------------------------------------
# Canonical unit examples (specific multi-episode sequences)
# ---------------------------------------------------------------------------
def _run_sequence(ks: list[float]) -> list[TriggerDecision]:
    """Feed a list of %K values through a single MonitorState; return decisions."""
    state = MonitorState(buffer=[], is_oversold=False)
    out: list[TriggerDecision] = []
    for k in ks:
        out.append(
            evaluate_trigger(state, StochasticReading(k=k, d=0.0, close=0.0, close_time_ms=0))
        )
    return out


def test_one_fire_per_episode_two_episodes() -> None:
    """A two-episode trajectory fires exactly once per oversold episode."""
    decisions = _run_sequence([25.0, 15.0, 10.0, 18.0, 30.0, 12.0])
    assert decisions == [
        TriggerDecision.QUIET,     # 25: not oversold, k >= 20
        TriggerDecision.FIRE,      # 15: engage, fire (episode 1)
        TriggerDecision.SUPPRESS,  # 10: oversold, k <= 20
        TriggerDecision.SUPPRESS,  # 18: oversold, k <= 20
        TriggerDecision.RESET,     # 30: oversold, k > 20 -> release
        TriggerDecision.FIRE,      # 12: engage, fire (episode 2)
    ]
    # Exactly two FIRE decisions, separated by a RESET.
    assert decisions.count(TriggerDecision.FIRE) == 2


def test_boundary_run_at_twenty_never_fires() -> None:
    """A contiguous run sitting exactly at the threshold never fires or resets."""
    decisions = _run_sequence([20.0, 20.0, 20.0])
    assert decisions == [TriggerDecision.QUIET, TriggerDecision.QUIET, TriggerDecision.QUIET]
    assert TriggerDecision.FIRE not in decisions
    assert TriggerDecision.RESET not in decisions


def test_single_oversold_run_fires_once_then_suppresses() -> None:
    """A single contiguous oversold run produces exactly one FIRE (Invariant 2)."""
    decisions = _run_sequence([19.999, 5.0, 0.0, 20.0, 19.0])
    assert decisions[0] is TriggerDecision.FIRE
    assert all(d is TriggerDecision.SUPPRESS for d in decisions[1:])
    assert decisions.count(TriggerDecision.FIRE) == 1
