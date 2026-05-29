"""Property-based and unit tests for WebSocket message routing.

``btc_stochastic_monitor.on_message(state, ws, raw)`` is the WebSocket_Client
message handler. It parses the raw payload as JSON, extracts the ``"k"`` kline
object, and forwards a :class:`Candle` to the module-level ``process_data``
function if and only if the payload is valid JSON containing a ``"k"`` object
whose ``"x"`` (is-closed) field is ``True``. Every other payload shape
(malformed JSON, missing ``"k"``, missing ``"k.x"``, or ``"x" == False``) is
ignored without forwarding.

This file hosts:

* **Property 6 (task 11.2): WebSocket forwards iff candle is closed and
  preserves fields.** **Validates: Requirements 3.3, 3.4, 3.5, 3.6, 3.7.**

The strategy in this module produces a *labeled union* of payload categories.
``process_data`` is replaced with a spy (via ``monkeypatch.setattr`` /
``mock.patch``) so ``on_message`` itself performs no buffer mutation; the test
then asserts the biconditional:

  ``process_data`` is invoked  <=>  payload is valid JSON with ``"k.x" == True``

and, for the closed-valid category, that the forwarded ``Candle``'s six fields
equal ``int(t)`` / ``float(o, h, l, c, v)`` of the source kline. For EVERY
category the test also asserts that ``on_message`` leaves ``state.buffer`` and
``state.is_oversold`` unchanged (the real buffer mutation is delegated to the
spied-out ``process_data``).
"""

from __future__ import annotations

import json
from unittest import mock

from hypothesis import given, settings
from hypothesis import strategies as st

import btc_stochastic_monitor as bsm
from btc_stochastic_monitor import Candle, MonitorState

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Finite numeric kline field. ``on_message`` calls ``int(k["t"])`` and
# ``float(k[...])`` on the o/h/l/c/v fields. Plain ints/floats round-trip
# exactly through ``json.dumps`` / ``json.loads`` (json uses the shortest
# round-tripping repr), so exact-equality assertions on the forwarded Candle
# fields are sound.
finite_floats = st.floats(
    allow_nan=False,
    allow_infinity=False,
    min_value=-1e9,
    max_value=1e9,
)

# Non-negative open time in epoch milliseconds.
open_time_ms = st.integers(min_value=0, max_value=2_000_000_000_000)


@st.composite
def kline_fields(draw):
    """Draw the six raw kline field values ``(t, o, h, l, c, v)``.

    ``t`` is a non-negative integer (epoch ms); ``o/h/l/c/v`` are finite floats.
    These map onto Binance's kline object keys and are the values
    ``on_message`` parses via ``int(k["t"])`` and ``float(...)``.
    """
    t = draw(open_time_ms)
    o = draw(finite_floats)
    h = draw(finite_floats)
    low = draw(finite_floats)
    c = draw(finite_floats)
    v = draw(finite_floats)
    return t, o, h, low, c, v


def _kline_obj(fields, *, x):
    """Build a Binance-style ``"k"`` kline object dict from drawn fields.

    ``x`` is the is-closed flag placed under key ``"x"``.
    """
    t, o, h, low, c, v = fields
    return {"t": t, "o": o, "h": h, "l": low, "c": c, "v": v, "x": x}


# A "category" is (label, raw_string, fields_or_None). ``fields`` is populated
# only for the closed-valid category so the test can assert field preservation.
@st.composite
def ws_payloads(draw):
    """Generate a labeled WebSocket payload across all routing categories.

    Returns a ``(label, raw, fields)`` tuple where ``raw`` is the string handed
    to ``on_message``:

    * ``"closed_valid"`` -- valid JSON, ``"k"`` present, ``"x": true``. The only
      category for which ``process_data`` MUST be invoked. ``fields`` carries
      the source ``(t, o, h, l, c, v)`` for field-preservation assertions
      (REQ-3.3, REQ-3.5).
    * ``"non_closed"`` -- valid JSON, ``"k"`` present, ``"x": false`` -> NOT
      forwarded (REQ-3.4).
    * ``"missing_k"`` -- valid JSON object lacking ``"k"`` -> NOT forwarded
      (REQ-3.7).
    * ``"missing_kx"`` -- valid JSON, ``"k"`` present but lacking ``"x"`` -> NOT
      forwarded (REQ-3.7).
    * ``"malformed_json"`` -- a body that is not valid JSON -> NOT forwarded
      (REQ-3.6).
    """
    label = draw(
        st.sampled_from(
            [
                "closed_valid",
                "non_closed",
                "missing_k",
                "missing_kx",
                "malformed_json",
            ]
        )
    )
    fields = draw(kline_fields())

    if label == "closed_valid":
        raw = json.dumps({"e": "kline", "k": _kline_obj(fields, x=True)})
        return label, raw, fields

    if label == "non_closed":
        raw = json.dumps({"e": "kline", "k": _kline_obj(fields, x=False)})
        return label, raw, None

    if label == "missing_k":
        # A valid JSON object with no kline object. Vary between an event-only
        # object and an empty object.
        payload = draw(st.sampled_from([{"e": "kline"}, {}]))
        return label, json.dumps(payload), None

    if label == "missing_kx":
        # "k" present but with the "x" flag removed entirely.
        k = _kline_obj(fields, x=True)
        del k["x"]
        raw = json.dumps({"e": "kline", "k": k})
        return label, raw, None

    # malformed_json: a body that json.loads cannot parse.
    raw = draw(
        st.sampled_from(
            [
                "not json{",
                "{",
                "{'k': }",
                "",
                "[unclosed",
                "}{",
            ]
        )
    )
    return label, raw, None


def _sentinel_state():
    """Return a fresh MonitorState with a non-trivial sentinel buffer.

    The buffer holds a few candles and ``is_oversold`` is True so the test can
    detect any mutation by ``on_message`` itself. ``logger`` is left ``None``;
    ``on_message`` resolves the module Logger via ``logging.getLogger`` rather
    than ``state.logger``, so no logger wiring is required here.
    """
    buffer = [
        Candle(open_time_ms=1_000, open=1.0, high=2.0, low=0.5, close=1.5, volume=10.0),
        Candle(open_time_ms=2_000, open=1.5, high=2.5, low=1.0, close=2.0, volume=11.0),
        Candle(open_time_ms=3_000, open=2.0, high=3.0, low=1.5, close=2.5, volume=12.0),
    ]
    return MonitorState(buffer=buffer, is_oversold=True, logger=None)


# ---------------------------------------------------------------------------
# Property 6: WebSocket forwards iff candle is closed and preserves fields
# Validates: Requirements 3.3, 3.4, 3.5, 3.6, 3.7
# ---------------------------------------------------------------------------
@given(payload=ws_payloads())
@settings(max_examples=100, deadline=None)
def test_property6_forwards_iff_closed_and_preserves_fields(payload):
    """**Property 6: WebSocket forwards iff candle is closed and preserves fields**

    **Validates: Requirements 3.3, 3.4, 3.5, 3.6, 3.7**

    For any WebSocket payload, ``on_message`` invokes ``process_data`` iff the
    payload is valid JSON containing a ``"k"`` object with ``"x" == True``. When
    invoked, the forwarded ``Candle``'s six fields equal ``int(t)`` /
    ``float(o, h, l, c, v)`` of the source kline. For every other payload
    category ``process_data`` is NOT invoked. In all cases ``on_message`` itself
    leaves ``state.buffer`` and ``state.is_oversold`` unchanged (the real buffer
    mutation lives in the spied-out ``process_data``).
    """
    label, raw, fields = payload

    state = _sentinel_state()
    # Snapshot the pre-call state to assert on_message mutates nothing itself.
    buffer_before = list(state.buffer)
    oversold_before = state.is_oversold

    # Spy on the module-level process_data. The spy records its call args and
    # does nothing else, so any buffer change could only come from on_message.
    spy = mock.Mock(name="process_data_spy")
    with mock.patch.object(bsm, "process_data", spy):
        # ws is ignored by on_message; pass a dummy object.
        bsm.on_message(state, object(), raw)

    should_forward = label == "closed_valid"

    # --- Biconditional: forwarded iff valid JSON with "k.x" == True -----------
    assert spy.called is should_forward, (
        f"category {label!r}: expected process_data.called={should_forward}, "
        f"got {spy.called}"
    )

    if should_forward:
        # Exactly one forward per closed-valid payload (REQ-3.5).
        assert spy.call_count == 1, (
            f"expected exactly one process_data call, got {spy.call_count}"
        )
        # on_message calls process_data(state, candle): same state, a Candle.
        call_args, call_kwargs = spy.call_args
        assert call_kwargs == {}, f"unexpected kwargs forwarded: {call_kwargs}"
        forwarded_state, candle = call_args
        assert forwarded_state is state, "process_data must receive the same state"
        assert isinstance(candle, Candle)

        # --- Field preservation (REQ-3.3): six fields == source kline fields --
        t, o, h, low, c, v = fields
        assert candle.open_time_ms == int(t)
        assert candle.open == float(o)
        assert candle.high == float(h)
        assert candle.low == float(low)
        assert candle.close == float(c)
        assert candle.volume == float(v)

    # --- on_message itself mutates neither buffer nor is_oversold (all cases) -
    assert state.buffer == buffer_before, (
        f"category {label!r}: on_message mutated state.buffer"
    )
    assert state.is_oversold == oversold_before, (
        f"category {label!r}: on_message mutated state.is_oversold"
    )
