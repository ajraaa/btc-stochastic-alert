"""Dependency-free reproduction of Property 1 (buffer invariants).

Hypothesis/pytest cannot be installed in this environment (no pip), so this
script reimplements the property test's core logic with the stdlib ``random``
module to gain confidence the property holds and the oracle agrees with
``update_buffer``. It mirrors tests/test_buffer_properties.py:
  - draws candles with open_time_ms from a small domain (collisions + older
    arrivals + newer arrivals),
  - classifies each via the independent oracle,
  - applies update_buffer, asserts equality + structural invariants after each.
"""
import random
import sys
import types

# pandas is a runtime dependency imported at module load, but it is not
# installable in this environment (no pip). Inject a stub so the *real*
# btc_stochastic_monitor.update_buffer / Candle (which do not use pandas) can be
# imported and exercised. This stubbing is only for the dependency-free repro;
# the actual property test (tests/test_buffer_properties.py) runs against the
# genuine module with pandas installed via requirements-dev.txt.
sys.modules.setdefault("pandas", types.ModuleType("pandas"))

import btc_stochastic_monitor as bsm  # noqa: E402
from btc_stochastic_monitor import Candle, HISTORY_LIMIT, update_buffer  # noqa: E402


# Inlined copies of the oracle / invariant helpers from
# tests/test_buffer_properties.py (the test module itself imports hypothesis,
# which is also unavailable here). These MUST stay in sync with the test file.
def classify_and_expect(buffer, candle):
    open_times = [c.open_time_ms for c in buffer]
    if candle.open_time_ms in open_times:
        idx = open_times.index(candle.open_time_ms)
        expected = list(buffer)
        expected[idx] = candle
        return "replace", expected
    if not open_times or candle.open_time_ms > max(open_times):
        expected = list(buffer) + [candle]
        while len(expected) > HISTORY_LIMIT:
            expected = expected[1:]
        return "append", expected
    return "discard", list(buffer)


def _assert_buffer_invariants(buffer):
    times = [c.open_time_ms for c in buffer]
    assert len(buffer) <= HISTORY_LIMIT, f"buffer exceeded HISTORY_LIMIT: {len(buffer)}"
    assert all(
        times[i] < times[i + 1] for i in range(len(times) - 1)
    ), f"open times not strictly ascending: {times}"
    assert len(set(times)) == len(times), f"duplicate open times present: {times}"


def random_candle(rng: random.Random, max_time: int) -> Candle:
    return Candle(
        open_time_ms=rng.randint(0, max_time),
        open=rng.uniform(-1e9, 1e9),
        high=rng.uniform(-1e9, 1e9),
        low=rng.uniform(-1e9, 1e9),
        close=rng.uniform(-1e9, 1e9),
        volume=rng.uniform(-1e9, 1e9),
    )


def run() -> None:
    counts = {"append": 0, "replace": 0, "discard": 0}
    trims = 0
    # Many independent "state machine" runs, varied open-time domains so we hit
    # the HISTORY_LIMIT trim branch as well (max_time > 50).
    for seed in range(400):
        rng = random.Random(seed)
        max_time = rng.choice([5, 10, 30, 60, 80])
        buffer: list[Candle] = []
        steps = rng.randint(0, 120)
        for _ in range(steps):
            candle = random_candle(rng, max_time)
            before = list(buffer)
            classification, expected = classify_and_expect(before, candle)
            result = update_buffer(buffer, candle)

            assert result is buffer, "update_buffer must return the mutated buffer"
            assert result == expected, (
                f"{classification} mismatch\n before={[c.open_time_ms for c in before]}"
                f"\n candle={candle.open_time_ms}"
                f"\n expected={[c.open_time_ms for c in expected]}"
                f"\n actual={[c.open_time_ms for c in result]}"
            )

            if classification == "replace":
                assert len(result) == len(before)
                assert {c.open_time_ms for c in result} == {c.open_time_ms for c in before}
                assert candle in result
            elif classification == "append":
                assert result[-1] == candle
                assert len(result) == min(len(before) + 1, HISTORY_LIMIT)
                if len(before) + 1 > HISTORY_LIMIT:
                    trims += 1
            else:
                assert result == before

            _assert_buffer_invariants(buffer)
            counts[classification] += 1

    print("Property 1 reproduction PASSED")
    print(f"  operations exercised: {counts} (trim-branch hits: {trims})")
    print(f"  HISTORY_LIMIT = {HISTORY_LIMIT}")

    # Deterministic monotonic-append run to explicitly exercise the trim branch
    # (REQ-4.2): feed 200 strictly-increasing candles and confirm the buffer
    # stays pinned at HISTORY_LIMIT holding the most-recent window.
    buffer: list[Candle] = []
    rng = random.Random(12345)
    for t in range(200):
        candle = Candle(t, rng.random(), rng.random(), rng.random(), rng.random(), rng.random())
        classification, expected = classify_and_expect(buffer, candle)
        assert classification == "append"
        result = update_buffer(buffer, candle)
        assert result == expected
        _assert_buffer_invariants(buffer)
    assert len(buffer) == HISTORY_LIMIT
    assert [c.open_time_ms for c in buffer] == list(range(150, 200))
    print("  trim-branch monotonic run PASSED (buffer pinned at "
          f"{len(buffer)} holding open_times {buffer[0].open_time_ms}..{buffer[-1].open_time_ms})")


if __name__ == "__main__":
    run()
    sys.exit(0)
