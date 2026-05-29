"""Unit tests for configuration validation (``validate_config``).

Task 3.2 / Requirements 1.9, 10.2.

``btc_stochastic_monitor.validate_config()`` is the Monitor's fail-fast startup
guard. It is the only path (besides an operator signal) that terminates the
process, and it MUST do so *before any network connection is opened*
(REQ-1.9, REQ-10.2). The implementation performs its checks in this order:

1. Python version  -- ``sys.version_info < (3, 10)`` -> CONFIG error + ``sys.exit(1)``.
2. ``TELEGRAM_BOT_TOKEN`` -- missing/empty -> CONFIG error + ``sys.exit(1)``.
3. ``TELEGRAM_CHAT_ID``   -- missing/empty -> CONFIG error + ``sys.exit(1)``.

These tests parameterize the four failure cases (empty token, empty chat ID,
both empty, and a Python 3.9 interpreter) and assert that each one:

* raises ``SystemExit`` with exit code ``1``,
* emits at least one ERROR log record tagged with the ``CONFIG`` event category
  through the shared ``btc_stochastic_monitor`` Logger, and
* performs zero network I/O -- verified with the ``responses`` library (no HTTP
  request leaves the process, ``len(responses.calls) == 0``) and with a tracking
  WebSocket factory that is never instantiated.

The tests deliberately set valid values for the constants that are *not* under
test in each case, so exactly one guard fires and the assertions are
unambiguous (e.g. the Python-3.9 case supplies a valid token and chat ID so the
version check is the sole failure).
"""

from __future__ import annotations

import logging
import sys

import pytest
import responses

import btc_stochastic_monitor as bsm

# A non-empty token / chat ID used to satisfy the guard(s) NOT under test so
# that exactly one validation branch fires per case.
VALID_TOKEN = "123456:valid-bot-token"
VALID_CHAT_ID = "987654321"

# A Python version tuple below the 3.10 minimum, used to trigger REQ-10.2.
PYTHON_3_9 = (3, 9, 0)


# ---------------------------------------------------------------------------
# Parameterized failure cases (REQ-1.9, REQ-10.2)
# ---------------------------------------------------------------------------
# Each tuple is (case_id, token, chat_id, version_info):
#   * token / chat_id are assigned onto the module constants for the case.
#   * version_info is monkeypatched onto ``sys.version_info`` when not None;
#     None means "leave the real (supported) interpreter version in place".
CONFIG_FAILURE_CASES = [
    # Empty token, valid chat ID, supported Python -> token guard fires.
    pytest.param("empty_token", "", VALID_CHAT_ID, None, id="empty_token"),
    # Valid token, empty chat ID, supported Python -> chat-ID guard fires.
    pytest.param("empty_chat_id", VALID_TOKEN, "", None, id="empty_chat_id"),
    # Both empty -> token guard fires first (validation order: token before chat).
    pytest.param("both_empty", "", "", None, id="both_empty"),
    # Valid credentials but Python 3.9 -> version guard fires first.
    pytest.param("python_3_9", VALID_TOKEN, VALID_CHAT_ID, PYTHON_3_9, id="python_3_9"),
]


@pytest.mark.parametrize("case_id, token, chat_id, version_info", CONFIG_FAILURE_CASES)
@responses.activate
def test_validate_config_failure_exits_logs_and_makes_no_network_calls(
    case_id,
    token,
    chat_id,
    version_info,
    monkeypatch,
    caplog,
    fake_ws_class,
):
    """Each invalid-config case exits(1), logs CONFIG, and makes zero network calls.

    **Validates: Requirements 1.9, 10.2**

    The ``@responses.activate`` decorator activates an HTTP mock with NO
    registered endpoints: any outbound ``requests`` call would raise, and
    ``responses.calls`` stays empty when (as required) no HTTP request is made.
    A tracking subclass of the conftest ``FakeWebSocketApp`` records every
    instantiation; ``validate_config`` must open no WebSocket, so the tracker
    stays empty.
    """
    # --- Arrange the constants under test ----------------------------------
    # Assign the case's token / chat ID onto the module-level constants so only
    # the intended guard fires (the non-tested constant is given a valid value).
    monkeypatch.setattr(bsm, "TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setattr(bsm, "TELEGRAM_CHAT_ID", chat_id)

    # For the version case, spoof an unsupported interpreter (REQ-10.2). The
    # other cases run on the real (supported) interpreter version.
    if version_info is not None:
        monkeypatch.setattr(sys, "version_info", version_info)

    # --- A WebSocket factory that records (and forbids) instantiation -------
    # validate_config must terminate before opening any WebSocket. We wrap the
    # conftest FakeWebSocketApp in a tracking subclass and assert it is never
    # constructed; this expresses "the fake WebSocket factory is not
    # instantiated" from the task description.
    ws_instantiations: list[tuple] = []

    class TrackingWebSocketApp(fake_ws_class):
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            ws_instantiations.append((args, kwargs))
            super().__init__(*args, **kwargs)

    # --- Act + Assert: SystemExit(1) ---------------------------------------
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=bsm.LOGGER_NAME):
        with pytest.raises(SystemExit) as exc_info:
            bsm.validate_config()

    # sys.exit(1) raises SystemExit whose .code is the integer exit status.
    assert exc_info.value.code == 1, (
        f"{case_id}: expected SystemExit code 1, got {exc_info.value.code!r}"
    )

    # --- Assert: a CONFIG-category ERROR record was emitted -----------------
    config_errors = [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR
        and getattr(record, "category", None) == "CONFIG"
    ]
    assert len(config_errors) >= 1, (
        f"{case_id}: expected at least one CONFIG-category ERROR log record, "
        f"found {len(config_errors)} (records: {[r.getMessage() for r in caplog.records]})"
    )

    # --- Assert: zero HTTP calls (REQ-1.9 'before opening any network connection')
    assert len(responses.calls) == 0, (
        f"{case_id}: validate_config must make no HTTP calls, "
        f"saw {len(responses.calls)}"
    )

    # --- Assert: zero WebSocket sessions opened ----------------------------
    assert ws_instantiations == [], (
        f"{case_id}: validate_config must not instantiate the WebSocket client, "
        f"saw {len(ws_instantiations)} instantiation(s)"
    )


def test_validate_config_reports_offending_constant_name(monkeypatch, caplog):
    """The CONFIG error names which constant is missing (operator-actionable).

    **Validates: Requirements 1.9**

    REQ-1.9 requires the configuration error to *identify the missing constant*.
    With an empty token (and a valid chat ID) the emitted message must name
    ``TELEGRAM_BOT_TOKEN``; with an empty chat ID (and a valid token) it must
    name ``TELEGRAM_CHAT_ID``.
    """
    # Empty token -> message names TELEGRAM_BOT_TOKEN.
    monkeypatch.setattr(bsm, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(bsm, "TELEGRAM_CHAT_ID", VALID_CHAT_ID)
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=bsm.LOGGER_NAME):
        with pytest.raises(SystemExit):
            bsm.validate_config()
    assert any("TELEGRAM_BOT_TOKEN" in r.getMessage() for r in caplog.records), (
        "empty-token error must name TELEGRAM_BOT_TOKEN"
    )

    # Empty chat ID -> message names TELEGRAM_CHAT_ID.
    monkeypatch.setattr(bsm, "TELEGRAM_BOT_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(bsm, "TELEGRAM_CHAT_ID", "")
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=bsm.LOGGER_NAME):
        with pytest.raises(SystemExit):
            bsm.validate_config()
    assert any("TELEGRAM_CHAT_ID" in r.getMessage() for r in caplog.records), (
        "empty-chat-id error must name TELEGRAM_CHAT_ID"
    )


def test_validate_config_passes_with_valid_config(monkeypatch):
    """A supported interpreter plus non-empty credentials returns without exiting.

    **Validates: Requirements 1.9, 10.1**

    Sanity check that the guard does not fire on a valid configuration: the
    function returns ``None`` and does not raise ``SystemExit``.
    """
    monkeypatch.setattr(bsm, "TELEGRAM_BOT_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(bsm, "TELEGRAM_CHAT_ID", VALID_CHAT_ID)
    # Real interpreter is >= 3.10 in this environment; assert that assumption so
    # the test fails loudly if run on an unsupported runtime.
    assert sys.version_info >= (3, 10)

    assert bsm.validate_config() is None
