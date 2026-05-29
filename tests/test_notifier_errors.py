"""Failure-mode unit tests for the Telegram Notifier (``send_telegram_notification``).

Task 8.3 / Requirements 7.3, 7.5, 7.6.

``btc_stochastic_monitor.send_telegram_notification(message: str) -> bool`` is the
Monitor's outbound notification boundary. It follows the project's
"log and continue" philosophy: it MUST never raise, it MUST return ``False`` on
any failure, and it MUST emit exactly one diagnostic log record describing the
failure so an operator can act on it. The behaviour under test (verified against
the implementation) is:

* Missing/empty ``TELEGRAM_BOT_TOKEN`` or ``TELEGRAM_CHAT_ID`` at send time ->
  a ``CONFIG``-category ERROR naming the missing constant, the HTTPS POST is
  skipped entirely, returns ``False`` (REQ-7.6).
* The Telegram API responds with a non-2xx status (e.g. 400/401/403/500) ->
  a ``NOTIFY``-category ERROR including the HTTP status code, returns ``False``
  (REQ-7.3).
* ``requests.post`` raises a transport exception (timeout / connection error) ->
  a ``NOTIFY``-category ERROR including the exception type name, returns
  ``False`` (REQ-7.5).

The credentials are module-level constants read at call time, so each test
monkeypatches ``bsm.TELEGRAM_BOT_TOKEN`` / ``bsm.TELEGRAM_CHAT_ID`` and builds
the mocked endpoint URL from the *same* token value so the ``responses`` mock
matches the request the function actually issues.
"""

from __future__ import annotations

import logging

import pytest
import requests
import responses

import btc_stochastic_monitor as bsm

# Valid credentials used for the cases where config is NOT the failure under
# test (HTTP-status and transport-exception cases). The endpoint URL is derived
# from this token exactly as the implementation builds it.
VALID_TOKEN = "123456:test-bot-token"
VALID_CHAT_ID = "987654321"
TELEGRAM_URL = f"https://api.telegram.org/bot{VALID_TOKEN}/sendMessage"

ALERT_MESSAGE = "BTC Stochastic DCA alert: btcusdt @ 1d is oversold."


def _notify_errors(caplog) -> list[logging.LogRecord]:
    """Return the captured ERROR records tagged with the NOTIFY category."""
    return [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR
        and getattr(record, "category", None) == "NOTIFY"
    ]


def _config_errors(caplog) -> list[logging.LogRecord]:
    """Return the captured ERROR records tagged with the CONFIG category."""
    return [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR
        and getattr(record, "category", None) == "CONFIG"
    ]


# ---------------------------------------------------------------------------
# Non-2xx HTTP responses -> NOTIFY ERROR with the status code, returns False
# Validates: Requirements 7.3
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status_code", [400, 401, 403, 500])
@responses.activate
def test_non_2xx_status_logs_notify_error_and_returns_false(
    status_code, monkeypatch, caplog
):
    """**Validates: Requirements 7.3**

    A non-success HTTP status causes exactly one NOTIFY-category ERROR that
    includes the status code, returns ``False``, and never raises.
    """
    # Valid credentials so the config guard does not fire; the URL is built from
    # the same token the implementation will use so the mock matches.
    monkeypatch.setattr(bsm, "TELEGRAM_BOT_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(bsm, "TELEGRAM_CHAT_ID", VALID_CHAT_ID)
    responses.add(responses.POST, TELEGRAM_URL, status=status_code)

    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=bsm.LOGGER_NAME):
        # No exception must propagate from a non-2xx response.
        result = bsm.send_telegram_notification(ALERT_MESSAGE)

    assert result is False

    # Exactly one POST was attempted (the request was made, then judged failed).
    assert len(responses.calls) == 1

    notify_errors = _notify_errors(caplog)
    assert len(notify_errors) == 1, (
        f"expected exactly one NOTIFY ERROR for HTTP {status_code}, "
        f"found {len(notify_errors)}"
    )
    assert str(status_code) in notify_errors[0].getMessage(), (
        f"NOTIFY error must include the HTTP status code {status_code}: "
        f"{notify_errors[0].getMessage()!r}"
    )


# ---------------------------------------------------------------------------
# Transport exceptions -> NOTIFY ERROR with the exception type, returns False
# Validates: Requirements 7.5
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "exc_factory, expected_name",
    [
        (requests.exceptions.Timeout, "Timeout"),
        (requests.exceptions.ConnectionError, "ConnectionError"),
    ],
    ids=["timeout", "connection_error"],
)
@responses.activate
def test_transport_exception_logs_notify_error_and_returns_false(
    exc_factory, expected_name, monkeypatch, caplog
):
    """**Validates: Requirements 7.5**

    A connection/timeout exception from ``requests.post`` is caught: exactly one
    NOTIFY-category ERROR naming the exception type is emitted, ``False`` is
    returned, and the exception does not propagate.
    """
    monkeypatch.setattr(bsm, "TELEGRAM_BOT_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(bsm, "TELEGRAM_CHAT_ID", VALID_CHAT_ID)
    # The responses library raises the exception instance supplied as ``body``
    # when the endpoint is hit, simulating a transport-level failure.
    responses.add(responses.POST, TELEGRAM_URL, body=exc_factory())

    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=bsm.LOGGER_NAME):
        # The transport exception must be swallowed, not re-raised.
        result = bsm.send_telegram_notification(ALERT_MESSAGE)

    assert result is False

    notify_errors = _notify_errors(caplog)
    assert len(notify_errors) == 1, (
        f"expected exactly one NOTIFY ERROR for {expected_name}, "
        f"found {len(notify_errors)}"
    )
    assert expected_name in notify_errors[0].getMessage(), (
        f"NOTIFY error must name the exception type {expected_name!r}: "
        f"{notify_errors[0].getMessage()!r}"
    )


# ---------------------------------------------------------------------------
# Missing/empty credentials -> CONFIG ERROR, POST skipped, returns False
# Validates: Requirements 7.6
# ---------------------------------------------------------------------------
# Each case sets exactly one credential to an empty value (the other stays
# valid) so a single CONFIG guard fires and the assertions are unambiguous. The
# whitespace case confirms the implementation treats blank strings as missing.
@pytest.mark.parametrize(
    "token, chat_id, missing_constant",
    [
        pytest.param("", VALID_CHAT_ID, "TELEGRAM_BOT_TOKEN", id="empty_token"),
        pytest.param("   ", VALID_CHAT_ID, "TELEGRAM_BOT_TOKEN", id="whitespace_token"),
        pytest.param(VALID_TOKEN, "", "TELEGRAM_CHAT_ID", id="empty_chat_id"),
        pytest.param(VALID_TOKEN, "   ", "TELEGRAM_CHAT_ID", id="whitespace_chat_id"),
    ],
)
@responses.activate
def test_missing_credentials_logs_config_error_and_skips_post(
    token, chat_id, missing_constant, monkeypatch, caplog
):
    """**Validates: Requirements 7.6**

    When a credential is missing/empty at send time, the Notifier logs a
    CONFIG-category ERROR naming the missing constant, skips the HTTPS POST
    entirely (zero HTTP calls), returns ``False``, and never raises.
    """
    monkeypatch.setattr(bsm, "TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setattr(bsm, "TELEGRAM_CHAT_ID", chat_id)
    # Intentionally register no endpoints: if the function wrongly attempted a
    # POST, ``responses`` would record a call (and raise), failing the test.

    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=bsm.LOGGER_NAME):
        result = bsm.send_telegram_notification(ALERT_MESSAGE)

    assert result is False

    # The POST must be skipped before any network I/O (REQ-7.6).
    assert len(responses.calls) == 0, (
        "send_telegram_notification must skip the POST when config is missing, "
        f"saw {len(responses.calls)} HTTP call(s)"
    )

    config_errors = _config_errors(caplog)
    assert len(config_errors) == 1, (
        f"expected exactly one CONFIG ERROR, found {len(config_errors)}"
    )
    assert missing_constant in config_errors[0].getMessage(), (
        f"CONFIG error must name the missing constant {missing_constant!r}: "
        f"{config_errors[0].getMessage()!r}"
    )


# ---------------------------------------------------------------------------
# Control: a 2xx response returns True (guards against trivially-passing
# failure assertions caused by URL mismatch).
# ---------------------------------------------------------------------------
@responses.activate
def test_success_response_returns_true(monkeypatch, caplog):
    """A 2xx Telegram response returns ``True`` and emits no NOTIFY/CONFIG ERROR.

    This control confirms the mocked endpoint URL matches the request the
    implementation issues, so the failure-mode assertions above exercise the
    intended code paths rather than an accidental URL mismatch.
    """
    monkeypatch.setattr(bsm, "TELEGRAM_BOT_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(bsm, "TELEGRAM_CHAT_ID", VALID_CHAT_ID)
    responses.add(responses.POST, TELEGRAM_URL, json={"ok": True}, status=200)

    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=bsm.LOGGER_NAME):
        result = bsm.send_telegram_notification(ALERT_MESSAGE)

    assert result is True
    assert len(responses.calls) == 1
    assert _notify_errors(caplog) == []
    assert _config_errors(caplog) == []
