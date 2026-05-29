"""Property-based tests for the centralized Logger (``init_logger``).

This module hosts the logger fan-out / formatting property test for the BTC
Stochastic DCA Monitor.

* Property 9 (this file): every record emitted at INFO / WARNING / ERROR is
  delivered to BOTH the ``StreamHandler`` (stdout) and the ``FileHandler``
  (Backup_Log_File), and each formatted line begins with an ISO-8601 UTC
  timestamp (with millisecond precision and a trailing ``Z``) followed by the
  bracketed event-category identifier supplied by the emitting component.

The module is intentionally structured as standalone, top-level test functions
with shared imports/helpers at module scope so additional logger tests (e.g. the
``FileHandler`` degradation unit test) can be appended without conflict.
"""

from __future__ import annotations

import io
import logging
import re
import uuid

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import btc_stochastic_monitor as bsm

# ---------------------------------------------------------------------------
# Shared strategies / regexes / helpers
# ---------------------------------------------------------------------------

# Property 9 quantifies over records emitted at INFO, WARNING, or ERROR.
LOG_LEVELS = (logging.INFO, logging.WARNING, logging.ERROR)

# Event-category identifiers as supplied via ``extra={"category": ...}``.
# Constrained to realistic tokens (upper-case letters, digits, underscore) so
# they never contain a closing bracket or whitespace that would confuse the
# "[CATEGORY]" portion of a formatted line.
category_strategy = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",
    min_size=1,
    max_size=12,
)

# Log messages: single-line printable ASCII (space..tilde). Excluding control
# characters keeps each emitted record on exactly one formatted line while still
# exercising tricky content such as '%', '[' and ']' inside the message body.
message_strategy = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    min_size=0,
    max_size=120,
)

# ISO-8601 date+time with millisecond precision and a trailing UTC 'Z', matching
# the formatter pattern "%(asctime)s.%(msecs)03dZ" with datefmt
# "%Y-%m-%dT%H:%M:%S" (e.g. 2024-01-01T00:00:00.123Z).
_ISO_TS = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"

# A formatted line must begin with the ISO timestamp, a space, the bracketed
# category, a space, then the level name.
LINE_PREFIX_RE = re.compile(
    rf"^{_ISO_TS} \[(?P<category>[^\]]*)\] (?P<level>INFO|WARNING|ERROR) "
)


def _find_stream_handler(logger: logging.Logger) -> logging.StreamHandler:
    """Return the lone non-file ``StreamHandler`` attached to ``logger``.

    ``FileHandler`` subclasses ``StreamHandler``, so it is explicitly excluded
    to isolate the stdout branch (REQ-11.2).
    """
    candidates = [
        h
        for h in logger.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    assert len(candidates) == 1, (
        f"expected exactly one StreamHandler, found {len(candidates)}"
    )
    return candidates[0]


def _find_file_handler(logger: logging.Logger) -> logging.FileHandler:
    """Return the lone ``FileHandler`` attached to ``logger`` (REQ-11.3)."""
    candidates = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(candidates) == 1, (
        f"expected exactly one FileHandler, found {len(candidates)}"
    )
    return candidates[0]


def _capture_stream(handler: logging.StreamHandler) -> io.StringIO:
    """Redirect ``handler``'s output to an in-memory buffer and return it."""
    buffer = io.StringIO()
    if hasattr(handler, "setStream"):
        handler.setStream(buffer)
    else:  # pragma: no cover - setStream exists on all supported (3.10+) runtimes
        handler.stream = buffer
    return buffer


# ---------------------------------------------------------------------------
# Property 9: Logger fan-out and formatting
# Validates: Requirements 11.4, 11.5
# ---------------------------------------------------------------------------
@settings(
    max_examples=100,
    deadline=None,
    # tmp_path and caplog are function-scoped; each example builds its own
    # isolated log file, so reusing the fixture handles across examples is safe.
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    level=st.sampled_from(LOG_LEVELS),
    category=category_strategy,
    message=message_strategy,
)
# Representative concrete examples for each level / category combination.
@example(level=logging.INFO, category="WS", message="websocket opened")
@example(level=logging.WARNING, category="INDICATOR", message="NaN in last row")
@example(level=logging.ERROR, category="NOTIFY", message="dispatch failed: 500")
def test_property9_logger_fanout_and_formatting(tmp_path, caplog, level, category, message):
    """**Property 9: Logger fan-out and formatting**

    **Validates: Requirements 11.4, 11.5**

    For any record emitted at INFO / WARNING / ERROR through the configured
    Logger, BOTH the ``StreamHandler`` and the ``FileHandler`` receive the
    record, and each formatted line begins with an ISO-8601-with-milliseconds
    UTC timestamp followed by the bracketed event-category identifier.
    """
    caplog.clear()
    # Unique backup-log path per example for full isolation between runs.
    log_path = tmp_path / f"fanout_{uuid.uuid4().hex}.log"

    with caplog.at_level(logging.INFO, logger=bsm.LOGGER_NAME):
        logger = bsm.init_logger(str(log_path))

        # Capture the stdout branch by swapping the StreamHandler's stream for
        # an in-memory buffer; the FileHandler branch is read back from disk.
        stream_handler = _find_stream_handler(logger)
        file_handler = _find_file_handler(logger)
        stream_buffer = _capture_stream(stream_handler)

        # Emit one record carrying the generated event category (REQ-11.4).
        logger.log(level, message, extra={"category": category})

        # Flush both branches, capture the stream output, then close the file
        # handler so its buffer is fully flushed and the OS handle released
        # before we read the backup file back.
        for handler in logger.handlers:
            handler.flush()
        stream_output = stream_buffer.getvalue()
        file_handler.close()
        logger.removeHandler(file_handler)
        file_content = log_path.read_text(encoding="utf-8")

    # --- Fan-out: both handlers received the record (REQ-11.4) -------------
    assert message in stream_output, "StreamHandler did not receive the record"
    assert message in file_content, "FileHandler did not receive the record"

    # Each emitted record yields exactly one formatted line on each branch.
    stream_lines = stream_output.splitlines()
    file_lines = file_content.splitlines()
    assert len(stream_lines) == 1
    assert len(file_lines) == 1

    expected_level = logging.getLevelName(level)

    # --- Formatting: ISO-8601(ms, UTC) timestamp + [CATEGORY] prefix (REQ-11.5)
    for line in (stream_lines[0], file_lines[0]):
        match = LINE_PREFIX_RE.match(line)
        assert match is not None, (
            f"line did not start with ISO-8601 ms UTC timestamp + [category]: {line!r}"
        )
        assert match.group("category") == category, (
            f"expected category {category!r} in {line!r}"
        )
        assert match.group("level") == expected_level

    # Both handlers share a single formatter, so the formatted lines match.
    assert stream_lines[0] == file_lines[0]

    # --- caplog confirms the record propagated with the expected attributes -
    captured = [r for r in caplog.records if getattr(r, "category", None) == category]
    assert len(captured) == 1, "expected exactly one captured record for the category"
    assert captured[0].levelno == level
    assert captured[0].getMessage() == message


# ---------------------------------------------------------------------------
# Unit test: FileHandler degradation (StreamHandler-only fallback)
# Validates: Requirements 11.7
# ---------------------------------------------------------------------------
def test_filehandler_permission_error_degrades_to_stdout_only(tmp_path, caplog, monkeypatch):
    """``init_logger`` degrades to StreamHandler-only when the FileHandler fails.

    **Validates: Requirements 11.7**

    When the backup log file cannot be opened (here simulated by forcing
    ``logging.FileHandler.__init__`` to raise ``PermissionError`` -- a subclass
    of the ``OSError`` that ``init_logger`` catches), the returned Logger must:

    * not propagate the exception (``init_logger`` does not raise),
    * carry exactly one handler -- the stdout ``StreamHandler`` that is NOT a
      ``FileHandler``, and
    * have emitted a single WARNING explaining the degradation (logged under the
      ``STARTUP`` event category via the already-attached StreamHandler).
    """
    # Force FileHandler construction to fail exactly as a non-writable backup
    # path would. PermissionError <: OSError, so init_logger's ``except OSError``
    # branch (the degradation path) is exercised.
    def _raise_permission_error(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(logging.FileHandler, "__init__", _raise_permission_error)

    log_path = tmp_path / "unwritable" / "monitor.log"

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=bsm.LOGGER_NAME):
        # Must NOT raise even though the FileHandler cannot be constructed.
        logger = bsm.init_logger(str(log_path))

    # --- A usable Logger is returned ---------------------------------------
    assert isinstance(logger, logging.Logger)

    # --- Exactly one handler, the StreamHandler (never a FileHandler) -------
    assert len(logger.handlers) == 1, (
        f"expected the degraded Logger to keep a single handler, "
        f"found {len(logger.handlers)}"
    )
    stream_handler = _find_stream_handler(logger)  # asserts a lone non-file StreamHandler
    assert not isinstance(stream_handler, logging.FileHandler)
    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers), (
        "degraded Logger must not retain any FileHandler"
    )

    # --- A single WARNING explaining the degradation was emitted (STARTUP) --
    degradation_warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and getattr(record, "category", None) == "STARTUP"
    ]
    assert len(degradation_warnings) == 1, (
        "expected exactly one STARTUP-category WARNING describing the "
        f"FileHandler degradation, found {len(degradation_warnings)}"
    )

    # The message should explain what degraded (the FileHandler) and that the
    # Monitor continues via stdout -- robust to minor wording changes by
    # matching the salient, lower-cased tokens rather than the exact string.
    warning_message = degradation_warnings[0].getMessage().lower()
    assert "filehandler" in warning_message, (
        f"degradation WARNING should mention the FileHandler: {warning_message!r}"
    )
    assert "stdout" in warning_message, (
        f"degradation WARNING should mention continuing on stdout: {warning_message!r}"
    )
