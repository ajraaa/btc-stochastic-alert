"""BTC Stochastic DCA Monitor.

A single-file, production-ready Python 3.10+ script that supports a manual
Dollar-Cost Averaging (DCA) strategy on Bitcoin. It streams live 1-day
BTC/USDT candles from the Binance public WebSocket, computes the Stochastic
Oscillator (%K, %D) with pandas / pandas-ta using parameters (5, 3, 3), and
sends a single Telegram notification at the moment %K crosses below the
oversold threshold (20). The Monitor runs unattended with historical
bootstrapping via Binance REST, auto-reconnect logic, and a boolean cool-down
lock that prevents notification spam.

This module is intended to be edited in one place by an operator: all runtime
configuration is declared as module-level constants near the top of the file
(see the "Module-Level Constants" section below).
"""

from __future__ import annotations  # Enable PEP 604 type hints under Python 3.10

# --- Standard library imports ------------------------------------------------
import dataclasses
import enum
import json
import logging
import sys
import time
import typing

# =============================================================================
# Module-Level Constants
# -----------------------------------------------------------------------------
# All runtime configuration is declared here, before any function or class
# definition, so an operator can edit one place without searching through logic
# (REQ-1.1). Each constant traces back to a specific acceptance criterion.
# =============================================================================

# Telegram credentials (operator-supplied; empty by default, REQ-1.1, REQ-1.9).
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# Trading pair and timeframe (REQ-1.2).
SYMBOL = "btcusdt"
INTERVAL = "1d"

# Stochastic Oscillator parameters: %K Length, %K Smoothing, %D Smoothing
# (REQ-1.3).
STOCH_K_LENGTH = 5
STOCH_K_SMOOTH = 3
STOCH_D_SMOOTH = 3

# pandas-ta output column names for the configured parameters (REQ-1.5).
STOCH_K_COL = "STOCHk_5_3_3"
STOCH_D_COL = "STOCHd_5_3_3"

# Oversold threshold below which %K indicates an oversold market (REQ-1.4).
OVERSOLD_THRESHOLD = 20

# Historical kline limit / bounded candle-buffer size (REQ-1.6).
HISTORY_LIMIT = 50

# Reconnect delay applied before each WebSocket reconnection attempt (REQ-1.7).
RECONNECT_DELAY_S = 5

# Backup log file path used by the Logger's FileHandler (REQ-1.8).
BACKUP_LOG_FILE = "btc_stochastic_monitor.log"

# Stale-buffer threshold (1 day, in ms): a larger gap triggers re-bootstrap
# (REQ-8.5).
STALE_BUFFER_THRESHOLD_MS = 86_400_000

# Request timeouts (seconds).
REST_TIMEOUT_S = 10  # Binance REST bootstrap (REQ-2.1).
TELEGRAM_TIMEOUT_S = 10  # Telegram sendMessage (REQ-7.1).

# Binance endpoints.
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"  # REST klines (REQ-2.1).
BINANCE_WS_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL}@kline_{INTERVAL}"  # WS stream (REQ-3.1).

# Default event category applied to any log record whose caller did not supply
# an explicit ``extra={"category": ...}`` value. Keeping a fallback here means a
# plain ``logger.info("msg")`` call still formats cleanly instead of raising a
# ``KeyError`` on the ``%(category)s`` placeholder (REQ-11.5).
DEFAULT_LOG_CATEGORY = "MONITOR"

# Logger name shared by every component so they all resolve the same configured
# Logger via ``logging.getLogger(LOGGER_NAME)`` (REQ-11.6).
LOGGER_NAME = "btc_stochastic_monitor"

# Formatter pattern and ISO-8601 date format. ``asctime`` is rendered with the
# ``datefmt`` below (UTC via ``time.gmtime``) and the ``.%(msecs)03dZ`` suffix
# appends millisecond precision plus the ``Z`` UTC designator, yielding e.g.
# ``2024-01-01T00:00:00.123Z`` (REQ-11.5).
_LOG_FORMAT = "%(asctime)s.%(msecs)03dZ [%(category)s] %(levelname)s %(message)s"
_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"


# =============================================================================
# Logger
# =============================================================================


class _UtcCategoryFormatter(logging.Formatter):
    """Formatter that renders UTC timestamps and tolerates a missing category.

    Two behaviours layer on top of the stdlib :class:`logging.Formatter`:

    * ``converter`` is set to :func:`time.gmtime` so ``asctime`` is always UTC,
      independent of the host's local timezone (REQ-11.5).
    * Any record lacking a ``category`` attribute (a caller that omitted
      ``extra={"category": ...}``) is given :data:`DEFAULT_LOG_CATEGORY` before
      formatting, so the ``%(category)s`` placeholder never raises. Callers that
      *do* pass ``extra={"category": "WS"}`` keep their value untouched, so both
      ``logger.info("msg")`` and ``logger.info("msg", extra={"category": "WS"})``
      format without error.
    """

    converter = time.gmtime  # UTC timestamps (REQ-11.5).

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "category"):
            # Inject the fallback category for un-categorized records so the
            # ``%(category)s`` field always resolves (REQ-11.5).
            record.category = DEFAULT_LOG_CATEGORY
        return super().format(record)


def init_logger(log_path: str) -> logging.Logger:
    """Initialize and return the Monitor's centralized Logger.

    Purpose:
        Configure a single ``logging.Logger`` (named :data:`LOGGER_NAME`) that
        fans every record out to standard output and to a backup log file, with
        a UTC-timestamped, category-tagged formatter (REQ-11.1..REQ-11.5).

    Inputs:
        log_path: Filesystem path for the backup log file's ``FileHandler``
            (the Backup_Log_File, REQ-1.8, REQ-11.3).

    Returns / side effects:
        Returns the configured ``logging.Logger``. Attaches a
        ``StreamHandler(sys.stdout)`` and, when it can be opened, a
        ``FileHandler(log_path)``; both share one :class:`_UtcCategoryFormatter`.
        If the ``FileHandler`` cannot be opened (e.g. permission denied or a
        missing directory), the Logger logs a one-time WARNING through the
        StreamHandler and degrades to StreamHandler-only rather than crashing
        (REQ-11.7). Existing handlers are cleared first so repeated calls do not
        accumulate duplicate handlers.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    # Clear any handlers from a previous init_logger call so repeated
    # invocations don't fan a single record out through duplicate handlers.
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        try:
            existing.close()
        except Exception:  # pragma: no cover - defensive cleanup only
            pass

    formatter = _UtcCategoryFormatter(fmt=_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    # Belt-and-suspenders: the class attribute already sets UTC conversion, but
    # set it on the instance too so the contract is explicit (REQ-11.5).
    formatter.converter = time.gmtime

    # StreamHandler (stdout) is attached first so it is available to report a
    # FileHandler degradation through the Logger itself (REQ-11.2, REQ-11.7).
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler (Backup_Log_File). A permission error or missing directory
    # raises OSError; degrade to StreamHandler-only instead of crashing
    # (REQ-11.3, REQ-11.7).
    try:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        # One-time WARNING emitted via the already-attached StreamHandler so the
        # operator learns persistence is disabled, without terminating the
        # Monitor (REQ-11.7).
        logger.warning(
            "FileHandler init failed for %r (%s); continuing with stdout only",
            log_path,
            exc,
            extra={"category": "STARTUP"},
        )

    return logger


# =============================================================================
# Configuration Validation & Startup Guard
# =============================================================================


def validate_config() -> None:
    """Validate the runtime environment and required credentials before startup.

    Purpose:
        Enforce the only two fail-fast termination conditions in the Monitor
        (see design "Termination Conditions"): an unsupported Python version
        and missing/empty Telegram credentials. Both checks run *before any
        network connection is opened* (REQ-1.9, REQ-10.2). This function is
        invoked from ``main`` (wired in task 12.2) ahead of ``bootstrap_history``.

    Inputs:
        None. Reads the module-level interpreter version (``sys.version_info``)
        and the :data:`TELEGRAM_BOT_TOKEN` / :data:`TELEGRAM_CHAT_ID` constants.

    Returns / side effects:
        Returns ``None`` when the environment is valid. Otherwise emits a
        ``CONFIG``-category ERROR log record through the shared Logger and
        terminates the process via ``sys.exit(1)``. The Logger may not yet have
        handlers attached when this runs (``main`` calls ``validate_config``
        before ``init_logger``); in that case Python's logging "last resort"
        handler still surfaces the ERROR to stderr, so the operator always sees
        the reason for termination without resorting to ``print`` (REQ-11.6).
    """
    # All components resolve the same named Logger; obtain it here even though
    # handlers may not be configured yet (REQ-11.6).
    logger = logging.getLogger(LOGGER_NAME)

    # --- Python version guard (REQ-10.1, REQ-10.2) ---------------------------
    # Check the interpreter version first so an unsupported runtime is reported
    # before we even look at credentials, and always before any network call.
    if sys.version_info < (3, 10):
        detected = ".".join(str(part) for part in sys.version_info[:3])
        logger.error(
            "Unsupported Python version %s detected; the Monitor requires "
            "Python 3.10 or later. Terminating before opening any network "
            "connection.",
            detected,
            extra={"category": "CONFIG"},
        )
        sys.exit(1)

    # --- Telegram credential guard (REQ-1.9) ---------------------------------
    # A missing or empty (after stripping whitespace) token or chat ID makes
    # notification delivery impossible, so terminate before any network call and
    # name the offending constant so the operator knows exactly what to fix.
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_BOT_TOKEN.strip():
        logger.error(
            "Required configuration constant TELEGRAM_BOT_TOKEN is missing or "
            "empty. Terminating before opening any network connection.",
            extra={"category": "CONFIG"},
        )
        sys.exit(1)

    if not TELEGRAM_CHAT_ID or not TELEGRAM_CHAT_ID.strip():
        logger.error(
            "Required configuration constant TELEGRAM_CHAT_ID is missing or "
            "empty. Terminating before opening any network connection.",
            extra={"category": "CONFIG"},
        )
        sys.exit(1)
