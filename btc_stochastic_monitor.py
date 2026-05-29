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
import math
import sys
import time
import typing

# --- Third-party imports -----------------------------------------------------
import pandas as pd

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


# =============================================================================
# Data Models
# -----------------------------------------------------------------------------
# Small, explicit value/state containers shared across components. ``Candle``
# and ``StochasticReading`` are immutable (frozen) value objects; ``MonitorState``
# is the single mutable container that survives WebSocket session restarts
# (REQ-8.4). ``TriggerDecision`` enumerates the cool-down state-machine outcomes
# and doubles as the trigger-decision field in indicator-evaluation log records
# (REQ-10.5).
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Candle:
    """An immutable OHLCV record for one 1-day kline.

    Open time is stored as Unix epoch milliseconds (Binance's native unit) so
    candles compare and sort unambiguously and so buffer operations can key on
    ``open_time_ms`` for append/replace/discard decisions (REQ-2.3, REQ-4.1,
    REQ-4.3, REQ-4.4, REQ-4.5). Constructed from either a REST bootstrap row
    ``[openTime, open, high, low, close, volume, ...]`` (REQ-2.3) or a WebSocket
    kline object ``{"t", "o", "h", "l", "c", "v", "x"}`` (REQ-3.3).
    """

    open_time_ms: int  # epoch ms; equality and ordering key (REQ-4.1, REQ-4.4, REQ-4.5)
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclasses.dataclass(frozen=True)
class StochasticReading:
    """An immutable snapshot of the latest Stochastic Oscillator output.

    Produced by ``compute_stochastic`` from the last row of the pandas-ta
    result and forwarded to the Trigger_Evaluator (REQ-5.6). ``k`` and ``d`` are
    the finite (non-NaN) latest ``STOCHk_5_3_3`` / ``STOCHd_5_3_3`` values;
    ``close`` is the closing price of the Closed_Candle; ``close_time_ms`` is the
    candle close time used in indicator-evaluation log records (REQ-10.5).
    """

    k: float  # latest STOCHk_5_3_3, finite (not NaN)
    d: float  # latest STOCHd_5_3_3, finite (not NaN)
    close: float  # close price of the Closed_Candle (REQ-5.6)
    close_time_ms: int  # candle close time, for log records (REQ-10.5)


class TriggerDecision(enum.Enum):
    """Outcomes of the cool-down trigger state machine.

    Each member's string value is the exact human-readable decision recorded in
    indicator-evaluation log records, so the enum value can be logged directly
    (REQ-10.5).
    """

    FIRE = "alert dispatched"  # k < threshold and not previously oversold (REQ-6.2)
    SUPPRESS = "suppressed by cool-down"  # oversold and k <= threshold (REQ-6.3)
    RESET = "cool-down released"  # oversold and k > threshold (REQ-6.4)
    QUIET = "no oversold condition"  # k >= threshold and not oversold


@dataclasses.dataclass
class MonitorState:
    """The single mutable container shared across components.

    Lives in module scope so its contents survive WebSocket session restarts:
    on reconnect the candle buffer and ``is_oversold`` cool-down lock are
    preserved rather than reset (REQ-8.4). ``is_oversold`` is the boolean
    Cool_Down_Lock, initialized to ``False`` at startup (REQ-6.1). ``logger`` is
    the shared Logger, assigned during ``main`` after ``init_logger`` runs.
    """

    buffer: list[Candle]  # bounded, chronological candle buffer (REQ-2.2, REQ-4)
    is_oversold: bool = False  # Cool_Down_Lock, starts False (REQ-6.1)
    logger: logging.Logger | None = None  # shared Logger, set in main()


# =============================================================================
# Candle Buffer Management
# =============================================================================


def update_buffer(buffer: list[Candle], candle: Candle) -> list[Candle]:
    """Update the bounded, chronological candle buffer with a Closed_Candle.

    Purpose:
        Apply one of three mutually exclusive operations to the candle buffer
        when a Closed_Candle arrives, preserving the buffer invariants
        (REQ-4.1..REQ-4.5, REQ-1.6). The buffer is kept sorted strictly
        ascending by ``open_time_ms`` with unique open times and a length capped
        at :data:`HISTORY_LIMIT` (50).

    Inputs:
        buffer: The current candle buffer, sorted strictly ascending by
            ``open_time_ms`` with unique open times (may be empty).
        candle: The incoming Closed_Candle to apply.

    Returns / side effects:
        Returns the resulting buffer (mutated in place). The applied operation
        depends on ``candle.open_time_ms`` relative to the existing entries:

        * Replace (REQ-4.4): when an existing entry shares the same
          ``open_time_ms``, that entry is overwritten with ``candle`` in place,
          leaving the buffer length and ordering unchanged.
        * Append (REQ-4.1): when ``candle.open_time_ms`` is strictly greater
          than every existing entry's open time (always true for an empty
          buffer), ``candle`` is appended to the end. Leading (oldest) entries
          are then dropped until ``len(buffer) <= HISTORY_LIMIT`` (REQ-4.2).
        * Discard (REQ-4.5): when ``candle.open_time_ms`` is earlier than the
          latest open time and no existing entry shares its open time, the
          buffer is returned unchanged.

        Strict ascending order by ``open_time_ms`` is maintained across every
        operation (REQ-4.3).
    """
    new_open_time = candle.open_time_ms

    # Replace case (REQ-4.4): an entry with the same open time already exists.
    # Overwrite it in place so the buffer length and ordering are unchanged.
    # This is checked first because a matching open time takes precedence over
    # the "earlier than latest" discard rule (REQ-4.5).
    for index, existing in enumerate(buffer):
        if existing.open_time_ms == new_open_time:
            buffer[index] = candle
            return buffer

    # Append case (REQ-4.1): strictly newer than every existing candle. The
    # buffer is sorted ascending, so the latest open time is buffer[-1]; an
    # empty buffer satisfies the condition vacuously.
    if not buffer or new_open_time > buffer[-1].open_time_ms:
        buffer.append(candle)
        # Trim oldest (leading) entries until the bounded-length invariant holds
        # (REQ-4.2, REQ-1.6).
        while len(buffer) > HISTORY_LIMIT:
            buffer.pop(0)
        return buffer

    # Discard case (REQ-4.5): earlier than the latest open time with no matching
    # entry. Leave the buffer untouched.
    return buffer


# =============================================================================
# Stochastic Oscillator Computation
# =============================================================================


def compute_stochastic(buffer: list[Candle]) -> StochasticReading | None:
    """Compute the latest Stochastic Oscillator (%K, %D) over the candle buffer.

    Purpose:
        Recompute the Stochastic Oscillator with the configured parameters
        (%K Length 5, %K Smoothing 3, %D Smoothing 3) on every Closed_Candle and
        return the latest %K / %D values for the Trigger_Evaluator
        (REQ-5.1..REQ-5.6).

        This implementation uses NATIVE vanilla pandas ``.rolling()`` window
        functions rather than ``pandas-ta``. The runtime targets Python 3.14+,
        where ``numba`` (a ``pandas-ta`` build dependency) refuses to build, so
        ``pandas-ta`` is deprecated entirely. The math below reproduces the same
        STOCH(5, 3, 3) calculation and writes its results into the
        :data:`STOCH_K_COL` / :data:`STOCH_D_COL` columns so the column names
        stay meaningful and consistent with the rest of the Monitor (REQ-1.5).

    Inputs:
        buffer: The candle buffer, sorted strictly ascending by ``open_time_ms``
            (the most recent candle is ``buffer[-1]``). At least ~8 candles are
            needed before the smoothed %K / %D rows are non-NaN; shorter buffers
            simply yield ``None`` (warm-up).

    Returns / side effects:
        Returns a :class:`StochasticReading` whose ``k`` and ``d`` are the
        last-row values of the :data:`STOCH_K_COL` / :data:`STOCH_D_COL` columns,
        ``close`` is the closing price of the most recent candle, and
        ``close_time_ms`` is that candle's open time advanced by one interval
        (REQ-5.2, REQ-5.3, REQ-5.6).

        Returns ``None`` (skip trigger evaluation) when:

        * either column is missing from the computed DataFrame, or
        * the last-row %K or %D value is ``NaN``. NaN arises during warm-up and
          also when ``high_max == low_min`` over the %K window (a flat-price
          window makes the ``(close - low_min) / (high_max - low_min)`` division
          ``0 / 0``), which must be treated as "skip" (REQ-5.4).

        On any raised exception the computation is wrapped in ``try``/``except``:
        a WARNING is logged under category ``INDICATOR`` via the shared Logger
        and ``None`` is returned, so a single bad computation never terminates
        the Monitor (REQ-5.5).
    """
    try:
        # Build the OHLCV DataFrame from the buffer in chronological order so
        # the last row corresponds to the most recently closed candle.
        df = pd.DataFrame(
            {
                "open": [c.open for c in buffer],
                "high": [c.high for c in buffer],
                "low": [c.low for c in buffer],
                "close": [c.close for c in buffer],
                "volume": [c.volume for c in buffer],
            }
        )

        # Native Stochastic Oscillator (5, 3, 3) via vanilla pandas rolling
        # windows (no pandas-ta / numba). REQ-5.1.
        # 1. Rolling low/high over the %K length window.
        low_min = df["low"].rolling(window=STOCH_K_LENGTH).min()
        high_max = df["high"].rolling(window=STOCH_K_LENGTH).max()
        # 2. Fast %K. When high_max == low_min the denominator is 0, producing
        #    NaN/inf; the NaN check below treats that flat-price window as skip
        #    (REQ-5.4).
        fast_k = 100 * ((df["close"] - low_min) / (high_max - low_min))
        # 3. Smoothed %K (slow %K) = SMA of fast %K over the %K smoothing window.
        stoch_k_series = fast_k.rolling(window=STOCH_K_SMOOTH).mean()
        # 4. %D = SMA of smoothed %K over the %D smoothing window.
        stoch_d_series = stoch_k_series.rolling(window=STOCH_D_SMOOTH).mean()

        # Assign onto the DataFrame under the configured column names so the
        # output columns stay meaningful and match the rest of the Monitor
        # (REQ-1.5).
        df[STOCH_K_COL] = stoch_k_series
        df[STOCH_D_COL] = stoch_d_series

        # Skip if either column is missing from the computed output (REQ-5.4).
        if STOCH_K_COL not in df.columns or STOCH_D_COL not in df.columns:
            return None

        # Read the latest %K / %D from the last row (REQ-5.2, REQ-5.3).
        latest_k = float(df[STOCH_K_COL].iloc[-1])
        latest_d = float(df[STOCH_D_COL].iloc[-1])

        # Skip when either value is NaN: warm-up rows or a flat-price window
        # (high_max == low_min) yield NaN, which must suppress trigger
        # evaluation (REQ-5.4).
        if math.isnan(latest_k) or math.isnan(latest_d):
            return None

        # Success: forward the latest reading for trigger evaluation (REQ-5.6).
        return StochasticReading(
            k=latest_k,
            d=latest_d,
            close=buffer[-1].close,
            close_time_ms=buffer[-1].open_time_ms + STALE_BUFFER_THRESHOLD_MS,
        )
    except Exception as exc:
        # Log-and-continue: any failure computing the indicator skips trigger
        # evaluation for this candle without terminating the Monitor (REQ-5.5).
        logging.getLogger(LOGGER_NAME).warning(
            "Stochastic computation failed (%s: %s); skipping trigger "
            "evaluation for this candle",
            type(exc).__name__,
            exc,
            extra={"category": "INDICATOR"},
        )
        return None
