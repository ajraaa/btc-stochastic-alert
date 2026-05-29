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
import functools  # binds MonitorState to WS callbacks in run_forever (REQ-8.2)
import json
import logging
import math
import sys
import time
import typing

# --- Third-party imports -----------------------------------------------------
import pandas as pd
import requests  # Telegram Bot HTTP API client for the Notifier (REQ-7.4)

# WebSocketApp is imported to module scope (rather than referenced as
# ``websocket.WebSocketApp``) so ``run_forever`` can call ``WebSocketApp(...)``
# directly and tests can patch ``btc_stochastic_monitor.WebSocketApp`` with a
# fake for the reconnect / integration suites (REQ-3.1, REQ-8).
from websocket import WebSocketApp

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


# =============================================================================
# Cool-Down Trigger State Machine
# =============================================================================


def evaluate_trigger(state: MonitorState, reading: StochasticReading) -> TriggerDecision:
    """Evaluate the oversold cool-down state machine for one Stochastic reading.

    Purpose:
        Decide, for a single forwarded :class:`StochasticReading`, whether the
        Monitor should fire an oversold alert, suppress under the active
        Cool_Down_Lock, release the lock, or stay quiet (REQ-6.1..REQ-6.5).
        This is a PURE state-machine function over the pair
        ``(state.is_oversold, reading.k)``: its only side effect is the
        documented mutation of ``state.is_oversold`` below. It does NOT send
        notifications or emit log records; the caller ``process_data`` is
        responsible for logging the decision and dispatching the alert.

    Inputs:
        state: The shared :class:`MonitorState`; only ``state.is_oversold`` (the
            boolean Cool_Down_Lock, REQ-6.1) is read and possibly mutated.
        reading: The latest :class:`StochasticReading`; only ``reading.k`` (the
            latest %K value) participates in the decision.

    Returns / side effects:
        Returns the :class:`TriggerDecision` for this reading. The four cases
        are mutually exclusive and total over ``(is_oversold, k)``:

        * ``FIRE`` — engagement: ``k < OVERSOLD_THRESHOLD`` and not yet oversold;
          mutates ``state.is_oversold = True`` (REQ-6.2). Engagement uses strict
          ``<`` so ``k == OVERSOLD_THRESHOLD`` does NOT fire.
        * ``SUPPRESS`` — already oversold and ``k <= OVERSOLD_THRESHOLD``; no
          state mutation (REQ-6.3).
        * ``RESET`` — already oversold and ``k > OVERSOLD_THRESHOLD``; mutates
          ``state.is_oversold = False`` (REQ-6.4). Release uses strict ``>`` so
          ``k == OVERSOLD_THRESHOLD`` does NOT reset.
        * ``QUIET`` — none of the above (not oversold and ``k >= threshold``); no
          state mutation.

        The trigger is evaluated only after the Indicator_Engine has completed
        evaluation for a Closed_Candle (REQ-6.5); ``process_data`` enforces that
        ordering by calling this function only on a non-``None`` reading.
    """
    # Engagement (REQ-6.2): %K is strictly below the threshold and the
    # Cool_Down_Lock is currently disengaged. Engage the lock and fire exactly
    # one alert. Strict ``<`` means k == OVERSOLD_THRESHOLD does NOT fire.
    # Resulting state: is_oversold transitions False -> True.
    if not state.is_oversold and reading.k < OVERSOLD_THRESHOLD:
        state.is_oversold = True  # lock engaged; suppresses further alerts
        return TriggerDecision.FIRE

    # Suppression check (REQ-6.3): the lock is already engaged and %K is still at
    # or below the threshold (``<=``, so the boundary k == OVERSOLD_THRESHOLD
    # suppresses rather than resets). Stay silent without mutating state.
    # Resulting state: is_oversold unchanged (remains True).
    if state.is_oversold and reading.k <= OVERSOLD_THRESHOLD:
        return TriggerDecision.SUPPRESS

    # Reset (REQ-6.4): the lock is engaged and %K has risen strictly above the
    # threshold (``>``, so k == OVERSOLD_THRESHOLD does NOT reset). Release the
    # lock so the next oversold crossing can fire again.
    # Resulting state: is_oversold transitions True -> False.
    if state.is_oversold and reading.k > OVERSOLD_THRESHOLD:
        state.is_oversold = False  # lock released; re-armed for next episode
        return TriggerDecision.RESET

    # Quiet: not oversold and %K is at or above the threshold. No oversold
    # condition and no state change.
    # Resulting state: is_oversold unchanged (remains False).
    return TriggerDecision.QUIET


# =============================================================================
# Notifier (Telegram delivery)
# =============================================================================


def format_alert(symbol: str, interval: str, reading: StochasticReading) -> str:
    """Build the human-readable Telegram alert text for an oversold signal.

    Purpose:
        Render the message body the Notifier delivers when the Trigger_Evaluator
        fires an oversold alert, surfacing every field a trader needs to act:
        the asset, the timeframe, the latest %K and %D (each to two decimals),
        and the closing price (REQ-7.2).

    Inputs:
        symbol: The trading symbol (e.g. :data:`SYMBOL`, ``"btcusdt"``).
        interval: The kline interval (e.g. :data:`INTERVAL`, ``"1d"``).
        reading: The :class:`StochasticReading` whose ``k``, ``d``, and ``close``
            are embedded in the message. ``k`` and ``d`` are formatted with the
            ``"{:.2f}"`` two-decimal pattern; ``close`` is rendered as a numeric
            value.

    Returns / side effects:
        Returns the formatted message string. No side effects.
    """
    # Two-decimal formatting for %K and %D per REQ-7.2; ``close`` is rendered as
    # a plain numeric value so the price is unambiguous.
    return (
        f"BTC Stochastic DCA alert: {symbol} @ {interval} is oversold. "
        f"%K={reading.k:.2f} %D={reading.d:.2f} close={reading.close}"
    )


def send_telegram_notification(message: str) -> bool:
    """Deliver a message to the configured Telegram chat via the Bot HTTP API.

    Purpose:
        Invoke the Notifier to POST ``message`` to the Telegram ``sendMessage``
        endpoint using the operator-supplied bot token and chat ID. The function
        follows the Monitor's "log and continue" philosophy: it never raises, so
        a delivery failure can never crash the Monitor (REQ-7.1, REQ-7.3..7.6).

    Inputs:
        message: The alert text to deliver (typically produced by
            :func:`format_alert`).

    Returns / side effects:
        Returns ``True`` when Telegram responds with a 2xx status, ``False``
        otherwise (missing config, non-2xx response, or a transport exception).
        Emits exactly one log record through the shared Logger describing the
        outcome:

        * Missing/empty token or chat ID at send time -> CONFIG-category ERROR,
          POST skipped, returns ``False`` (REQ-7.6).
        * Non-2xx HTTP response -> NOTIFY-category ERROR including the status
          code, returns ``False`` (REQ-7.3).
        * ``requests.RequestException`` (connection/timeout) -> NOTIFY-category
          ERROR including the exception type and message, returns ``False``
          (REQ-7.5).
        * Success -> NOTIFY-category INFO recording dispatch success, returns
          ``True`` (REQ-10.6).
    """
    logger = logging.getLogger(LOGGER_NAME)

    # Re-validate credentials at send time so a token/chat ID cleared after
    # startup (or a test monkeypatching the module globals) is caught here
    # rather than producing a malformed request. On missing config, log a
    # CONFIG error, skip the POST, and return False (REQ-7.6).
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_BOT_TOKEN.strip():
        logger.error(
            "Cannot dispatch Telegram notification: TELEGRAM_BOT_TOKEN is "
            "missing or empty. Skipping send.",
            extra={"category": "CONFIG"},
        )
        return False

    if not TELEGRAM_CHAT_ID or not TELEGRAM_CHAT_ID.strip():
        logger.error(
            "Cannot dispatch Telegram notification: TELEGRAM_CHAT_ID is "
            "missing or empty. Skipping send.",
            extra={"category": "CONFIG"},
        )
        return False

    # Build the endpoint URL from the current token value so monkeypatched
    # credentials take effect (do NOT capture the token at import time).
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
            timeout=TELEGRAM_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        # Connection/timeout/transport failure: log type + message and continue
        # without crashing the Monitor (REQ-7.5).
        logger.error(
            "Telegram notification dispatch failed (%s: %s); continuing.",
            type(exc).__name__,
            exc,
            extra={"category": "NOTIFY"},
        )
        return False

    # Treat any 2xx as success; otherwise log the HTTP status code and continue
    # without crashing (REQ-7.3).
    status_code = response.status_code
    if not 200 <= status_code < 300:
        logger.error(
            "Telegram notification dispatch returned non-success HTTP status "
            "%s; continuing.",
            status_code,
            extra={"category": "NOTIFY"},
        )
        return False

    # Success: record dispatch completion (REQ-10.6).
    logger.info(
        "Telegram notification dispatched successfully (HTTP %s).",
        status_code,
        extra={"category": "NOTIFY"},
    )
    return True


# =============================================================================
# Historical Data Bootstrapping
# =============================================================================


def bootstrap_history(state: MonitorState) -> None:
    """Seed the candle buffer with recent historical 1-day klines from Binance.

    Purpose:
        Run the Bootstrapper at startup (and again on stale-buffer recovery) so
        the Stochastic Oscillator has enough warm-up candles to produce finite
        %K / %D values immediately, instead of waiting days for live candles to
        accumulate (REQ-2.1..REQ-2.6). The Monitor opens the WebSocket_Client
        only after this function returns, so a populated buffer is guaranteed
        before live ingestion begins (REQ-2.4).

    Inputs:
        state: The shared :class:`MonitorState`. On success ``state.buffer`` is
            replaced with the parsed historical candles in chronological order;
            no other field is touched.

    Returns / side effects:
        Returns ``None`` once a single valid response has been parsed and
        assigned to ``state.buffer``. Issues HTTPS GET requests to
        :data:`BINANCE_REST_URL` with query parameters ``symbol=BTCUSDT``
        (uppercase per REQ-2.1; the WebSocket stream uses the lowercase
        :data:`SYMBOL`), ``interval=INTERVAL``, and ``limit=HISTORY_LIMIT``, with
        a :data:`REST_TIMEOUT_S` request timeout.

    Retry behaviour (REQ-2.5, REQ-2.6):
        Loops until the first valid response. Each of the following failure
        modes is logged distinctly as a BOOTSTRAP-category ERROR through the
        shared Logger, followed by ``time.sleep(RECONNECT_DELAY_S)`` before the
        next attempt:

        * a network or timeout exception from ``requests.get``
          (:class:`requests.RequestException`),
        * a non-200 HTTP status code,
        * a body that cannot be parsed as JSON,
        * a parsed body that is not a JSON array (list),
        * an array with fewer than 10 entries,
        * a row that cannot be parsed into a :class:`Candle`
          (``ValueError`` / ``IndexError`` / ``TypeError``).

        Once a valid response is parsed the function assigns ``state.buffer`` and
        returns immediately; it never issues an additional request after the
        first success in a startup cycle (REQ-2.6).
    """
    logger = logging.getLogger(LOGGER_NAME)

    # Minimum number of historical entries required before the buffer is
    # considered usable for warm-up (REQ-2.2, REQ-2.5).
    min_entries = 10

    # Retry loop: keep requesting until one valid response is parsed. Every
    # failure mode logs a BOOTSTRAP error, sleeps the Reconnect_Delay, and
    # continues; success assigns the buffer and returns (REQ-2.5, REQ-2.6).
    while True:
        # --- Issue the request (REQ-2.1) -------------------------------------
        # ``symbol`` is the uppercase "BTCUSDT" mandated by REQ-2.1 for the REST
        # endpoint, distinct from the lowercase SYMBOL used by the WS stream.
        try:
            response = requests.get(
                BINANCE_REST_URL,
                params={
                    "symbol": "BTCUSDT",
                    "interval": INTERVAL,
                    "limit": HISTORY_LIMIT,
                },
                timeout=REST_TIMEOUT_S,
            )
        except requests.RequestException as exc:
            # Failure mode 1: network/timeout exception (REQ-2.5).
            logger.error(
                "Bootstrap request failed (%s: %s); retrying after %s s.",
                type(exc).__name__,
                exc,
                RECONNECT_DELAY_S,
                extra={"category": "BOOTSTRAP"},
            )
            time.sleep(RECONNECT_DELAY_S)
            continue

        # --- Failure mode 2: non-200 HTTP status (REQ-2.5) -------------------
        if response.status_code != 200:
            logger.error(
                "Bootstrap request returned non-200 HTTP status %s; retrying "
                "after %s s.",
                response.status_code,
                RECONNECT_DELAY_S,
                extra={"category": "BOOTSTRAP"},
            )
            time.sleep(RECONNECT_DELAY_S)
            continue

        # --- Failure mode 3: body not parseable as JSON (REQ-2.5) ------------
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error(
                "Bootstrap response body is not valid JSON (%s: %s); retrying "
                "after %s s.",
                type(exc).__name__,
                exc,
                RECONNECT_DELAY_S,
                extra={"category": "BOOTSTRAP"},
            )
            time.sleep(RECONNECT_DELAY_S)
            continue

        # --- Failure mode 4: parsed JSON is not an array (REQ-2.5) -----------
        if not isinstance(payload, list):
            logger.error(
                "Bootstrap response JSON is not an array (got %s); retrying "
                "after %s s.",
                type(payload).__name__,
                RECONNECT_DELAY_S,
                extra={"category": "BOOTSTRAP"},
            )
            time.sleep(RECONNECT_DELAY_S)
            continue

        # --- Failure mode 5: fewer than 10 entries (REQ-2.5) -----------------
        if len(payload) < min_entries:
            logger.error(
                "Bootstrap response array has only %s entries (need at least "
                "%s); retrying after %s s.",
                len(payload),
                min_entries,
                RECONNECT_DELAY_S,
                extra={"category": "BOOTSTRAP"},
            )
            time.sleep(RECONNECT_DELAY_S)
            continue

        # --- Parse rows into Candles (REQ-2.3) -------------------------------
        # Each Binance kline row is a list like
        # [openTime(int ms), open(str), high(str), low(str), close(str),
        #  volume(str), closeTime, ...]. Convert defensively; a malformed row
        # (short list, non-numeric field) raises ValueError/IndexError/
        # TypeError and is treated as a failure mode -> log + sleep + retry.
        try:
            candles = [
                Candle(
                    open_time_ms=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
                for row in payload
            ]
        except (ValueError, IndexError, TypeError) as exc:
            logger.error(
                "Bootstrap response row could not be parsed into a Candle "
                "(%s: %s); retrying after %s s.",
                type(exc).__name__,
                exc,
                RECONNECT_DELAY_S,
                extra={"category": "BOOTSTRAP"},
            )
            time.sleep(RECONNECT_DELAY_S)
            continue

        # --- Success (REQ-2.2, REQ-2.6) --------------------------------------
        # Sort ascending by open time so the buffer satisfies the chronological
        # invariant shared with ``update_buffer`` (REQ-4.3), then assign and
        # return without issuing any further request this startup cycle.
        candles.sort(key=lambda c: c.open_time_ms)
        state.buffer = candles
        logger.info(
            "Bootstrap succeeded: seeded candle buffer with %s historical "
            "candles.",
            len(candles),
            extra={"category": "BOOTSTRAP"},
        )
        return


# =============================================================================
# Indicator + Trigger Orchestrator and WebSocket Callbacks
# -----------------------------------------------------------------------------
# These four functions tie the pure data transformations (buffer update,
# indicator computation, trigger state machine) to the live WebSocket stream.
#
# State-binding convention (IMPORTANT, REQ-3.2, REQ-8.4):
#   ``on_open``, ``on_message``, and ``on_close`` are registered as
#   ``websocket.WebSocketApp`` callbacks. The WebSocketApp invokes them as
#   ``on_open(ws)``, ``on_message(ws, raw)``, and ``on_close(ws, code, reason)``
#   -- i.e. WITHOUT the shared MonitorState. To give them access to the state
#   that must survive session restarts, each is defined here with ``state`` as
#   its FIRST positional parameter, and the reconnection loop (task 12.1) binds
#   it via ``functools.partial(on_message, state)``. The partial yields exactly
#   the ``(ws, raw)`` callable the WebSocketApp expects, while the buffer and
#   ``is_oversold`` lock live on the long-lived ``MonitorState`` outside the
#   session lifecycle (REQ-8.4).
# =============================================================================


def process_data(state: MonitorState, candle: Candle) -> None:
    # Purpose: orchestrate a single Closed_Candle through the indicator and
    #   trigger pipeline -- update the buffer, recompute the Stochastic
    #   Oscillator, evaluate the cool-down trigger, dispatch a Telegram alert on
    #   FIRE, and emit exactly one indicator-evaluation log record.
    # Inputs: ``state`` (shared MonitorState; ``buffer`` and ``is_oversold`` may
    #   mutate) and ``candle`` (the Closed_Candle to ingest).
    # Returns / side effects: returns None. Mutates ``state.buffer`` (via
    #   ``update_buffer``) and possibly ``state.is_oversold`` (via
    #   ``evaluate_trigger``); may POST a Telegram notification; always emits one
    #   INDICATOR-category INFO record for the candle (REQ-9.2, REQ-10.4,
    #   REQ-10.5).
    """Run one Closed_Candle through indicator computation and trigger evaluation.

    Purpose:
        Implement the ``process_data`` orchestrator referenced throughout the
        design's Data Flow sequence: apply the Closed_Candle to the bounded
        candle buffer (REQ-4), recompute the Stochastic Oscillator over the
        buffer (REQ-5), and -- only when a finite reading is produced -- evaluate
        the cool-down trigger (REQ-6) and dispatch a Telegram alert on a ``FIRE``
        decision (REQ-7). Exactly one INDICATOR-category log record is emitted
        per Closed_Candle so an operator can audit every evaluation (REQ-10.4,
        REQ-10.5).

    Inputs:
        state: The shared :class:`MonitorState`. ``state.buffer`` is reassigned
            by :func:`update_buffer`; ``state.is_oversold`` may be mutated by
            :func:`evaluate_trigger`.
        candle: The Closed_Candle (``"x" == True``) forwarded by
            :func:`on_message`.

    Returns / side effects:
        Returns ``None``. Side effects:

        * ``state.buffer = update_buffer(state.buffer, candle)`` (REQ-4).
        * When :func:`compute_stochastic` returns ``None`` (warm-up / NaN /
          computation error, REQ-5.4, REQ-5.5): emits a single INDICATOR INFO
          record noting the evaluation was skipped (no finite reading) and
          returns without touching the trigger (REQ-6.5).
        * Otherwise calls :func:`evaluate_trigger`; on a ``FIRE`` decision
          dispatches ``send_telegram_notification(format_alert(...))`` (REQ-6.2,
          REQ-7) and records the dispatch outcome. Emits exactly one INDICATOR
          INFO record carrying ``close_time_ms``, ``%K``, ``%D``, and the trigger
          decision string (plus the dispatch outcome on ``FIRE``) per REQ-10.5.
    """
    logger = logging.getLogger(LOGGER_NAME)

    # Step 1 -- buffer update (REQ-4). ``update_buffer`` appends/replaces/discards
    # and trims to HISTORY_LIMIT while preserving chronological order.
    state.buffer = update_buffer(state.buffer, candle)

    # Step 2 -- recompute the Stochastic Oscillator over the updated buffer
    # (REQ-5.1). A ``None`` reading means warm-up rows, a NaN last row, a missing
    # column, or a computation error -- all of which skip trigger evaluation
    # (REQ-5.4, REQ-5.5).
    reading = compute_stochastic(state.buffer)

    if reading is None:
        # No finite reading: emit one INDICATOR record noting the skip so the
        # closed candle is still auditable, then return without evaluating the
        # trigger (REQ-6.5). There are no %K/%D values to log in this case.
        logger.info(
            "Indicator evaluation skipped for closed candle "
            "(open_time_ms=%s): no finite Stochastic reading (warm-up or NaN).",
            candle.open_time_ms,
            extra={"category": "INDICATOR"},
        )
        return

    # Step 3 -- evaluate the cool-down trigger only after a finite reading is in
    # hand (REQ-6.5). ``evaluate_trigger`` returns the decision and may mutate
    # ``state.is_oversold``.
    decision = evaluate_trigger(state, reading)

    if decision == TriggerDecision.FIRE:
        # Step 4 -- engagement fired: dispatch exactly one Telegram alert
        # (REQ-6.2, REQ-7). ``send_telegram_notification`` never raises and emits
        # its own NOTIFY dispatch record (REQ-10.6); we capture its boolean
        # outcome to include in this candle's indicator record.
        dispatched = send_telegram_notification(
            format_alert(SYMBOL, INTERVAL, reading)
        )
        # One INDICATOR record for this candle including the dispatch outcome
        # (REQ-10.4, REQ-10.5).
        logger.info(
            "Indicator evaluation complete: close_time_ms=%s %%K=%.4f "
            "%%D=%.4f decision=%r dispatched=%s",
            reading.close_time_ms,
            reading.k,
            reading.d,
            decision.value,
            dispatched,
            extra={"category": "INDICATOR"},
        )
        return

    # Non-FIRE decision (SUPPRESS / RESET / QUIET): no alert dispatched. Emit the
    # single INDICATOR record with the decision string for this candle (REQ-10.4,
    # REQ-10.5).
    logger.info(
        "Indicator evaluation complete: close_time_ms=%s %%K=%.4f %%D=%.4f "
        "decision=%r",
        reading.close_time_ms,
        reading.k,
        reading.d,
        decision.value,
        extra={"category": "INDICATOR"},
    )


def on_open(state: MonitorState, ws: typing.Any) -> None:
    # Purpose: WebSocket ``on_open`` callback; logs that a live kline session has
    #   been established on the subscribed Binance stream.
    # Inputs: ``state`` (shared MonitorState, bound first via functools.partial)
    #   and ``ws`` (the WebSocketApp instance passed by the client library).
    # Returns / side effects: returns None; emits one WS-category INFO record
    #   including BINANCE_WS_URL (REQ-3.1, REQ-9.3).
    """Handle WebSocket session establishment.

    Purpose:
        Mark the moment a live kline session opens so operators can correlate
        reconnects with stream activity (REQ-3.1, REQ-9.3).

    Inputs:
        state: The shared :class:`MonitorState`, bound as the first positional
            argument via ``functools.partial`` (see the state-binding convention
            above). Not mutated here.
        ws: The ``WebSocketApp`` instance supplied by the client library
            (unused beyond logging).

    Returns / side effects:
        Returns ``None``. Emits one WS-category INFO record including the
        subscribed :data:`BINANCE_WS_URL` (REQ-3.1).
    """
    logging.getLogger(LOGGER_NAME).info(
        "WebSocket session opened on %s",
        BINANCE_WS_URL,
        extra={"category": "WS"},
    )


def on_message(state: MonitorState, ws: typing.Any, raw: str) -> None:
    # Purpose: WebSocket ``on_message`` callback; parse an inbound kline payload
    #   and forward only finalized (closed) candles to ``process_data``.
    # Inputs: ``state`` (shared MonitorState, bound first via functools.partial),
    #   ``ws`` (WebSocketApp instance), and ``raw`` (the raw message body string).
    # Returns / side effects: returns None. Forwards to ``process_data`` only on
    #   a closed candle (``"k.x" == True``); logs a WS WARNING on JSON parse
    #   failure or a malformed closed kline; returns silently for non-closed
    #   candles, missing ``"k"``, or missing ``"k.x"`` (REQ-3.3..REQ-3.7).
    """Parse an inbound WebSocket message and route Closed_Candles downstream.

    Purpose:
        Implement the WebSocket_Client message handler from the design's Data
        Flow: parse JSON, extract the ``"k"`` kline object, and forward a
        :class:`Candle` to :func:`process_data` if and only if the kline is
        closed (``"x" == True``). All other payloads are ignored, with malformed
        JSON logged as a parse failure (REQ-3.3..REQ-3.7).

    Inputs:
        state: The shared :class:`MonitorState`, bound as the first positional
            argument via ``functools.partial`` (see the state-binding convention
            above). Passed through to :func:`process_data`.
        ws: The ``WebSocketApp`` instance supplied by the client library
            (unused).
        raw: The raw message body (a JSON string) delivered by the stream.

    Returns / side effects:
        Returns ``None`` in every branch. Control flow:

        1. ``json.loads(raw)``; on :class:`json.JSONDecodeError` log a
           WS-category WARNING and return without forwarding (REQ-3.6).
        2. Extract ``k = payload["k"]`` (only when ``payload`` is a dict). If the
           kline object is missing or is not a dict, return silently (REQ-3.7).
        3. If ``"x"`` is absent from the kline object, return silently (REQ-3.7).
        4. If ``"x"`` is not ``True`` (a non-closed candle), return silently
           without forwarding (REQ-3.4).
        5. If ``"x"`` is ``True``, build a :class:`Candle` from the
           ``t/o/h/l/c/v`` fields and call ``process_data(state, candle)``
           (REQ-3.3, REQ-3.5). A malformed closed kline (missing or non-numeric
           field) is caught, logged as a WS-category WARNING, and skipped rather
           than crashing the Monitor (defensive, aligns with REQ-3.7).
    """
    logger = logging.getLogger(LOGGER_NAME)

    # Step 1 -- parse the body as JSON. A non-JSON / truncated frame is logged
    # once and skipped without forwarding any data downstream (REQ-3.6).
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "Could not parse WebSocket message as JSON; skipping message.",
            extra={"category": "WS"},
        )
        return

    # Step 2 -- extract the kline object. Binance wraps the kline under "k". A
    # payload that is not a dict, or that lacks "k", carries no kline object and
    # is skipped silently (REQ-3.7).
    k = payload.get("k") if isinstance(payload, dict) else None
    if not isinstance(k, dict):
        return

    # Step 3 -- the kline object must carry the "x" (is-closed) flag; without it
    # we cannot tell whether the candle is final, so skip silently (REQ-3.7).
    if "x" not in k:
        return

    # Step 4 -- only finalized candles update the indicator. A non-closed candle
    # (``"x" == False``) is intentionally NOT forwarded (REQ-3.4). Strict
    # identity against ``True`` ignores non-closed / unexpected values.
    if k["x"] is not True:
        return

    # Step 5 -- closed candle: build the immutable Candle from the kline fields
    # (REQ-3.3) and forward it (REQ-3.5). Guard against a malformed closed kline
    # (missing or non-numeric o/h/l/c/v/t): log + skip instead of crashing.
    try:
        candle = Candle(
            open_time_ms=int(k["t"]),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "Malformed closed kline payload (%s: %s); skipping message.",
            type(exc).__name__,
            exc,
            extra={"category": "WS"},
        )
        return

    process_data(state, candle)


def on_close(
    state: MonitorState,
    ws: typing.Any,
    code: int | None,
    reason: str | None,
) -> None:
    # Purpose: WebSocket ``on_close`` callback; logs session termination so the
    #   outer reconnection loop (task 12.1) can correlate disconnects.
    # Inputs: ``state`` (shared MonitorState, bound first via functools.partial),
    #   ``ws`` (WebSocketApp instance), ``code`` (close status code or None), and
    #   ``reason`` (close reason text or None).
    # Returns / side effects: returns None; emits one WS-category INFO record
    #   including the close ``code`` and ``reason`` (REQ-8.6). The reconnection
    #   loop owns the retry/backoff (REQ-8.7).
    """Handle WebSocket session termination.

    Purpose:
        Record the close code and reason for each ended session so an operator
        can diagnose disconnects; the outer ``run_forever`` reconnection loop
        (task 12.1) is responsible for the actual retry (REQ-8.6, REQ-8.7).

    Inputs:
        state: The shared :class:`MonitorState`, bound as the first positional
            argument via ``functools.partial`` (see the state-binding convention
            above). Not mutated here -- the buffer and ``is_oversold`` lock are
            deliberately preserved across the disconnect (REQ-8.4).
        ws: The ``WebSocketApp`` instance supplied by the client library
            (unused).
        code: The WebSocket close status code (e.g. ``1006``), or ``None`` when
            the library did not supply one.
        reason: The close reason text, or ``None`` when not supplied.

    Returns / side effects:
        Returns ``None``. Emits one WS-category INFO record including ``code``
        and ``reason`` (REQ-8.6).
    """
    logging.getLogger(LOGGER_NAME).info(
        "WebSocket session closed (code=%r, reason=%r)",
        code,
        reason,
        extra={"category": "WS"},
    )


# =============================================================================
# Buffer Freshness / Gap Detection and the Resilient Reconnection Loop
# =============================================================================


def check_buffer_freshness(state: MonitorState, now_ms: int) -> bool:
    """Decide whether the candle buffer is fresh enough to keep across a reconnect.

    Purpose:
        Implement the gap-detection check from the design's Reconnection Flow
        (REQ-8.5). After a WebSocket disconnect the candle buffer may be stale --
        e.g. the process slept or lost connectivity for longer than one full
        interval -- in which case the indicator output would be computed over old
        data. This function tells :func:`run_forever` whether the buffer can be
        preserved (fresh) or must be discarded and re-bootstrapped (stale).

    Inputs:
        state: The shared :class:`MonitorState`; only ``state.buffer`` is read.
        now_ms: The current wall-clock time in Unix epoch milliseconds (the
            caller supplies ``int(time.time() * 1000)``).

    Returns / side effects:
        Returns ``True`` (fresh) iff ``state.buffer`` is non-empty AND the age of
        the most recent candle -- ``now_ms - state.buffer[-1].open_time_ms`` -- is
        at most :data:`STALE_BUFFER_THRESHOLD_MS` (1 day). Returns ``False``
        (stale, re-bootstrap required) when the buffer is empty or the latest
        candle is older than that threshold. No side effects.
    """
    # An empty buffer has no most-recent candle to age-check; treat it as stale
    # so the caller re-bootstraps before opening a session (REQ-8.5).
    if not state.buffer:
        return False

    # Fresh iff the newest candle's age is within the stale-buffer threshold.
    # ``buffer[-1]`` is the latest candle because the buffer is kept sorted
    # strictly ascending by open_time_ms (REQ-4.3).
    age_ms = now_ms - state.buffer[-1].open_time_ms
    return age_ms <= STALE_BUFFER_THRESHOLD_MS


def run_forever(state: MonitorState) -> None:
    """Run the persistent WebSocket reconnection loop for the lifetime of the Monitor.

    Purpose:
        Drive the runtime half of the design's Reconnection Flow: keep a live
        Binance kline WebSocket session open inside a ``while True`` loop, recover
        transparently from any disconnect or exception after a fixed delay, and
        re-bootstrap whenever the candle buffer has gone stale across a gap
        (REQ-8.1..REQ-8.7, REQ-10.4). The candle buffer and the ``is_oversold``
        Cool_Down_Lock live on the long-lived ``state`` outside the session
        lifecycle, so a normal reconnect preserves them automatically (REQ-8.4).

    Inputs:
        state: The shared :class:`MonitorState`. ``state.buffer`` and
            ``state.is_oversold`` are read for freshness and may be reset/repopulated
            on a stale-buffer recovery; ``state`` is bound to the WebSocket
            callbacks so they can mutate the buffer and lock across sessions.

    Returns / side effects:
        Never returns under normal operation -- the loop is infinite by design
        (REQ-8.1); the process ends only via a fatal signal (see design
        "Termination Conditions"). Each iteration:

        1. Computes ``now_ms = int(time.time() * 1000)`` and calls
           :func:`check_buffer_freshness`. When the buffer is stale (REQ-8.5),
           discards ``state.buffer`` (set to ``[]``), resets
           ``state.is_oversold = False``, logs a RECONNECT-category WARNING, and
           re-runs :func:`bootstrap_history` to repopulate the buffer before a
           new session opens.
        2. Constructs a :data:`WebSocketApp` for :data:`BINANCE_WS_URL` with the
           ``on_open`` / ``on_message`` / ``on_close`` callbacks bound to ``state``
           via :func:`functools.partial` (so the app sees the ``(ws)``,
           ``(ws, raw)``, and ``(ws, code, reason)`` callables it expects, REQ-3.2),
           and calls ``ws.run_forever()`` inside a ``try``/``except Exception`` so
           any connection or runtime error is caught rather than crashing the
           Monitor (REQ-8.2).
        3. On any exception OR a normal session close, increments the
           reconnect-attempt counter, logs a RECONNECT-category INFO record
           carrying that counter (REQ-10.4), and sleeps
           :data:`RECONNECT_DELAY_S` seconds before the next iteration (REQ-8.3).
    """
    logger = logging.getLogger(LOGGER_NAME)

    # Reconnect-attempt counter included in every RECONNECT INFO record so an
    # operator can audit how many sessions have been opened (REQ-10.4). Starts at
    # 0 and is incremented once per loop iteration before logging.
    reconnect_attempts = 0

    # Persistent reconnection loop (REQ-8.1). Infinite by design -- no break.
    while True:
        # --- Step 1: gap detection / stale-buffer recovery (REQ-8.5) ---------
        # Use the current wall-clock time in ms to age the newest candle.
        now_ms = int(time.time() * 1000)
        if not check_buffer_freshness(state, now_ms):
            # The buffer is empty or too old to trust: discard it, release the
            # cool-down lock, and re-bootstrap so the new session starts from a
            # fresh, chronologically-seeded buffer (REQ-8.5).
            state.buffer = []
            state.is_oversold = False
            logger.warning(
                "Candle buffer is stale or empty; discarding buffer, resetting "
                "is_oversold, and re-bootstrapping before reconnecting.",
                extra={"category": "RECONNECT"},
            )
            bootstrap_history(state)

        # --- Step 2: open a WebSocket session (REQ-3.1, REQ-3.2, REQ-8.2) ----
        # Bind the shared state as the first positional argument of each callback
        # so the WebSocketApp receives exactly the (ws), (ws, raw), and
        # (ws, code, reason) callables it invokes, while the buffer/lock survive
        # across sessions on ``state`` (REQ-8.4). ``WebSocketApp`` is referenced
        # as a module-level name so tests can patch it (see import note above).
        try:
            ws = WebSocketApp(
                BINANCE_WS_URL,
                on_open=functools.partial(on_open, state),
                on_message=functools.partial(on_message, state),
                on_close=functools.partial(on_close, state),
            )
            # Blocks for the lifetime of the session; returns on a normal close.
            ws.run_forever()
        except Exception as exc:
            # Any connection/protocol/runtime error is caught so a single failure
            # never crashes the Monitor; it simply triggers a reconnect (REQ-8.2).
            logger.warning(
                "WebSocket session raised an exception (%s: %s); will reconnect.",
                type(exc).__name__,
                exc,
                extra={"category": "RECONNECT"},
            )

        # --- Step 3: reconnect bookkeeping + delay (REQ-8.3, REQ-10.4) -------
        # Reached on BOTH a normal close and a caught exception. Count the
        # attempt, log a RECONNECT record carrying the counter (REQ-10.4), then
        # wait the fixed Reconnect_Delay before opening the next session
        # (REQ-8.3, REQ-8.7).
        reconnect_attempts += 1
        logger.info(
            "WebSocket session ended; reconnecting in %s s "
            "(reconnect attempt #%s).",
            RECONNECT_DELAY_S,
            reconnect_attempts,
            extra={"category": "RECONNECT"},
        )
        time.sleep(RECONNECT_DELAY_S)
