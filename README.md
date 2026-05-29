# BTC Stochastic DCA Monitor

A single-file, production-ready Python script that supports a manual Dollar-Cost Averaging (DCA) strategy on Bitcoin. It streams live 1-day BTC/USDT candles from the Binance public WebSocket, computes the Stochastic Oscillator (%K/%D) using native pandas rolling windows, and sends **exactly one Telegram notification** the moment %K crosses below the oversold threshold of 20. The Monitor runs unattended on a Linux VPS with automatic reconnection, historical bootstrapping, and a cool-down lock that prevents notification spam.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Signal Logic](#signal-logic)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Monitor](#running-the-monitor)
- [Smoke Check](#smoke-check)
- [Running as a systemd Service](#running-as-a-systemd-service)
- [Logging](#logging)
- [Project Structure](#project-structure)
- [Development & Testing](#development--testing)
- [Test Coverage](#test-coverage)
- [Design Decisions](#design-decisions)

---

## How It Works

```
Startup
  │
  ├─ validate_config()          ← fail-fast: checks Python ≥ 3.10 and credentials
  ├─ init_logger()              ← stdout + backup log file, UTC timestamps
  ├─ bootstrap_history()        ← seeds 50 historical 1-day candles via Binance REST
  │
  └─ run_forever()              ← while True reconnection loop
       │
       ├─ check_buffer_freshness()   ← if stale (> 1 day gap), re-bootstrap
       ├─ WebSocketApp.run_forever() ← streams btcusdt@kline_1d
       │    │
       │    ├─ on_message()          ← filters x=false (non-closed) candles
       │    │    └─ process_data()
       │    │         ├─ update_buffer()        ← append / replace / discard
       │    │         ├─ compute_stochastic()   ← native pandas STOCH(5,3,3)
       │    │         ├─ evaluate_trigger()     ← cool-down state machine
       │    │         └─ send_telegram_notification()  ← on FIRE only
       │    │
       │    └─ on_close()            ← logs code + reason
       │
       └─ sleep(5s) → reconnect
```

The Monitor acts **only on closed daily candles** (`"x": true` in the Binance kline payload). Mid-candle ticks are silently discarded.

---

## Signal Logic

The Stochastic Oscillator is computed with parameters **(5, 3, 3)** using native vanilla pandas rolling windows — no external indicator library required:

```
low_min      = low.rolling(5).min()
high_max     = high.rolling(5).max()
fast_k       = 100 × (close − low_min) / (high_max − low_min)
%K (STOCHk)  = fast_k.rolling(3).mean()
%D (STOCHd)  = %K.rolling(3).mean()
```

The cool-down trigger state machine fires **at most once per oversold episode**:

| Condition | State before | Action | State after |
|---|---|---|---|
| `%K < 20` | not oversold | **FIRE** — send Telegram alert | oversold = True |
| `%K ≤ 20` | oversold | **SUPPRESS** — stay silent | oversold = True |
| `%K > 20` | oversold | **RESET** — re-arm the lock | oversold = False |
| `%K ≥ 20` | not oversold | **QUIET** — no action | oversold = False |

Engagement uses strict `<` and release uses strict `>`, so a %K value sitting exactly at 20 neither fires nor resets.

---

## Architecture

The Monitor is intentionally minimalist: **one source file**, one in-memory candle buffer, one HTTP bootstrap on startup, one persistent WebSocket session inside a reconnection loop, and one boolean cool-down lock.

| Component | Responsibility |
|---|---|
| **Bootstrapper** | One-shot HTTPS GET to Binance REST `/api/v3/klines` to seed 50 historical 1-day candles. Retries on any failure after 5 s. |
| **WebSocket_Client** | Persistent connection to `wss://stream.binance.com:9443/ws/btcusdt@kline_1d`. Parses payloads, filters non-closed candles. |
| **Indicator_Engine** | Updates the bounded candle buffer, recomputes STOCH(5,3,3) over the buffer using native pandas. |
| **Trigger_Evaluator** | Holds the `is_oversold` cool-down lock. Decides FIRE / SUPPRESS / RESET / QUIET. |
| **Notifier** | POSTs the formatted alert to the Telegram Bot API via `requests`. Logs success/failure; never crashes the Monitor. |
| **Logger** | Centralized `logging.Logger` with a `StreamHandler` (stdout) and a `FileHandler` (backup log file). UTC timestamps + event-category tags. |
| **Monitor** | Top-level orchestrator: config validation, Logger init, Bootstrapper, and the WebSocket reconnection loop. |

---

## Requirements

- **Python 3.10 or later** (tested on Python 3.14)
- A **Telegram Bot** token and chat ID ([create a bot via @BotFather](https://core.telegram.org/bots#botfather))
- A **Binance account is not required** — the Monitor uses only the public WebSocket and REST endpoints

### Runtime dependencies

| Package | Version |
|---|---|
| `pandas` | `>=2.3.3` |
| `requests` | `>=2.31.0` |
| `websocket-client` | `>=1.6.0` |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/btc-stochastic-alert.git
cd btc-stochastic-alert

# 2. (Recommended) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install runtime dependencies
pip install -r requirements.txt
```

---

## Configuration

All configuration lives at the **top of `btc_stochastic_monitor.py`** as named module-level constants. Open the file and set your credentials before running:

```python
# ── Telegram credentials (required) ──────────────────────────────────────────
TELEGRAM_BOT_TOKEN = "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ"   # from @BotFather
TELEGRAM_CHAT_ID   = "987654321"                                # your chat / channel ID

# ── Trading pair and timeframe ────────────────────────────────────────────────
SYMBOL   = "btcusdt"   # Binance symbol (lowercase for WebSocket)
INTERVAL = "1d"        # Kline interval

# ── Stochastic parameters ─────────────────────────────────────────────────────
STOCH_K_LENGTH = 5     # %K lookback window
STOCH_K_SMOOTH = 3     # %K smoothing (SMA periods)
STOCH_D_SMOOTH = 3     # %D smoothing (SMA periods)

# ── Signal threshold ──────────────────────────────────────────────────────────
OVERSOLD_THRESHOLD = 20   # %K strictly below this value triggers an alert

# ── Operational settings ──────────────────────────────────────────────────────
HISTORY_LIMIT       = 50                          # candle buffer size
RECONNECT_DELAY_S   = 5                           # seconds between reconnects
BACKUP_LOG_FILE     = "btc_stochastic_monitor.log"
```

> **Never commit your credentials to version control.** Consider using environment variable injection or a secrets manager for production deployments.

---

## Running the Monitor

```bash
# From the repository root
python btc_stochastic_monitor.py
```

The Monitor will:
1. Validate your configuration and Python version
2. Initialize the logger (stdout + `btc_stochastic_monitor.log`)
3. Fetch 50 historical daily candles from Binance REST
4. Connect to the Binance WebSocket and begin streaming
5. Send a Telegram alert when %K crosses below 20

To run in the background with output redirected:

```bash
nohup python btc_stochastic_monitor.py >> btc_stochastic_monitor.log 2>&1 &
```

---

## Smoke Check

Before launching the Monitor on a new VPS, run the pre-flight smoke check to verify that all dependencies are installed and your credentials are valid — **without opening any WebSocket connection**:

```bash
TELEGRAM_BOT_TOKEN=your_token TELEGRAM_CHAT_ID=your_chat_id \
    python scripts/smoke_check.py
```

Expected output on success:

```
[smoke_check] OK: btc_stochastic_monitor imported successfully
[smoke_check] OK: configuration is valid
[smoke_check] PASS: import + configuration checks succeeded
```

Exit code `0` = ready to run. Exit code `1` = something needs fixing (the error is printed above the FAIL line).

---

## Running as a systemd Service

For unattended 24/7 operation on a Linux VPS, create a systemd unit file:

```ini
# /etc/systemd/system/btc-monitor.service
[Unit]
Description=BTC Stochastic DCA Monitor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/home/your_user/btc-stochastic-alert
ExecStart=/home/your_user/btc-stochastic-alert/.venv/bin/python btc_stochastic_monitor.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable btc-monitor
sudo systemctl start btc-monitor
sudo systemctl status btc-monitor
```

---

## Logging

Every component logs through a single centralized logger. Each record is formatted as:

```
2026-05-29T14:32:01.123Z [CATEGORY] LEVEL message
```

Timestamps are always **UTC**. The `[CATEGORY]` tag identifies the emitting component:

| Category | Emitted by |
|---|---|
| `STARTUP` | Monitor initialization |
| `BOOTSTRAP` | Historical data fetch (success and each retry) |
| `WS` | WebSocket open, message parse errors, close |
| `INDICATOR` | Every closed candle evaluation (including %K, %D, decision) |
| `NOTIFY` | Every Telegram dispatch attempt (success or failure reason) |
| `RECONNECT` | Each reconnect attempt with a counter |
| `CONFIG` | Configuration errors (missing credentials, wrong Python version) |

Log output goes to **both stdout and the backup log file** (`btc_stochastic_monitor.log` by default). If the log file cannot be opened (e.g., permission error), the Monitor degrades to stdout-only and continues running.

### Example log output

```
2026-05-29T14:30:00.001Z [STARTUP]   INFO  Monitor startup complete; beginning bootstrap for btcusdt @ 1d.
2026-05-29T14:30:00.412Z [BOOTSTRAP] INFO  Bootstrap succeeded: seeded candle buffer with 50 historical candles.
2026-05-29T14:30:00.501Z [WS]        INFO  WebSocket session opened on wss://stream.binance.com:9443/ws/btcusdt@kline_1d
2026-05-29T23:59:59.999Z [INDICATOR] INFO  Indicator evaluation complete: close_time_ms=1748563200000 %K=18.7432 %D=22.1105 decision='alert dispatched' dispatched=True
2026-05-29T23:59:59.999Z [NOTIFY]    INFO  Telegram notification dispatched successfully (HTTP 200).
```

---

## Project Structure

```
btc-stochastic-alert/
│
├── btc_stochastic_monitor.py      # Single-file Monitor (all components)
├── requirements.txt               # Runtime dependencies
├── requirements-dev.txt           # Dev/test dependencies (not deployed)
│
├── scripts/
│   └── smoke_check.py             # Pre-flight operator tool
│
├── tests/
│   ├── conftest.py                # FakeWebSocketApp, Hypothesis profile, fixtures
│   ├── test_bootstrap_properties.py   # P2: bootstrap chronological order + retry
│   ├── test_buffer_properties.py      # P1: buffer invariants (RuleBasedStateMachine)
│   ├── test_config_validation.py      # Config guard unit tests
│   ├── test_constants_and_structure.py # AST/structural verification
│   ├── test_indicator_properties.py   # P3: STOCH columns/params; P4: NaN suppression
│   ├── test_integration_fake_ws.py    # End-to-end integration test
│   ├── test_log_records.py            # P10: required log records + fields
│   ├── test_logger_fanout.py          # P9: Logger fan-out + FileHandler degradation
│   ├── test_message_format.py         # P7: alert message formatting
│   ├── test_notifier_errors.py        # Notifier failure-mode unit tests
│   ├── test_reconnect_state.py        # P8: state preservation across reconnects
│   ├── test_trigger_state_machine.py  # P5: cool-down SM fires once per episode
│   └── test_websocket_routing.py      # P6: WS forwards iff candle is closed
```

---

## Development & Testing

### Install dev dependencies

```bash
pip install -r requirements-dev.txt
```

### Run the full test suite

```bash
pytest -q
```

Expected output: **110 tests pass** in approximately 20 seconds.

### Run a specific test file

```bash
pytest tests/test_trigger_state_machine.py -v
```

### Run only property-based tests

```bash
pytest -k "property" -v
```

### Adjust Hypothesis example count

The default profile runs 100 examples per property. To run more during a deep investigation:

```bash
pytest --hypothesis-seed=0 -v
```

---

## Test Coverage

The test suite covers **110 tests** across 13 test files, including 10 formal correctness properties validated with [Hypothesis](https://hypothesis.readthedocs.io/) property-based testing:

| Property | Test file | Validates |
|---|---|---|
| **P1** Buffer invariants under any update sequence | `test_buffer_properties.py` | REQ-4.1–4.5 |
| **P2** Bootstrap preserves rows in chronological order | `test_bootstrap_properties.py` | REQ-2.2, 2.3 |
| **P3** Stochastic computation uses configured columns and parameters | `test_indicator_properties.py` | REQ-1.5, 5.1–5.3 |
| **P4** NaN or missing columns suppress trigger evaluation | `test_indicator_properties.py` | REQ-5.4 |
| **P5** Cool-down state machine fires exactly once per oversold episode | `test_trigger_state_machine.py` | REQ-6.1–6.5 |
| **P6** WebSocket forwards iff candle is closed and preserves fields | `test_websocket_routing.py` | REQ-3.3–3.7 |
| **P7** Alert message contains required fields with two-decimal formatting | `test_message_format.py` | REQ-7.2 |
| **P8** State preservation across reconnects with stale-buffer recovery | `test_reconnect_state.py` | REQ-8.3–8.5 |
| **P9** Logger fan-out and formatting | `test_logger_fanout.py` | REQ-11.4, 11.5 |
| **P10** Required log records emitted with required fields | `test_log_records.py` | REQ-10.4–10.6 |

Additional coverage includes:
- **Config validation** — empty token, empty chat ID, Python 3.9 version mock, each asserting `SystemExit(1)` with zero network calls
- **Notifier failure modes** — HTTP 400/401/403/500, request timeout, connection error, missing credentials
- **Bootstrap retry** — all six failure modes (ConnectionError, Timeout, HTTP 500, non-JSON body, JSON object, 9-entry array), each asserting exactly one retry with a 5-second sleep
- **AST/structural verification** — all 18 required constants precede any function/class definition; all 15 required functions are present; zero `print()` calls; `pandas-ta` is absent from `requirements.txt`
- **End-to-end integration** — full startup → bootstrap → live stream → oversold episode → Telegram dispatch → reconnect cycle using a fake WebSocket and mocked HTTP

---

## Design Decisions

**Single-file script.** All components live in `btc_stochastic_monitor.py`. This simplifies VPS deployment to a single `scp` or `git pull` and lets an operator edit configuration in one place.

**Native pandas Stochastic.** The Stochastic Oscillator is computed with vanilla `pandas` rolling windows rather than an external indicator library. The original design used `pandas-ta`, but its transitive build dependency (`numba`) cannot build on Python 3.14+. The native implementation produces mathematically identical results and has no C-extension dependencies.

**Boolean cool-down lock with asymmetric thresholds.** Engagement uses strict `<` (fires when %K drops below 20) and release uses strict `>` (re-arms when %K rises above 20). A %K value sitting exactly at 20 neither fires nor resets, preventing oscillation-induced repeat alerts when %K hovers near the threshold.

**Bounded buffer of 50 candles.** STOCH(5,3,3) needs approximately 9 candles of warm-up. 50 gives ample headroom, matches the Binance REST `limit` parameter, and keeps memory usage negligible.

**`while True` reconnection loop with a fixed 5-second delay.** Trivially correct and sufficient for this workload. No exponential backoff is needed for a single-stream monitor.

**State preserved across reconnects.** The candle buffer and `is_oversold` flag live on a `MonitorState` object outside the WebSocket session lifecycle. A normal reconnect is transparent — no re-bootstrap, no missed signals.

**Stale-buffer gap detection.** If the most recent candle's open time is more than one full day old when a new session opens, the buffer is discarded and re-bootstrapped. This handles cases where the process was suspended or the VPS lost connectivity for an extended period.

**Log-and-continue philosophy.** Every external I/O boundary is wrapped. The only conditions that terminate the process are a missing/empty Telegram credential and an unsupported Python version — both checked before any network connection is opened.

---

## Telegram Alert Format

When %K crosses below 20, the Monitor sends a message in this format:

```
BTC Stochastic DCA alert: btcusdt @ 1d is oversold.
%K=18.74 %D=22.11 close=94250.5
```

- `%K` and `%D` are formatted to **two decimal places**
- `close` is the closing price of the candle that triggered the signal

---

## License

This project is released under the [MIT License](LICENSE).
