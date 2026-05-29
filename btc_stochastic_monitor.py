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
