#!/usr/bin/env python3
"""Operator smoke check for the BTC Stochastic DCA Monitor.

Purpose
-------
A lightweight pre-flight tool an operator runs on a freshly provisioned VPS to
confirm two things *before* launching the long-lived Monitor:

1. **Importability** - ``btc_stochastic_monitor`` loads without raising. Because
   importing the module pulls in its third-party dependencies (``pandas``,
   ``requests``, and ``websocket-client`` via ``from websocket import
   WebSocketApp``), a successful import doubles as a check that the runtime
   environment has all dependencies installed (REQ-10.3).
2. **Configuration validity** - the operator-supplied Telegram credentials and
   the Python interpreter version satisfy ``validate_config`` (REQ-1.9,
   REQ-10.1, REQ-10.2).

This script does **not** open any network connection and does **not** start the
WebSocket loop. It only verifies importability and configuration validity, then
exits ``0`` on success or ``1`` on failure.

Operator command
-----------------
Run from the repository root, supplying the credentials as environment
variables::

    TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... python scripts/smoke_check.py

On success the script prints a confirmation line and exits ``0``. On failure
(missing credentials, an unsupported Python version, or a failed import) it
surfaces the reason and exits ``1``.

Note
----
This is an operator tool, not part of the Monitor module itself. Using
``print`` here is intentional and acceptable - the Monitor's "no print" rule
(REQ-11.6) applies to ``btc_stochastic_monitor.py``, not to this helper script.
"""

from __future__ import annotations

import os
import sys

# Ensure the repository root (the parent of this ``scripts/`` directory) is on
# ``sys.path`` so ``import btc_stochastic_monitor`` resolves when this script is
# launched as ``python scripts/smoke_check.py`` from the repository root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    """Run the smoke check and return a process exit code.

    Returns:
        ``0`` when the module imports and ``validate_config`` succeeds; ``1``
        when the import fails or ``validate_config`` terminates the process.
    """
    # --- Step 1: module-load check (REQ-10.3) --------------------------------
    # Importing the module exercises every third-party dependency. An
    # ImportError here means the runtime environment is missing a dependency
    # (or the module itself is broken), so report it and fail.
    try:
        import btc_stochastic_monitor
    except Exception as exc:  # noqa: BLE001 - surface any import-time failure
        print(
            f"[smoke_check] FAIL: could not import btc_stochastic_monitor: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1

    print("[smoke_check] OK: btc_stochastic_monitor imported successfully")

    # --- Step 2: map operator config onto the module constants ---------------
    # validate_config reads the module-level TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
    # globals, so copy the operator-supplied environment values onto them before
    # the call (REQ-1.9, REQ-10.2).
    btc_stochastic_monitor.TELEGRAM_BOT_TOKEN = os.environ.get(
        "TELEGRAM_BOT_TOKEN", ""
    )
    btc_stochastic_monitor.TELEGRAM_CHAT_ID = os.environ.get(
        "TELEGRAM_CHAT_ID", ""
    )

    # --- Step 3: configuration validity check (REQ-1.9, REQ-10.1, REQ-10.2) --
    # validate_config returns None on success and calls sys.exit(1) (raising
    # SystemExit) on failure, after logging a CONFIG-category error explaining
    # what is wrong. Catch SystemExit so we can translate it into our own exit
    # code rather than letting it propagate as an uncaught exception.
    try:
        btc_stochastic_monitor.validate_config()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        print(
            f"[smoke_check] FAIL: configuration is invalid (validate_config "
            f"exited with status {code}); see the CONFIG error above"
        )
        return code or 1

    print("[smoke_check] OK: configuration is valid")
    print("[smoke_check] PASS: import + configuration checks succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
