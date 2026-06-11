from ib_insync import IB, Stock, Option, LimitOrder, MarketOrder, util
import threading
import json
import csv
import datetime as dt
import os
import time

import telegram

# ── Start ib_insync event loop once at module level ───────────────────────────
util.startLoop()

# ── CONFIG ────────────────────────────────────────────────────────────────────

EXECUTOR_TICKERS = {
    "SPY":  Stock("SPY",  "SMART", "USD"),
    "TSLA": Stock("TSLA", "SMART", "USD"),
}

ENTRY_CUTOFF      = dt.time(14,  0)   # no new entries after 14:00
EOD_CUTOFF        = dt.time(14, 55)   # hard exit all positions
OTM_CUTOFF        = dt.time(13, 30)   # after this, use 1DTE instead of 0DTE
STOP_LOSS_PCT     = 0.50              # exit if premium drops to 50% of entry
TRAIL_ACTIVATE    = 1.50              # arm trailing once premium reaches 150% of entry
TRAIL_FROM_HWM    = 0.75              # trail at 75% of high-water mark
TRACK_INTERVAL    = 5                 # seconds between position checks
ORDER_RETRY_SECS  = 30                # seconds to wait per fill attempt
ORDER_MAX_RETRIES = 3                 # give up after this many retries
POSITIONS_FILE    = "option_positions.json"
TRADES_FILE       = "option_trades.csv"


# ── PURE HELPERS (fully testable without IB) ──────────────────────────────────

def _find_atm_strike(strikes: list, current_price: float) -> float:
    """Return the strike closest to current_price."""
    return min(strikes, key=lambda s: abs(s - current_price))


def _select_expiry(expirations: list, today: dt.date, current_time: dt.time) -> str | None:
    """
    Return 0DTE if available AND current_time <= OTM_CUTOFF (13:30).
    Otherwise return 1DTE. Returns None if neither is in expirations.
    """
    today_str    = today.strftime("%Y%m%d")
    tomorrow_str = (today + dt.timedelta(days=1)).strftime("%Y%m%d")
    exps         = set(expirations)

    if today_str in exps and current_time <= OTM_CUTOFF:
        return today_str
    if tomorrow_str in exps:
        return tomorrow_str
    return None


def _check_exit(
    entry_price: float,
    current_price: float,
    high_water_mark: float,
    trailing_active: bool,
) -> tuple:
    """
    Returns (should_exit: bool, reason: str | None).
    EOD check is NOT here — the caller handles it (needs the real clock).
    Priority: stop loss first, then trailing.
    """
    if current_price <= entry_price * STOP_LOSS_PCT:
        return True, "Stop Loss (50%)"
    if trailing_active and current_price <= high_water_mark * TRAIL_FROM_HWM:
        return True, "Trailing Stop"
    return False, None
