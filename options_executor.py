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


class OptionsExecutor:

    def __init__(self):
        self._ib      = IB()
        self._positions = {}    # ticker -> position state dict
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 111):
        self._ib.connect(host, port, clientId=client_id, timeout=10)
        self._load_positions()
        self._running = True
        self._thread  = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()

    def disconnect(self):
        self._running = False
        if self._ib.isConnected():
            self._ib.disconnect()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_positions(self):
        data = {}
        with self._lock:
            for ticker, pos in self._positions.items():
                data[ticker] = {
                    k: v for k, v in pos.items()
                    if k not in ("contract", "ticker_obj")
                }
        try:
            with open(POSITIONS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[executor] save error: {e}")

    def _load_positions(self):
        if not os.path.exists(POSITIONS_FILE):
            return
        try:
            with open(POSITIONS_FILE) as f:
                data = json.load(f)
        except Exception:
            return

        today = dt.date.today()
        for ticker, state in data.items():
            fill_dt = dt.datetime.fromisoformat(state["fill_time"])
            if fill_dt.date() < today:
                print(f"[executor] Skipping expired position for {ticker}")
                continue
            try:
                contract = Option(**state["contract_params"])
                self._ib.qualifyContracts(contract)
                ticker_obj = self._ib.reqMktData(contract, "", False, False)
                self._positions[ticker] = {
                    **state,
                    "contract":   contract,
                    "ticker_obj": ticker_obj,
                }
                print(f"[executor] Recovered open position: {ticker}")
            except Exception as e:
                print(f"[executor] Could not recover {ticker}: {e}")

    def _append_trade(self, ticker: str, pos: dict, exit_price: float, reason: str):
        file_exists = os.path.exists(TRADES_FILE)
        fill_dt     = dt.datetime.fromisoformat(pos["fill_time"])
        hold_mins   = int((dt.datetime.now() - fill_dt).total_seconds() / 60)
        pnl_pct     = round((exit_price - pos["entry_price"]) / pos["entry_price"] * 100, 2)
        fieldnames  = [
            "ticker", "direction", "option_symbol", "entry_price",
            "exit_price", "pnl_pct", "hold_minutes", "exit_reason",
            "entry_time", "exit_time",
        ]
        try:
            with open(TRADES_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "ticker":        ticker,
                    "direction":     pos["direction"],
                    "option_symbol": pos["symbol"],
                    "entry_price":   pos["entry_price"],
                    "exit_price":    exit_price,
                    "pnl_pct":       pnl_pct,
                    "hold_minutes":  hold_mins,
                    "exit_reason":   reason,
                    "entry_time":    pos["fill_time"],
                    "exit_time":     dt.datetime.now().isoformat(),
                })
        except Exception as e:
            print(f"[executor] trade log error: {e}")

    # ── Stubs — implemented in Tasks 4-6 ─────────────────────────────────────

    def on_signal(self, ticker: str, direction: str) -> None:
        """Called from alert engine. direction: 'long' | 'short'. Never raises."""
        try:
            self._handle_signal(ticker, direction)
        except Exception as e:
            print(f"[executor] on_signal error ({ticker} {direction}): {e}")

    def _handle_signal(self, ticker: str, direction: str) -> None:
        if ticker not in EXECUTOR_TICKERS:
            return

        now = dt.datetime.now()
        if now.time() >= ENTRY_CUTOFF:
            print(f"[executor] {ticker}: past entry cutoff ({now.strftime('%H:%M')}), skipping")
            return

        with self._lock:
            pos = self._positions.get(ticker)

        if pos is not None:
            if pos["direction"] == direction:
                print(f"[executor] {ticker}: already {direction}, skipping")
                return
            self._exit_position(ticker, "Signal Reversal")

        self._open_position(ticker, direction, now)

    def _open_position(self, ticker: str, direction: str, now: dt.datetime) -> None:
        raise NotImplementedError

    def _select_option(self, ticker: str, contract, direction: str, current_price: float):
        raise NotImplementedError

    def _place_order(self, ticker: str, opt_contract, direction: str, current_price: float):
        raise NotImplementedError

    def _tracking_loop(self):
        raise NotImplementedError

    def _exit_position(self, ticker: str, reason: str) -> None:
        raise NotImplementedError

    def _reconnect(self):
        raise NotImplementedError
