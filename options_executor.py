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
        contract = EXECUTOR_TICKERS[ticker]
        self._ib.qualifyContracts(contract)

        # Get current underlying price
        snap = self._ib.reqMktData(contract, "", True, False)
        self._ib.sleep(1.0)
        current_price = (
            snap.last
            or snap.close
            or ((snap.bid + snap.ask) / 2 if snap.bid and snap.ask else None)
        )
        self._ib.cancelMktData(contract)

        if not current_price:
            print(f"[executor] {ticker}: could not get underlying price")
            return

        opt_contract, strike, expiry = self._select_option(
            ticker, contract, direction, current_price
        )
        if opt_contract is None:
            return

        fill_price = self._place_order(ticker, opt_contract, direction, current_price)
        right = "CALL" if direction == "long" else "PUT"

        if fill_price is None:
            telegram.send_alert(
                f"*{ticker}* | {right} entry missed — unfilled after "
                f"{ORDER_MAX_RETRIES * ORDER_RETRY_SECS}s",
                channel="options",
            )
            return

        # Subscribe live streaming for position tracking
        ticker_obj   = self._ib.reqMktData(opt_contract, "", False, False)
        expiry_label = "0DTE" if expiry == now.strftime("%Y%m%d") else "1DTE"

        pos_state = {
            "contract": opt_contract,
            "contract_params": {
                "symbol":                       opt_contract.symbol,
                "lastTradeDateOrContractMonth": opt_contract.lastTradeDateOrContractMonth,
                "strike":                       opt_contract.strike,
                "right":                        opt_contract.right,
                "exchange":                     opt_contract.exchange,
                "currency":                     opt_contract.currency,
                "multiplier":                   opt_contract.multiplier,
            },
            "ticker_obj":      ticker_obj,
            "symbol":          opt_contract.localSymbol,
            "direction":       direction,
            "entry_price":     fill_price,
            "high_water_mark": fill_price,
            "trailing_active": False,
            "fill_time":       now.isoformat(),
            "contracts":       1,
        }

        with self._lock:
            self._positions[ticker] = pos_state
        self._save_positions()

        telegram.send_alert(
            f"*{ticker}* | {right} ATM | Strike: {strike:.0f} | {expiry_label} | "
            f"Fill: ${fill_price:.2f} | "
            f"Signal: {'ORB/MR Long' if direction == 'long' else 'ORB/MR Short'}",
            channel="options",
        )

    def _select_option(self, ticker: str, contract, direction: str, current_price: float):
        """Returns (Option, strike, expiry_str) or (None, None, None)."""
        try:
            cds = self._ib.reqContractDetails(contract)
            if not cds:
                print(f"[executor] {ticker}: no contract details from IB")
                return None, None, None
            con = cds[0].contract

            chains = self._ib.reqSecDefOptParams(
                con.symbol, con.exchange, con.secType, con.conId
            )
            if not chains:
                print(f"[executor] {ticker}: no option chain from IB")
                return None, None, None
            chain = chains[0]

            now    = dt.datetime.now()
            expiry = _select_expiry(list(chain.expirations), dt.date.today(), now.time())
            if expiry is None:
                print(f"[executor] {ticker}: no suitable expiry (need 0DTE or 1DTE)")
                return None, None, None

            strike = _find_atm_strike(list(chain.strikes), current_price)
            right  = "C" if direction == "long" else "P"

            opt = Option(
                symbol=con.symbol,
                lastTradeDateOrContractMonth=expiry,
                strike=strike,
                right=right,
                exchange="SMART",
                currency="USD",
                multiplier="100",
            )
            self._ib.qualifyContracts(opt)
            return opt, strike, expiry
        except Exception as e:
            print(f"[executor] _select_option error ({ticker}): {e}")
            return None, None, None

    def _place_order(self, ticker: str, opt_contract, direction: str, current_price: float):
        """
        Place limit order at bid/ask midpoint. Widen by 1 cent per retry.
        Returns fill price (float) or None if all retries exhausted.
        """
        for attempt in range(ORDER_MAX_RETRIES):
            snap = self._ib.reqMktData(opt_contract, "", True, False)
            self._ib.sleep(1.0)
            bid = snap.bid
            ask = snap.ask
            self._ib.cancelMktData(opt_contract)

            if not bid or not ask or bid <= 0 or ask <= 0:
                print(f"[executor] {ticker}: no bid/ask on attempt {attempt + 1}")
                time.sleep(ORDER_RETRY_SECS)
                continue

            limit_price = round((bid + ask) / 2 + attempt * 0.01, 2)
            order = LimitOrder("BUY", 1, limit_price)
            trade = self._ib.placeOrder(opt_contract, order)
            print(f"[executor] {ticker}: limit order ${limit_price:.2f} (attempt {attempt + 1})")

            deadline = time.time() + ORDER_RETRY_SECS
            while time.time() < deadline:
                self._ib.sleep(1)
                if trade.orderStatus.status == "Filled":
                    fill = trade.orderStatus.avgFillPrice
                    print(f"[executor] {ticker}: filled at ${fill:.2f}")
                    return fill

            self._ib.cancelOrder(order)
            print(f"[executor] {ticker}: unfilled on attempt {attempt + 1}, retrying")

        return None

    def _tracking_loop(self):
        raise NotImplementedError

    def _exit_position(self, ticker: str, reason: str) -> None:
        raise NotImplementedError

    def _reconnect(self):
        raise NotImplementedError
