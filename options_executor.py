from ib_insync import IB, Stock, Option, LimitOrder, MarketOrder, util
import threading
import logging
import json
import csv
import datetime as dt
import os
import time

import yfinance as yf
import telegram

log = logging.getLogger(__name__)

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
ORDER_RETRY_SECS  = 5                 # seconds per fill attempt (re-fetches fresh mid each retry)
ORDER_MAX_RETRIES = 6                 # 6 × 5s = 30s of limit attempts before market fallback
POSITIONS_FILE    = "option_positions.json"
TRADES_FILE       = "option_trades.csv"


# ── PURE HELPERS (fully testable without IB) ──────────────────────────────────

def _find_atm_strike(strikes: list, current_price: float) -> float:
    """Return the strike closest to current_price."""
    return min(strikes, key=lambda s: abs(s - current_price))


def _select_expiry(expirations: list, today: dt.date, current_time: dt.time) -> str | None:
    """
    Prefer 0DTE if available and time <= OTM_CUTOFF (13:30).
    Otherwise pick the nearest available expiry within 4 calendar days
    (covers weekly-expiry tickers like TSLA where Friday is 2-3 days out).
    Returns None if nothing found within that window.
    """
    exps = set(expirations)

    # Try 0DTE first if early enough
    today_str = today.strftime("%Y%m%d")
    if today_str in exps and current_time <= OTM_CUTOFF:
        return today_str

    # Walk forward up to 4 days to find nearest available expiry
    for days_out in range(1, 5):
        candidate = (today + dt.timedelta(days=days_out)).strftime("%Y%m%d")
        if candidate in exps:
            return candidate

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

    def __init__(self, positions_file: str = POSITIONS_FILE):
        self._ib             = IB()
        self._positions      = {}
        self._lock           = threading.Lock()
        self._running        = False
        self._thread         = None
        self._positions_file = positions_file

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
            with open(self._positions_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log.error(f"[executor] save error: {e}")

    def _load_positions(self):
        if not os.path.exists(self._positions_file):
            return
        try:
            with open(self._positions_file) as f:
                data = json.load(f)
        except Exception:
            return

        today = dt.date.today()
        for ticker, state in data.items():
            fill_dt = dt.datetime.fromisoformat(state["fill_time"])
            if fill_dt.date() < today:
                log.info(f"[executor] Skipping expired position for {ticker}")
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
                log.info(f"[executor] Recovered open position: {ticker}")
            except Exception as e:
                log.error(f"[executor] Could not recover {ticker}: {e}")

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
            log.error(f"[executor] trade log error: {e}")

    # ── Stubs — implemented in Tasks 4-6 ─────────────────────────────────────

    def on_signal(self, ticker: str, direction: str) -> None:
        """Called from alert engine. direction: 'long' | 'short'. Never raises."""
        try:
            self._handle_signal(ticker, direction)
        except Exception as e:
            log.error(f"[executor] on_signal error ({ticker} {direction}): {e}", exc_info=True)

    def _handle_signal(self, ticker: str, direction: str) -> None:
        if ticker not in EXECUTOR_TICKERS:
            return

        now = dt.datetime.now()
        if now.time() >= ENTRY_CUTOFF:
            log.info(f"[executor] {ticker}: past entry cutoff ({now.strftime('%H:%M')}), skipping")
            return

        with self._lock:
            pos = self._positions.get(ticker)

        if pos is not None:
            if pos["direction"] == direction:
                log.info(f"[executor] {ticker}: already {direction}, skipping")
                return
            self._exit_position(ticker, "Signal Reversal")

        self._open_position(ticker, direction, now)

    def _open_position(self, ticker: str, direction: str, now: dt.datetime) -> None:
        contract = EXECUTOR_TICKERS[ticker]
        self._ib.qualifyContracts(contract)

        # Get current underlying price from IB (blocking snapshot)
        current_price = None
        try:
            [snap] = self._ib.reqTickers(contract)
            p = snap.marketPrice()
            if p and p == p and p > 0:  # not None, not NaN, positive
                current_price = p
        except Exception as e:
            log.warning(f"[executor] {ticker}: IB price failed ({e}), falling back to yfinance")

        if not current_price:
            try:
                current_price = yf.Ticker(ticker).fast_info.last_price
            except Exception as e:
                log.warning(f"[executor] {ticker}: yfinance price also failed ({e})")

        if not current_price or current_price <= 0:
            log.warning(f"[executor] {ticker}: could not get underlying price")
            return
        log.info(f"[executor] {ticker}: underlying price ${current_price:.2f}")

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
        days_to_exp  = (dt.datetime.strptime(expiry, "%Y%m%d").date() - dt.date.today()).days
        expiry_label = f"{days_to_exp}DTE" if days_to_exp > 0 else "0DTE"

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
                log.warning(f"[executor] {ticker}: no contract details from IB")
                return None, None, None
            con = cds[0].contract

            chains = self._ib.reqSecDefOptParams(
                con.symbol, "", con.secType, con.conId
            )
            if not chains:
                log.warning(f"[executor] {ticker}: no option chain from IB")
                return None, None, None

            # Among SMART chains pick the fullest one; fall back to whatever has most expirations
            smart = [c for c in chains if c.exchange == "SMART"]
            chain = (max(smart, key=lambda c: len(c.expirations)) if smart
                     else max(chains, key=lambda c: len(c.expirations)))
            log.info(f"[executor] {ticker}: using chain exchange={chain.exchange}, "
                     f"{len(chain.expirations)} expirations, {len(chain.strikes)} strikes")

            now    = dt.datetime.now()
            expiry = _select_expiry(list(chain.expirations), dt.date.today(), now.time())
            if expiry is None:
                log.warning(f"[executor] {ticker}: no suitable expiry within 4 days")
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
            log.error(f"[executor] _select_option error ({ticker}): {e}", exc_info=True)
            return None, None, None

    def _place_order(self, ticker: str, opt_contract, direction: str, current_price: float):
        """
        Attempts 1-3: limit at mid. Attempts 4-6: limit at ask (walk-up).
        Falls back to market only if no bid/ask for all 6 attempts.
        Returns fill price (float) or None on failure.
        """
        action = "BUY"

        for attempt in range(ORDER_MAX_RETRIES):
            try:
                [snap] = self._ib.reqTickers(opt_contract)
                bid, ask = snap.bid, snap.ask
            except Exception:
                bid, ask = None, None

            if not bid or not ask or bid <= 0 or ask <= 0 or bid != bid or ask != ask:
                log.warning(f"[executor] {ticker}: no bid/ask on attempt {attempt + 1}")
                self._ib.sleep(ORDER_RETRY_SECS)
                continue

            # First half: mid price. Second half: walk up to ask.
            limit_price = round((bid + ask) / 2 if attempt < ORDER_MAX_RETRIES // 2 else ask, 2)
            order = LimitOrder(action, 1, limit_price)
            order.tif = "DAY"
            trade = self._ib.placeOrder(opt_contract, order)
            mode  = "mid" if attempt < ORDER_MAX_RETRIES // 2 else "ask"
            log.info(f"[executor] {ticker}: limit@{mode} ${limit_price:.2f}  bid=${bid:.2f} ask=${ask:.2f}  (attempt {attempt + 1})")

            deadline = time.time() + ORDER_RETRY_SECS
            while time.time() < deadline:
                self._ib.sleep(1)   # yield to event loop so fill notifications arrive
                if trade.orderStatus.status == "Filled":
                    fill = trade.orderStatus.avgFillPrice
                    log.info(f"[executor] {ticker}: filled at ${fill:.2f}")
                    return fill

            self._ib.cancelOrder(order)
            self._ib.sleep(1)
            log.warning(f"[executor] {ticker}: unfilled on attempt {attempt + 1}, retrying")

        # ── Last resort: market order (only if bid/ask unavailable throughout) ─
        log.warning(f"[executor] {ticker}: all limit attempts failed — last resort market order")
        try:
            order = MarketOrder(action, 1)
            order.tif = "DAY"
            trade = self._ib.placeOrder(opt_contract, order)
            log.info(f"[executor] {ticker}: market order submitted")
            deadline = time.time() + ORDER_RETRY_SECS
            while time.time() < deadline:
                self._ib.sleep(1)
                if trade.orderStatus.status == "Filled":
                    fill = trade.orderStatus.avgFillPrice
                    log.info(f"[executor] {ticker}: market fill at ${fill:.2f}")
                    return fill
        except Exception as e:
            log.error(f"[executor] {ticker}: market order failed: {e}")

        return None

    def _tracking_loop(self):
        while self._running:
            time.sleep(TRACK_INTERVAL)

            if not self._ib.isConnected():
                log.warning("[executor] IB disconnected — attempting reconnect")
                self._reconnect()
                continue

            now = dt.datetime.now()
            with self._lock:
                tickers = list(self._positions.keys())

            for ticker in tickers:
                with self._lock:
                    pos = self._positions.get(ticker)
                if pos is None:
                    continue

                # Read live streaming price (no new IB request needed)
                ticker_obj = pos["ticker_obj"]
                mid = ticker_obj.midpoint()
                if not mid or mid <= 0:
                    if ticker_obj.bid and ticker_obj.ask and ticker_obj.bid > 0:
                        mid = (ticker_obj.bid + ticker_obj.ask) / 2
                    else:
                        continue   # no price yet this cycle

                entry = pos["entry_price"]
                hwm   = pos["high_water_mark"]
                trail = pos["trailing_active"]

                # Update high-water mark
                if mid > hwm:
                    with self._lock:
                        if self._positions.get(ticker):
                            self._positions[ticker]["high_water_mark"] = mid
                    hwm = mid

                # Arm trailing stop once up 50%
                if not trail and mid >= entry * TRAIL_ACTIVATE:
                    with self._lock:
                        if self._positions.get(ticker):
                            self._positions[ticker]["trailing_active"] = True
                    trail = True
                    self._save_positions()
                    log.info(f"[executor] {ticker}: trailing stop armed at ${mid:.2f}")

                # EOD forced exit (highest priority — checked before rule exits)
                if now.time() >= EOD_CUTOFF:
                    self._exit_position(ticker, "EOD")
                    continue

                # Rule-based exits: stop loss then trailing
                should_exit, reason = _check_exit(entry, mid, hwm, trail)
                if should_exit:
                    self._exit_position(ticker, reason)

    def _exit_position(self, ticker: str, reason: str) -> None:
        with self._lock:
            pos = self._positions.get(ticker)
        if pos is None:
            return

        exit_price = pos["entry_price"]   # fallback if market order fails
        try:
            self._ib.cancelMktData(pos["contract"])
            self._ib.qualifyContracts(pos["contract"])
            order = MarketOrder("SELL", pos["contracts"])
            order.tif = "DAY"
            trade = self._ib.placeOrder(pos["contract"], order)

            deadline = time.time() + 30
            while time.time() < deadline:
                time.sleep(1)   # event loop updates trade status in background
                if trade.orderStatus.status == "Filled":
                    exit_price = trade.orderStatus.avgFillPrice
                    break
        except Exception as e:
            log.error(f"[executor] exit order error ({ticker}): {e}", exc_info=True)

        pnl_pct   = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
        fill_dt   = dt.datetime.fromisoformat(pos["fill_time"])
        hold_mins = int((dt.datetime.now() - fill_dt).total_seconds() / 60)
        right     = "CALL" if pos["direction"] == "long" else "PUT"

        telegram.send_alert(
            f"*{ticker}* | {right} EXIT | Reason: {reason}\n"
            f"Entry: ${pos['entry_price']:.2f} → Exit: ${exit_price:.2f} | "
            f"P&L: {pnl_pct:+.1f}% | Hold: {hold_mins}m",
            channel="options",
        )

        self._append_trade(ticker, pos, exit_price, reason)

        with self._lock:
            self._positions.pop(ticker, None)
        self._save_positions()

    def _reconnect(self):
        attempts = 0
        while self._running and not self._ib.isConnected():
            attempts += 1
            try:
                self._ib.connect("127.0.0.1", 4002, clientId=111, timeout=10)
                log.info(f"[executor] Reconnected to IB (attempt {attempts})")
                # Re-subscribe streaming data for all open positions
                with self._lock:
                    for ticker, pos in self._positions.items():
                        if pos.get("contract"):
                            ticker_obj = self._ib.reqMktData(pos["contract"], "", False, False)
                            self._positions[ticker]["ticker_obj"] = ticker_obj
                return
            except Exception as e:
                log.warning(f"[executor] Reconnect attempt {attempts} failed: {e}")
                time.sleep(30)
