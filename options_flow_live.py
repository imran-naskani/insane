"""
options_flow_live.py

Live TSLA options flow monitor via IB Gateway.

- 25 strikes above + 25 strikes below open price = 50 strikes
- Both calls and puts = 100 market data subscriptions (exactly at IBKR limit)
- Fixed subscription — no re-centering during session
- Alerts to Telegram on qualifying prints (>= 50 contracts, >= $75K notional)
- Saves every qualifying print to options_flow_data/flow_TSLA_{date}.json

Run: python options_flow_live.py
"""

import json, os, time, datetime as dt, threading
from pathlib import Path

import pandas as pd
from ib_insync import IB, Option, Stock, util
from dotenv import load_dotenv
import requests

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
IB_HOST     = "127.0.0.1"
IB_PORT     = 4002
CLIENT_ID   = 3           # distinct from executor (1) and alert engine (unused)

TICKER      = "TSLA"
EXCHANGE    = "SMART"
CURRENCY    = "USD"
STRIKE_STEP = 2.5
N_STRIKES   = 25          # each side → 25 up + 25 down = 50 strikes

# Alert thresholds (per single fill)
MIN_CONTRACTS = 50        # ignore fills < 50 contracts
MIN_NOTIONAL  = 25_000    # ignore fills < $25K
T2_NOTIONAL   = 75_000    # Telegram alert
T3_NOTIONAL   = 200_000   # Strong Telegram alert

# Session (ET)
SESSION_OPEN  = dt.time(9, 30)
SESSION_CLOSE = dt.time(16, 0)

# Telegram
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

DATA_DIR  = Path("options_flow_data")
DATA_DIR.mkdir(exist_ok=True)

# ── STORAGE ───────────────────────────────────────────────────────────────────
_file_lock = threading.Lock()

def flow_file():
    return DATA_DIR / f"flow_{TICKER}_{dt.date.today()}.json"

def save_print(record):
    fp = flow_file()
    with _file_lock:
        data = []
        if fp.exists():
            try:
                with open(fp) as f:
                    data = json.load(f)
            except Exception:
                data = []
        data.append(record)
        with open(fp, "w") as f:
            json.dump(data, f, indent=2)

# ── TELEGRAM ──────────────────────────────────────────────────────────────────
def send_telegram(msg):
    if not BOT_TOKEN or not CHAT_ID:
        print(f"  [telegram] {msg}")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=5,
        )
    except Exception as e:
        print(f"  [telegram error] {e}")

# ── IB HELPERS ────────────────────────────────────────────────────────────────
def get_open_price(ib):
    """Get TSLA's current mid as ATM reference (call at/just after open)."""
    stock = Stock(TICKER, EXCHANGE, CURRENCY)
    ib.qualifyContracts(stock)
    t = ib.reqMktData(stock, "", False, False)
    ib.sleep(2)
    price = None
    if t.last and t.last > 0:
        price = t.last
    elif t.bid and t.ask and t.bid > 0 and t.ask > 0:
        price = (t.bid + t.ask) / 2
    elif t.close and t.close > 0:
        price = t.close
    ib.cancelMktData(stock)
    return price

def _tsla_chain(chains):
    """Pick the standard TSLA chain (not 2TSLA mini) from reqSecDefOptParams results."""
    # Prefer chain whose tradingClass matches TICKER exactly
    for c in chains:
        if getattr(c, "tradingClass", "") == TICKER:
            return c
    return chains[0]  # fallback

def get_expiries(ib):
    """Return (0dte_expiry, next_expiry) as YYYYMMDD strings."""
    stock = Stock(TICKER, EXCHANGE, CURRENCY)
    ib.qualifyContracts(stock)
    chains = ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
    if not chains:
        raise RuntimeError("No option chains returned from IB")

    chain = _tsla_chain(chains)
    today_str = dt.date.today().strftime("%Y%m%d")

    # Filter to expiries >= today, sorted
    upcoming = sorted(e for e in chain.expirations if e >= today_str)
    if not upcoming:
        raise RuntimeError("No upcoming expiries found")

    expiry_0dte = upcoming[0]
    expiry_next = upcoming[1] if len(upcoming) > 1 else upcoming[0]
    return expiry_0dte, expiry_next

def nearest_strike(price, strikes_available):
    """Snap ATM reference to nearest available strike."""
    return min(strikes_available, key=lambda s: abs(s - price))

def build_contracts(ib, atm_strike, strikes_avail, expiry_0dte, expiry_next):
    """Build 100 option contracts: 50 strikes × C+P for 0DTE only.
    Use next expiry for informational context but keep 0DTE as primary.
    100 lines = 50 strikes × 2 rights (within IBKR limit).
    """
    # 25 above + 25 below + ATM itself = 51 strikes, trim to 50 each side
    above = sorted([s for s in strikes_avail if s > atm_strike])[:N_STRIKES]
    below = sorted([s for s in strikes_avail if s < atm_strike], reverse=True)[:N_STRIKES]
    strikes = sorted(below + [atm_strike] + above)
    # Keep exactly 50 strikes (25 each side, exclude ATM to stay at 50)
    strikes = sorted(below + above)  # 25 + 25 = 50 exactly

    contracts = []
    for strike in strikes:
        for right in ["C", "P"]:
            # tradingClass='TSLA' avoids ambiguity with '2TSLA' mini options
            c = Option(TICKER, expiry_0dte, strike, right, EXCHANGE,
                       currency=CURRENCY, tradingClass=TICKER)
            contracts.append(c)

    ib.qualifyContracts(*contracts)
    valid = [c for c in contracts if c.conId]
    print(f"  Qualified {len(valid)} contracts  (strikes: {strikes[0]}-{strikes[-1]})")
    return valid, strikes

# ── FLOW PROCESSOR ────────────────────────────────────────────────────────────
_prev_last_size  = {}   # conId -> last known lastSize value
_bid_ask_cache   = {}   # conId -> (bid, ask)

def classify_side(price, bid, ask):
    if not bid or not ask or bid <= 0 or ask <= 0:
        return "UNKNOWN"
    mid = (bid + ask) / 2
    if price >= ask:                         return "ASK+"
    elif price <= bid:                       return "BID-"
    elif price > mid + (ask - bid) * 0.1:   return "ABOVE MID"
    elif price < mid - (ask - bid) * 0.1:   return "BELOW MID"
    else:                                    return "MID"

def on_pending_tickers(tickers):
    now_et = dt.datetime.now(dt.timezone.utc).astimezone(
        dt.timezone(dt.timedelta(hours=-4))  # ET (EDT)
    ).time()

    if now_et < SESSION_OPEN or now_et >= SESSION_CLOSE:
        return

    for t in tickers:
        c = t.contract
        if not hasattr(c, "right"):
            continue   # skip underlying stock ticker if present

        con_id = c.conId

        # Update bid/ask cache
        if t.bid and t.bid > 0 and t.ask and t.ask > 0:
            _bid_ask_cache[con_id] = (t.bid, t.ask)

        # Detect new print via lastSize change
        if not t.last or t.last <= 0:
            continue
        if not t.lastSize or t.lastSize <= 0:
            continue

        prev = _prev_last_size.get(con_id, 0)
        if t.lastSize == prev:
            continue   # no new print

        _prev_last_size[con_id] = t.lastSize

        size     = int(t.lastSize)
        price    = float(t.last)
        notional = round(size * price * 100)

        if size < MIN_CONTRACTS:    continue
        if notional < MIN_NOTIONAL: continue

        bid, ask = _bid_ask_cache.get(con_id, (None, None))
        side     = classify_side(price, bid, ask)

        record = {
            "timestamp":  dt.datetime.now().isoformat(),
            "ticker":     TICKER,
            "expiry":     c.lastTradeDateOrContractMonth,
            "strike":     float(c.strike),
            "right":      c.right,
            "size":       size,
            "price":      price,
            "bid":        round(bid, 2) if bid else None,
            "ask":        round(ask, 2) if ask else None,
            "notional":   notional,
            "side":       side,
        }

        # Save every qualifying print
        save_print(record)

        # Telegram alert for T2+
        if notional >= T2_NOTIONAL:
            tier  = "STRONG ALERT" if notional >= T3_NOTIONAL else "ALERT"
            ba_str = f" | bid ${bid:.2f} ask ${ask:.2f}" if bid else ""
            msg = (
                f"[{tier}] TSLA {c.strike}{c.right} {c.lastTradeDateOrContractMonth}\n"
                f"{size} contracts @ ${price:.2f} {side}\n"
                f"Notional: ${notional:,}{ba_str}"
            )
            print(f"\n  >> {tier}: {TICKER} {c.strike}{c.right} | {size} @ ${price:.2f} "
                  f"{side} | ${notional:,}")
            send_telegram(msg)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  TSLA Options Flow Monitor — {dt.date.today()}")
    print(f"  {N_STRIKES} strikes each side | 100 IBKR lines")
    print(f"  Alert threshold: >= {MIN_CONTRACTS} contracts AND >= ${T2_NOTIONAL:,}")
    print(f"{'='*60}\n")

    # Connect
    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
    print(f"  Connected to IB Gateway  (clientId={CLIENT_ID})")

    # Get open price
    print("  Fetching TSLA open price...")
    open_price = get_open_price(ib)
    if not open_price:
        raise RuntimeError("Could not get TSLA price from IB")
    print(f"  TSLA reference price: ${open_price:.2f}")

    # Get expiries
    print("  Fetching option chain expiries...")
    expiry_0dte, expiry_next = get_expiries(ib)
    print(f"  0DTE: {expiry_0dte}  |  Next: {expiry_next}")

    # Get available strikes from chain
    stock = Stock(TICKER, EXCHANGE, CURRENCY)
    ib.qualifyContracts(stock)
    chains = ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
    chain = _tsla_chain(chains) if chains else None
    print(f"  Chain tradingClass: {getattr(chain, 'tradingClass', 'N/A')}  total chains: {len(chains)}")
    strikes_avail = sorted(chain.strikes) if chain else []
    atm_strike    = nearest_strike(open_price, strikes_avail)
    print(f"  ATM strike: ${atm_strike}")

    # Build and qualify contracts
    print(f"  Building contracts ({N_STRIKES}×2 sides × C+P = 100 subscriptions)...")
    contracts, strikes = build_contracts(
        ib, atm_strike, strikes_avail, expiry_0dte, expiry_next
    )

    # Subscribe to market data for all 100 contracts
    print(f"  Subscribing to {len(contracts)} contracts...")
    tickers = []
    for c in contracts:
        t = ib.reqMktData(c, "", False, False)
        tickers.append(t)

    print(f"  Subscribed. Listening for prints >= {MIN_CONTRACTS} contracts...\n")
    print(f"  {'TIME':>8}  {'CONTRACT':<14} {'SIZE':>5}  {'PRICE':>6}  {'NOTIONAL':>10}  SIDE")
    print(f"  {'-'*60}")

    # Wire up event handler
    ib.pendingTickersEvent += on_pending_tickers

    # Session loop
    try:
        while True:
            ib.sleep(1)
            now_et = dt.datetime.now(dt.timezone.utc).astimezone(
                dt.timezone(dt.timedelta(hours=-4))
            )
            if now_et.time() >= SESSION_CLOSE:
                print(f"\n  Session closed at {now_et.strftime('%H:%M')} ET — shutting down.")
                break
    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
    finally:
        for t in tickers:
            ib.cancelMktData(t.contract)
        ib.disconnect()
        print(f"  Disconnected. Flow data saved to: {flow_file()}")

        # Print session summary
        fp = flow_file()
        if fp.exists():
            with open(fp) as f:
                data = json.load(f)
            calls = [r for r in data if r["right"] == "C"]
            puts  = [r for r in data if r["right"] == "P"]
            t2    = [r for r in data if r["notional"] >= T2_NOTIONAL]
            print(f"\n  Session summary:")
            print(f"    Total qualifying prints: {len(data)}")
            print(f"    Call flow: ${sum(r['notional'] for r in calls):,}")
            print(f"    Put flow:  ${sum(r['notional'] for r in puts):,}")
            print(f"    Alerts sent (T2+): {len(t2)}")

if __name__ == "__main__":
    main()
