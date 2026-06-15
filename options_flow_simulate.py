"""
options_flow_simulate.py

Simulates TSLA options flow using Polygon /v3/trades + /v3/quotes endpoints.
Individual fill-level data — each record is one actual trade print, not a 1-min bar.

Run: python options_flow_simulate.py
"""

import os, json, time, requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
API_KEY = os.getenv("MASSIVE_API_KEY", os.getenv("POLYGON_API_KEY", "")).strip()

# ── CONFIG ────────────────────────────────────────────────────────────────────
SIM_DATE       = "2026-06-12"
TICKER         = "TSLA"
EXPIRY_0DTE    = "20260612"
EXPIRY_NEXT    = "20260617"
ATM_REF        = 392.5
STRIKE_STEP    = 2.5
N_STRIKES      = 10
RIGHTS         = ["C", "P"]

# Alert filters
MIN_CONTRACTS  = 50       # minimum single fill size
MIN_NOTIONAL   = 25_000   # minimum premium per fill
T2_NOTIONAL    = 75_000   # Telegram alert
T3_NOTIONAL    = 200_000  # Strong alert

SESSION_START_ET = "2026-06-12T09:30:00"   # ET (Polygon uses ET)
SESSION_END_ET   = "2026-06-12T16:00:00"
PRIME_START    = pd.Timestamp("09:45").time()  # ET
PRIME_END      = pd.Timestamp("13:30").time()  # ET
SKIP_OPEN_MIN  = 15  # skip first 15 min of session (opening rush)

# Polygon condition codes
SWEEP_CONDITIONS = {14, 41}  # 14=intermarket sweep, 41=ISO sweep

CACHE_DIR = Path("polygon_tick_cache")
CACHE_DIR.mkdir(exist_ok=True)
API_SLEEP = 12

# ── HELPERS ───────────────────────────────────────────────────────────────────
def option_symbol(ticker, expiry, right, strike):
    exp_yy   = expiry[2:]
    strike_i = int(round(strike * 1000))
    return f"O:{ticker}{exp_yy}{right}{strike_i:08d}"

def polygon_get(url, params, cache_file):
    """Fetch all pages for a Polygon v3 endpoint, cache result."""
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    all_results = []
    next_url = url
    first = True
    while next_url:
        if first:
            r = requests.get(next_url, params={**params, "apiKey": API_KEY}, timeout=15)
            first = False
        else:
            r = requests.get(next_url, params={"apiKey": API_KEY}, timeout=15)

        if r.status_code == 403:
            print(f"    403 — plan limit for {cache_file.name}")
            time.sleep(API_SLEEP)
            return []
        if r.status_code != 200:
            print(f"    HTTP {r.status_code} for {cache_file.name}")
            time.sleep(API_SLEEP)
            return []

        data = r.json()
        all_results.extend(data.get("results", []))
        next_url = data.get("next_url")   # pagination
        if next_url:
            time.sleep(API_SLEEP)         # rate-limit between pages

    time.sleep(API_SLEEP)
    with open(cache_file, "w") as f:
        json.dump(all_results, f)
    return all_results

def fetch_trades(sym, date):
    safe = sym.replace(":", "_")
    cf   = CACHE_DIR / f"{safe}_{date}_trades.json"
    url  = f"https://api.polygon.io/v3/trades/{sym}"
    params = {
        "timestamp.gte": f"{date}T14:30:00Z",  # 09:30 ET = 14:30 UTC
        "timestamp.lte": f"{date}T21:00:00Z",  # 16:00 ET = 21:00 UTC
        "order": "asc",
        "limit": 50000,
    }
    return polygon_get(url, params, cf)

def fetch_quotes(sym, date):
    safe = sym.replace(":", "_")
    cf   = CACHE_DIR / f"{safe}_{date}_quotes.json"
    url  = f"https://api.polygon.io/v3/quotes/{sym}"
    params = {
        "timestamp.gte": f"{date}T14:30:00Z",
        "timestamp.lte": f"{date}T21:00:00Z",
        "order": "asc",
        "limit": 50000,
    }
    return polygon_get(url, params, cf)

# ── BUILD QUOTE LOOKUP ────────────────────────────────────────────────────────
def build_quote_index(quotes):
    """Return list of (ts_ns, bid, ask) sorted by time for fast lookup."""
    idx = []
    for q in quotes:
        ts = q.get("sip_timestamp") or q.get("participant_timestamp") or q.get("t", 0)
        bid = q.get("bid_price", 0)
        ask = q.get("ask_price", 0)
        if bid > 0 and ask > 0:
            idx.append((ts, bid, ask))
    return sorted(idx, key=lambda x: x[0])

def get_bid_ask_at(quote_index, ts_ns):
    """Binary search for most recent quote before ts_ns."""
    lo, hi = 0, len(quote_index) - 1
    result = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if quote_index[mid][0] <= ts_ns:
            result = quote_index[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return result  # (ts_ns, bid, ask) or None

def classify_side(price, bid, ask):
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return "UNKNOWN"
    mid = (bid + ask) / 2
    spread = ask - bid
    if price >= ask:                    return "ASK+"
    elif price <= bid:                  return "BID-"
    elif price > mid + spread * 0.1:   return "ABOVE MID"
    elif price < mid - spread * 0.1:   return "BELOW MID"
    else:                               return "MID"

# ── MAIN ──────────────────────────────────────────────────────────────────────
def run():
    strikes = [round(ATM_REF + i * STRIKE_STEP, 1)
               for i in range(-N_STRIKES, N_STRIKES + 1)]

    print(f"\n{'='*65}")
    print(f"  TSLA Options Flow — {SIM_DATE}  (individual fill-level)")
    print(f"  ATM: ${ATM_REF}  |  Strikes: {strikes[0]}-{strikes[-1]}")
    print(f"  Filter: single fill >= {MIN_CONTRACTS} contracts AND >= ${MIN_NOTIONAL:,} notional")
    print(f"  Sweep condition codes: {SWEEP_CONDITIONS}")
    print(f"{'='*65}\n")

    all_alerts = []
    contracts_fetched = 0
    contracts_no_data = 0

    for expiry, label in [(EXPIRY_0DTE, "0DTE"), (EXPIRY_NEXT, "NEXT")]:
        for right in RIGHTS:
            for strike in strikes:
                sym = option_symbol(TICKER, expiry, right, strike)
                contracts_fetched += 1

                trades = fetch_trades(sym, SIM_DATE)
                if not trades:
                    contracts_no_data += 1
                    continue

                quotes = fetch_quotes(sym, SIM_DATE)
                quote_idx = build_quote_index(quotes)

                for tr in trades:
                    ts_ns  = tr.get("sip_timestamp") or tr.get("participant_timestamp", 0)
                    price  = tr.get("price", 0)
                    size   = tr.get("size", 0)
                    conds  = set(tr.get("conditions", []) or [])

                    if price <= 0 or size <= 0: continue

                    notional = round(price * size * 100)

                    # Core filters
                    if size < MIN_CONTRACTS:    continue
                    if notional < MIN_NOTIONAL: continue

                    # Convert timestamp to ET for display and time filtering
                    ts_et = pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert("US/Eastern")
                    if ts_et.time() < PRIME_START: continue
                    if ts_et.time() > PRIME_END:   continue

                    # Bid/ask context
                    q = get_bid_ask_at(quote_idx, ts_ns)
                    bid = q[1] if q else None
                    ask = q[2] if q else None
                    side = classify_side(price, bid, ask)

                    is_sweep = bool(conds & SWEEP_CONDITIONS)

                    all_alerts.append({
                        "ts":       ts_et,
                        "time":     ts_et.strftime("%H:%M:%S"),
                        "expiry":   label,
                        "strike":   strike,
                        "right":    right,
                        "size":     size,
                        "price":    price,
                        "bid":      round(bid, 2) if bid else None,
                        "ask":      round(ask, 2) if ask else None,
                        "notional": notional,
                        "side":     side,
                        "sweep":    is_sweep,
                        "conds":    sorted(conds),
                    })

    print(f"Contracts fetched: {contracts_fetched}  |  No data: {contracts_no_data}\n")

    if not all_alerts:
        print("No qualifying prints found.")
        return

    all_alerts.sort(key=lambda x: x["ts"])

    # ── PRINT TAPE ────────────────────────────────────────────────────────────
    print(f"{'TIME':>10}  {'CONTRACT':<14} {'SIZE':>5}  {'PRICE':>6}  {'BID':>5}  {'ASK':>5}  {'NOTIONAL':>10}  {'SIDE':<10}  NOTE")
    print("-" * 90)

    for a in all_alerts:
        if a["notional"] < T2_NOTIONAL: continue  # only show T2+ in tape

        tier  = "***" if a["notional"] >= T3_NOTIONAL else "** "
        sweep = "[SWEEP]" if a["sweep"] else "       "
        bid_s = f"${a['bid']:.2f}" if a["bid"] else "  n/a"
        ask_s = f"${a['ask']:.2f}" if a["ask"] else "  n/a"
        contract = f"{a['strike']}{a['right']} {a['expiry']}"

        print(f"{a['time']:>10}  {contract:<14} {a['size']:>5}  "
              f"${a['price']:>5.2f}  {bid_s:>6}  {ask_s:>6}  "
              f"${a['notional']:>9,}  {a['side']:<10}  {tier} {sweep}")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    t2 = [a for a in all_alerts if a["notional"] >= T2_NOTIONAL]
    t3 = [a for a in all_alerts if a["notional"] >= T3_NOTIONAL]
    sw = [a for a in all_alerts if a["sweep"]]
    call_flow = sum(a["notional"] for a in all_alerts if a["right"] == "C")
    put_flow  = sum(a["notional"] for a in all_alerts if a["right"] == "P")
    net       = call_flow - put_flow

    print(f"\n{'='*65}")
    print(f"SUMMARY  ({PRIME_START.strftime('%H:%M')}-{PRIME_END.strftime('%H:%M')} ET, fills >= {MIN_CONTRACTS} contracts)")
    print(f"{'='*65}")
    print(f"  Qualifying fills (>={MIN_CONTRACTS} contracts):  {len(all_alerts)}")
    print(f"  T2 alerts  (>=${T2_NOTIONAL//1000}K):            {len(t2)}")
    print(f"  T3 alerts  (>=${T3_NOTIONAL//1000}K):            {len(t3)}")
    print(f"  Sweeps detected:                {len(sw)}")
    print()
    print(f"  Call premium:   ${call_flow:>12,}  ({call_flow/(call_flow+put_flow)*100:.0f}%)" if call_flow+put_flow else "")
    print(f"  Put premium:    ${put_flow:>12,}  ({put_flow/(call_flow+put_flow)*100:.0f}%)" if call_flow+put_flow else "")
    print(f"  Net flow:       ${net:>+12,}  -> {'BULLISH' if net > 0 else 'BEARISH'}")
    print()

    # Top strikes
    from collections import defaultdict
    by_strike = defaultdict(int)
    for a in all_alerts:
        by_strike[f"{a['strike']}{a['right']}"] += a["notional"]
    top = sorted(by_strike.items(), key=lambda x: -x[1])[:8]
    print("  Top strikes by qualifying notional:")
    for k, v in top:
        print(f"    {k:<8}  ${v:>10,}")

    # Telegram sample messages
    print(f"\n  Sample Telegram alerts (T2+ shown):")
    shown = 0
    for a in all_alerts:
        if a["notional"] < T2_NOTIONAL: continue
        tier_label = "STRONG" if a["notional"] >= T3_NOTIONAL else "ALERT"
        sweep_tag  = " [SWEEP]" if a["sweep"] else ""
        bid_ask    = f" | bid ${a['bid']:.2f} ask ${a['ask']:.2f}" if a["bid"] else ""
        msg = (f"[{tier_label}] TSLA {a['strike']}{a['right']} {a['expiry']} "
               f"| {a['size']} contracts @ ${a['price']:.2f} {a['side']} "
               f"| ${a['notional']:,}{bid_ask}{sweep_tag}")
        print(f"  {a['time']}  {msg}")
        shown += 1
        if shown >= 10:
            print(f"  ... and {len(t2)-10} more T2+ alerts")
            break

if __name__ == "__main__":
    run()
