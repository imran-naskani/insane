"""
INSANE — Weekly Earnings Scanner
=================================
Fetches upcoming NASDAQ earnings for the next 7 days from Finnhub,
cross-references with signal_history, generates signals for any
missing tickers, and produces a consolidated watchlist.

Schedule via Windows Task Scheduler (e.g., every Sunday evening):
    python earnings_scanner.py
"""

import os
import json
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Reuse existing project functions
from daily_engine import (
    load_ticker_history,
    save_ticker_history,
    compute_signals,
)

load_dotenv()

FINNHUB_API_KEY = os.environ["FINNHUB_API_KEY"]
FINNHUB_BASE = "https://finnhub.io/api/v1"
OUTPUT_FILE = "earnings_watchlist.json"


# ============================================================
#  1) FINNHUB — EARNINGS CALENDAR
# ============================================================
def fetch_earnings_calendar(from_date: str, to_date: str) -> list[dict]:
    """
    Returns list of earnings events between from_date and to_date.
    Each item has: symbol, date, hour, epsEstimate, revenueEstimate, etc.
    """
    url = f"{FINNHUB_BASE}/calendar/earnings"
    params = {
        "from": from_date,
        "to": to_date,
        "token": FINNHUB_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("earningsCalendar", [])


# ============================================================
#  2) FINNHUB — NASDAQ SYMBOL SET
# ============================================================
def fetch_nasdaq_symbols() -> set[str]:
    """
    Fetches all US-listed symbols and filters to NASDAQ (MIC = XNAS).
    Returns a set of ticker strings.
    """
    url = f"{FINNHUB_BASE}/stock/symbol"
    params = {
        "exchange": "US",
        "token": FINNHUB_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    symbols = resp.json()
    return {s["symbol"] for s in symbols if s.get("mic") == "XNAS"}


# ============================================================
#  3) GET LAST SIGNAL FROM HISTORY (or generate it)
# ============================================================
def get_last_signal(ticker: str) -> dict:
    """
    Looks up signal_history/{ticker}.json.
    If missing, downloads 5y daily data, runs compute_signals(),
    saves the full signal history, then returns last signal.

    Returns: {"signal": "UP"|"DOWN"|"NONE", "date": "YYYY-MM-DD" | None}
    """
    history = load_ticker_history(ticker)

    # ---- History exists → grab last entry ----
    if history:
        last = history[-1]
        return {"signal": last["signal"], "date": last["date"], "close": last.get("close")}

    # ---- No history → generate from scratch ----
    print(f"  ⬇  No signal history for {ticker}. Downloading & generating...")
    MAX_RETRIES = 5
    RETRY_SLEEP = 3
    df = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(ticker, period="5y", interval="1d", auto_adjust=False)
            if df is not None and not df.empty:
                break
        except Exception as dl_err:
            print(f"     Attempt {attempt}/{MAX_RETRIES} failed for {ticker}: {dl_err}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)

    if df is None or df.empty:
        print(f"  ⚠  No data returned for {ticker} after {MAX_RETRIES} attempts. Skipping.")
        return {"signal": "NONE", "date": None, "close": None}

    try:

        # Flatten MultiIndex columns if present (yfinance quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns={"Adj Close": "Adj_Close"})
        df = compute_signals(df)

        # Build full flip history (same logic as snapshot_all_signals_first_time
        # but without the 3 PM CST gate so it always saves)
        flip_history = []
        last_signal_type = None

        for idx, row in df.iterrows():
            if row["Turn_Up"]:
                if last_signal_type == "UP":
                    continue
                signal_type = "UP"
                last_signal_type = "UP"
            elif row["Turn_Down"]:
                if last_signal_type == "DOWN":
                    continue
                signal_type = "DOWN"
                last_signal_type = "DOWN"
            else:
                continue

            flip_history.append({
                "date": idx.strftime("%Y-%m-%d"),
                "signal": signal_type,
                "turn_up": bool(row["Turn_Up"]),
                "turn_down": bool(row["Turn_Down"]),
                "close": float(row["Close"]),
                "smooth": float(row["Smooth"]),
                "slope": float(row["Slope"]),
            })

        if flip_history:
            save_ticker_history(ticker, flip_history)
            last = flip_history[-1]
            return {"signal": last["signal"], "date": last["date"], "close": last.get("close")}
        else:
            return {"signal": "NONE", "date": None, "close": None}

    except Exception as e:
        print(f"  ❌  Error generating signals for {ticker}: {e}")
        return {"signal": "NONE", "date": None, "close": None}


# ============================================================
#  4) MAIN — BUILD EARNINGS WATCHLIST
# ============================================================
def run_earnings_scanner():
    today = datetime.today()
    from_date = today.strftime("%Y-%m-%d")
    to_date = (today + timedelta(days=7)).strftime("%Y-%m-%d")

    print(f"📅  Scanning earnings from {from_date} to {to_date}\n")

    # --- Fetch earnings calendar ---
    print("1️⃣  Fetching Finnhub earnings calendar...")
    all_earnings = fetch_earnings_calendar(from_date, to_date)
    print(f"   Found {len(all_earnings)} total earnings events.\n")

    if not all_earnings:
        print("No earnings found for the coming week. Exiting.")
        return

    # --- Fetch NASDAQ symbols ---
    print("2️⃣  Fetching NASDAQ symbol list from Finnhub...")
    nasdaq_symbols = fetch_nasdaq_symbols()
    print(f"   {len(nasdaq_symbols)} NASDAQ symbols loaded.\n")

    # --- Filter to NASDAQ only ---
    nasdaq_earnings = [e for e in all_earnings if e.get("symbol") in nasdaq_symbols]
    print(f"3️⃣  {len(nasdaq_earnings)} NASDAQ earnings events in the next 7 days.\n")

    if not nasdaq_earnings:
        print("No NASDAQ earnings found. Exiting.")
        return

    # --- Build watchlist with signals ---
    total = len(nasdaq_earnings)
    print(f"4️⃣  Resolving signals for {total} tickers...\n")
    watchlist = []

    for i, event in enumerate(nasdaq_earnings, 1):
        ticker = event["symbol"]
        print(f"  [{i}/{total}]  🔍  {ticker} — earnings {event.get('date', '?')}")

        sig = get_last_signal(ticker)

        watchlist.append({
            "symbol": ticker,
            "earnings_date": event.get("date"),
            "hour": event.get("hour", ""),           # bmo / amc / dmh
            "eps_estimate": event.get("epsEstimate"),
            "revenue_estimate": event.get("revenueEstimate"),
            "last_signal": "Long" if sig["signal"] == "UP" else ("Short" if sig["signal"] == "DOWN" else "None"),
            "signal_date": sig["date"],
            "last_price": sig["close"],
        })

    # --- Save output ---
    output = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scan_from": from_date,
        "scan_to": to_date,
        "count": len(watchlist),
        "watchlist": watchlist,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=4)

    # --- Console summary ---
    print(f"\n{'='*80}")
    print(f"  INSANE Earnings Watchlist — {from_date} to {to_date}")
    print(f"  {len(watchlist)} NASDAQ stocks with upcoming earnings")
    print(f"{'='*80}\n")

    df_out = pd.DataFrame(watchlist)
    cols = ["symbol", "earnings_date", "hour", "eps_estimate",
            "revenue_estimate", "last_signal", "signal_date", "last_price"]
    print(df_out[cols].to_string(index=False))
    print(f"\n✅  Saved to {OUTPUT_FILE}")


# ============================================================
if __name__ == "__main__":
    start = time.time()
    run_earnings_scanner()
    elapsed = time.time() - start
    print(f"\n⏱  Total runtime: {elapsed:.1f}s")
