"""
INSANE — Weekly Earnings Scanner
=================================
Fetches upcoming NASDAQ earnings for the next 7 days from Finnhub
and saves a lean watchlist (Finnhub data only).

Signal, strength, and price are enriched at display time by insane.py
from signal_strength_history/ — so the watchlist never goes stale.

Schedule via Windows Task Scheduler (e.g., every Sunday evening):
    python earnings_scanner.py
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

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
#  3) MAIN — BUILD EARNINGS WATCHLIST
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

    # --- Build lean watchlist (Finnhub data only) ---
    watchlist = []
    for event in nasdaq_earnings:
        watchlist.append({
            "symbol": event["symbol"],
            "earnings_date": event.get("date"),
            "hour": event.get("hour", ""),           # bmo / amc / dmh
            "eps_estimate": event.get("epsEstimate"),
            "revenue_estimate": event.get("revenueEstimate"),
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
    cols = ["symbol", "earnings_date", "hour", "eps_estimate", "revenue_estimate"]
    print(df_out[cols].to_string(index=False))
    print(f"\n✅  Saved to {OUTPUT_FILE}")
    print(f"ℹ️  Signal/strength/price enriched at display time from signal_strength_history/")


# ============================================================
if __name__ == "__main__":
    start = time.time()
    run_earnings_scanner()
    elapsed = time.time() - start
    print(f"\n⏱  Total runtime: {elapsed:.1f}s")
