import yfinance as yf
import pandas as pd
from model import secret_sauce
from datetime import datetime
import json
import time
import os
import pytz 

# ============================================================
#  SNAPSHOT LEDGER — ONE JSON PER TICKER (FULL FLIP HISTORY)
# ============================================================

def get_existing_history_tickers():
    ensure_history_folder()
    return [
        f.replace(".json", "")
        for f in os.listdir("signal_history")
        if f.endswith(".json")
    ]

def ensure_history_folder():
    os.makedirs("signal_history", exist_ok=True)

def load_ticker_history(ticker):
    ensure_history_folder()
    filepath = f"signal_history/{ticker}.json"

    if not os.path.exists(filepath):
        return []  # no history yet

    with open(filepath, "r") as f:
        return json.load(f)

def save_ticker_history(ticker, history_list):
    ensure_history_folder()
    filepath = f"signal_history/{ticker}.json"

    with open(filepath, "w") as f:
        json.dump(history_list, f, indent=4)

def snapshot_all_signals_first_time(ticker, df):
    history = []
    last_signal = None

    for idx, row in df.iterrows():

        # Detect REAL flips (ignore repeated up/down)
        if row["Turn_Up"]:
            if last_signal == "UP":
                continue
            signal_type = "UP"
            last_signal = "UP"

        elif row["Turn_Down"]:
            if last_signal == "DOWN":
                continue
            signal_type = "DOWN"
            last_signal = "DOWN"

        else:
            continue

        # Full detail flip record
        record = {
            "date": idx.strftime("%Y-%m-%d"),
            "signal": signal_type,
            "turn_up": bool(row["Turn_Up"]),
            "turn_down": bool(row["Turn_Down"]),
            "close": float(row["Close"]),
            "smooth": float(row["Smooth"]),
            "slope": float(row["Slope"])
        }

        history.append(record)
    
    
    cst = pytz.timezone('America/Chicago')
    now_cst = datetime.now(cst)
    if now_cst.hour > 15:
        save_ticker_history(ticker, history)

def snapshot_new_signals_only(ticker, df):
    history = load_ticker_history(ticker)

    # FIRST RUN EVER → save full flip history
    if len(history) == 0:
        snapshot_all_signals_first_time(ticker, df)
        return

    last_saved_signal = history[-1]["signal"]
    last_saved_date = history[-1]["date"]

    # Append only REAL flips after last saved date
    for idx, row in df.iterrows():
        date_str = idx.strftime("%Y-%m-%d")
        if date_str <= last_saved_date:
            continue

        # Detect REAL flip
        if row["Turn_Up"]:
            if last_saved_signal == "UP":
                continue
            signal_type = "UP"

        elif row["Turn_Down"]:
            if last_saved_signal == "DOWN":
                continue
            signal_type = "DOWN"

        else:
            continue  # no flip

        # Build detail record
        record = {
            "date": date_str,
            "signal": signal_type,
            "turn_up": bool(row["Turn_Up"]),
            "turn_down": bool(row["Turn_Down"]),
            "close": float(row["Close"]),
            "smooth": float(row["Smooth"]),
            "slope": float(row["Slope"])
        }

        # Append + update last_saved_signal
        history.append(record)
        last_saved_signal = signal_type

    save_ticker_history(ticker, history)




# ----------------------------------------------------
# 1) Load S&P500 List (static file or online)
# ----------------------------------------------------
sp500_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
sp500 = pd.read_csv(sp500_url)


# TICKERS = sorted(sp500['Symbol'].unique())
# # Optional: add SPY, QQQ
# TICKERS += ["SPY", "QQQ", "^GSPC", "^IXIC", "^RUT", "^VIX"]

base_tickers = set(sp500['Symbol'].unique())
extra_tickers = {"SPY", "QQQ", "^GSPC", "^IXIC", "^RUT", "^VIX"}
history_tickers = set(get_existing_history_tickers())

TICKERS = sorted(base_tickers | extra_tickers | history_tickers)


# ----------------------------------------------------
# 2) Function: Run Model & Signals
# ----------------------------------------------------
def compute_signals(df):
    df = df.dropna().copy()
    df["Smooth"], df["Slope"] = secret_sauce(df["Close"])
    df["price_delta"] = df["Close"] - df["Smooth"]

    # Slope strength bands
    slope_q = df["Slope"].quantile([0.05, 0.35, 0.50, 0.65, 0.95]).tolist()
    slope_vals = [round(x / 0.25) * 0.25 for x in slope_q]

    df["Slope_Neg"] = (
        (df["Slope"] < (slope_vals[0] + slope_vals[1]) / 2) &
        (df["Close"] < df["Smooth"]) &
        (df["Slope"] < df["Slope"].shift(1))
    )

    df["Slope_Pos"] = (
        (df["Slope"] > (slope_vals[3] + slope_vals[4]) / 2) &
        (df["Close"] > df["Smooth"]) &
        (df["Slope"] > df["Slope"].shift(1))
    )

    df["Turn_Up"] = df["Slope_Pos"] & (~df["Slope_Pos"].shift(1).fillna(False).astype(bool))
    df["Turn_Down"] = df["Slope_Neg"] & (~df["Slope_Neg"].shift(1).fillna(False).astype(bool))

    return df


# ----------------------------------------------------
# 3) MAIN ENGINE
# ----------------------------------------------------
def run_daily_engine():

    long_signals = []
    short_signals = []
    last_signal = {}    
    today = None

    for ticker in TICKERS:
        print(f"processing {ticker}...")
        try:
            df = yf.download(ticker, period="5y", interval="1d", auto_adjust=False)
            if df.empty:
                continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.rename(columns={"Adj Close": "Adj_Close"})

            df = compute_signals(df)
            today = df.index[-1]
            snapshot_new_signals_only(ticker, df)
            
            if df["Turn_Up"].iloc[-1]:
                long_signals.append(ticker)

            if df["Turn_Down"].iloc[-1]:
                short_signals.append(ticker)

            # Last signal for special tickers SPY & QQQ
            if ticker in ["SPY", "QQQ", "^GSPC", "^IXIC", "^RUT", "^VIX"]:

                # Find last Turn Up
                if df["Turn_Up"].any():
                    last_up_idx = df[df["Turn_Up"]].index[-1]
                else:
                    last_up_idx = None

                # Find last Turn Down
                if df["Turn_Down"].any():
                    last_down_idx = df[df["Turn_Down"]].index[-1]
                else:
                    last_down_idx = None

                # Decide which one is latest
                if last_up_idx and (not last_down_idx or last_up_idx > last_down_idx):
                    last_signal[ticker] = {
                        "signal": "LONG",
                        "date": last_up_idx.strftime("%Y-%m-%d")
                    }
                elif last_down_idx and (not last_up_idx or last_down_idx > last_up_idx):
                    last_signal[ticker] = {
                        "signal": "SHORT",
                        "date": last_down_idx.strftime("%Y-%m-%d")
                    }
                else:
                    last_signal[ticker] = {"signal": "NONE", "date": None}

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # If nothing worked
    if today is None:
        print("No tickers processed. Aborting.")
        return
    
    # ----------------------------------------------------
    # SAVE RESULT
    # ----------------------------------------------------
    output = {
        "date": today.strftime("%Y-%m-%d"),
        "long": long_signals,
        "short": short_signals,
        "nasdaq": last_signal.get("^IXIC", {"signal": "NONE", "date": None}),
        "qqq": last_signal.get("QQQ", {"signal": "NONE", "date": None}),
        "spx": last_signal.get("^GSPC", {"signal": "NONE", "date": None}),
        "spy": last_signal.get("SPY", {"signal": "NONE", "date": None}),
        "russell": last_signal.get("^RUT", {"signal": "NONE", "date": None}),
        "vix": last_signal.get("^VIX", {"signal": "NONE", "date": None}),
        }

    with open("daily_signals.json", "w") as f:
        json.dump(output, f, indent=4)

    print("Daily signals updated!")


if __name__ == "__main__":
    start = time.time()
    run_daily_engine()
    end = time.time()
    print(f"\nTotal runtime: {end - start:.2f} seconds")
