import time
import pandas as pd
from build_dataset import build_feature_dataset
from model import spicy_sauce
import datetime as dt
from dotenv import load_dotenv
import os
import requests

load_dotenv()

end_date = dt.date.today() + dt.timedelta(days=1)
start_date = end_date - dt.timedelta(days=31)

# ==============================
# CONFIG
# ==============================
TICKERS = ["^GSPC", "TSLA"]
TIMEFRAME = "5m"

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

RETRY_WINDOW_SECONDS = 45   # how long to retry within same bar
RETRY_SLEEP_SECONDS = 3     # wait between retries

# ==============================
# STATE
# ==============================
last_alert = {}   # { ticker: (bar_timestamp, signal_type) }


# ==============================
# HELPERS
# ==============================
def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown"
    }
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()

def get_last_closed_bar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the currently forming 5m candle.
    Assumes data is ordered.
    """
    return df.iloc[:-1]

def sleep_until_next_5m(offset_seconds=3):
    """
    Sleeps until the next 5-minute boundary + offset.
    Offset ensures bar is fully closed and data is available.
    """
    now = time.time()

    interval = 300  # 5 minutes in seconds
    next_run = ((now // interval) + 1) * interval + offset_seconds

    sleep_for = max(0, next_run - now)
    time.sleep(sleep_for)

# ==============================
# MAIN LOOP
# ==============================
print("🚨 INSANE Spicy Alert Engine started (5m)")

while True:
    try:

        sleep_until_next_5m(offset_seconds=3)

        bar_start_time = time.time()
        bar_deadline = bar_start_time + RETRY_WINDOW_SECONDS

        combined_msgs = []
        current_bar_time = None

        for ticker in TICKERS:
            df = None
            last_exception = None
            while time.time() < bar_deadline:
                try:
                    df = build_feature_dataset(
                        ticker,
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"), 
                        timeframe=TIMEFRAME
                    )

                    # Require at least one closed candle
                    if df is not None and len(df) >= 10:
                        temp = get_last_closed_bar(df)
                        if temp is not None and len(temp) >= 10:
                            last_bar = temp.index[-1]
                            prev = last_alert.get(ticker)
                            if prev is None or last_bar > prev[0]:
                                df = temp
                                break


                except Exception as e:
                    last_exception = e

                time.sleep(RETRY_SLEEP_SECONDS)

            if df is None or len(df) < 10:
                print(f"[WARN] {ticker}: Yahoo retry failed for this bar")
                continue
        
            # ------------------------------
            # Remove partial candle
            # ------------------------------
            # df = get_last_closed_bar(df)
            # if df is None or len(df) < 10:
            #     continue

            last_bar_time = df.index[-1]

            # If bar is not new, skip this ticker (do NOT break loop)
            prev = last_alert.get(ticker)
            if prev is not None and last_bar_time <= prev[0]:
                continue

            # ------------------------------
            # Timezone fix: UTC → US/Eastern
            # ------------------------------
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")

            df.index = df.index.tz_convert("US/Central")

            # ------------------------------
            # Run spicy_sauce
            # ------------------------------
            df["Smooth"], df["Slope"] = spicy_sauce(df["Close"])

            # ------------------------------
            # Recreate your intraday signal logic
            # ------------------------------
            df["price_delta"] = df["Close"] - df["Smooth"]

            df["q05"] = df["price_delta"].rolling(84).quantile(0.05)
            df["q95"] = df["price_delta"].rolling(84).quantile(0.95)

            df["Slope_Neg"] = (df["price_delta"] < df["q05"]) & (df["Close"] < df["TOS_Trail"])
            df["Slope_Pos"] = (df["price_delta"] > df["q95"]) & (df["Close"] > df["TOS_Trail"])

            df["Turn_Up"] = df["Slope_Pos"] & (~df["Slope_Pos"].shift(1).fillna(False))
            df["Turn_Down"] = df["Slope_Neg"] & (~df["Slope_Neg"].shift(1).fillna(False))

            # ------------------------------
            # Exit Logic
            # ------------------------------
            
            df["Position"] = 0
            for i in range(1, len(df)):
                if df["Turn_Up"].iloc[i]:
                    df.at[df.index[i], "Position"] = 1

                elif df["Turn_Down"].iloc[i]:
                    df.at[df.index[i], "Position"] = -1

                else:
                    df.at[df.index[i], "Position"] = df["Position"].iloc[i-1]
            df["Sell_Long"] = (
                (df["Position"] == 1) &
                (
                    ((df["Close"].shift(1) >= df["VWAP_Upper"].shift(1)) &
                        (df["Close"] < df["VWAP_Upper"])) | 
                        ((df["Close"].shift(1) >= df["VWAP"].shift(1)) &
                        (df["Close"] < df["VWAP"])) |
                        ((df["Low"].shift(1) >= df["TOS_Trail"].shift(1)) &
                        (df["Low"] < df["TOS_Trail"])) |
                        ((df["Low"].shift(1) >= df["Low"]) &
                        (df["Close"].shift(1) >= df["Close"])) |
                        ((df["High"].shift(1) >= df["High"]) &
                        (df["Close"].shift(1) >= df["Close"])) 
                )
                )

            df["Sell_Short"] = (
                (df["Position"] == -1) &
                (
                    ((df["Close"].shift(1) <= df["VWAP_Lower"].shift(1)) &
                        (df["Close"] > df["VWAP_Lower"])) |
                        ((df["Close"].shift(1) <= df["VWAP"].shift(1)) &
                        (df["Close"] > df["VWAP"])) | 
                        ((df["High"].shift(1) <= df["TOS_Trail"].shift(1)) &
                        (df["High"] > df["TOS_Trail"])) |
                        ((df["Low"].shift(1) <= df["Low"]) &
                        (df["Close"].shift(1) <= df["Close"])) |
                        ((df["High"].shift(1) <= df["High"]) &
                        (df["Close"].shift(1) <= df["Close"])) 
                )
            )

            # ------------------------------
            # Check LAST CLOSED BAR ONLY
            # ------------------------------
            last = df.iloc[-1]
            bar_time = df.index[-1]
            display_time = bar_time.strftime("%H:%M CT")

            signal = None
            signal_type = None

            if last["Turn_Up"]:
                signal = "Momentum Rising - Potential Long"
                signal_type = "TURN_UP"
            elif last["Turn_Down"]:
                signal = "Momentum Declining - Potential Short"
                signal_type = "TURN_DOWN"
            elif last["Sell_Long"]:
                signal = "Exit Warning - Close Long"
                signal_type = "EXIT"
            elif last["Sell_Short"]:
                signal = "Exit Warning - Close Short"
                signal_type = "EXIT"


            # prev = last_alert.get(ticker)
            if signal:
                if prev is None:
                    allow = True
                else:
                    prev_time, prev_type = prev
                    # Block repeated EXITs
                    allow = not (signal_type == "EXIT" and prev_type == "EXIT")

                if allow:
                    combined_msgs.append(
                        f"*{'SPX' if ticker == '^GSPC' else ticker}*\n"
                        f"{signal}\n"
                        f"Price: {last['Close']:.2f}"
                    )
                    last_alert[ticker] = (bar_time, signal_type)

        
        # ------------------------------
        # Send ONE combined Telegram message
        # ------------------------------
        if combined_msgs:
            final_msg = (
                "🚨 INSANE ALERT 🚨\n\n"
                f"*Time:* {display_time}\n\n"
                + "\n\n".join(combined_msgs)
            )
            send_telegram(final_msg)
            print(f"[{dt.datetime.now()}] Combined alert sent")

    except Exception as e:
        print("❌ Alert engine error:", e)
        time.sleep(60)
