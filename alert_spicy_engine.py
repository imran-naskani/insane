import time
import pandas as pd
from build_dataset import build_feature_dataset
from model import spicy_sauce
from twilio.rest import Client
import datetime as dt
from dotenv import load_dotenv
import os

load_dotenv()

end_date = dt.date.today() + dt.timedelta(days=1)
start_date = end_date - dt.timedelta(days=31)

# ==============================
# CONFIG
# ==============================
TICKERS = ["^GSPC", "TSLA"]
TIMEFRAME = "5m"

TWILIO_SID = os.environ["TWILIO_SID"]
TWILIO_TOKEN = os.environ["TWILIO_TOKEN"]
TWILIO_FROM = os.environ["TWILIO_FROM"]
ALERT_TO = os.environ["ALERT_TO"].split(",")

# ==============================
# INIT
# ==============================
twilio = Client(TWILIO_SID, TWILIO_TOKEN)
last_alert = {}   # { ticker: bar_timestamp }


# ==============================
# HELPERS
# ==============================
def send_sms(message: str):
    for number in ALERT_TO:
        twilio.messages.create(
            body=message,
            from_=TWILIO_FROM,
            to=number
        )


def get_last_closed_bar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the currently forming 5m candle.
    Assumes data is ordered.
    """
    return df.iloc[:-1]

def sleep_until_next_5m(offset_seconds=10):
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
        sleep_until_next_5m(offset_seconds=10)
        for ticker in TICKERS:
            df = build_feature_dataset(
                ticker,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"), 
                timeframe=TIMEFRAME
            )

            if len(df) < 10:
                continue

            # ------------------------------
            # Remove partial candle
            # ------------------------------
            df = get_last_closed_bar(df)

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

            df["Slope_Neg"] = df["price_delta"] < df["q05"]
            df["Slope_Pos"] = df["price_delta"] > df["q95"]

            df["Turn_Up"] = df["Slope_Pos"] & (~df["Slope_Pos"].shift(1).fillna(False))
            df["Turn_Down"] = df["Slope_Neg"] & (~df["Slope_Neg"].shift(1).fillna(False))

            # ------------------------------
            # Check LAST CLOSED BAR ONLY
            # ------------------------------
            last = df.iloc[-1]
            bar_time = df.index[-1]

            signal = None
            if last["Turn_Up"]:
                signal = "Momentum Rising - Potential Long"
            elif last["Turn_Down"]:
                signal = "Momentum Declining - Potential Short"

            # ------------------------------
            # Alert (deduplicated)
            # ------------------------------
            if signal:
                if last_alert.get(ticker) != bar_time:
                    msg = (
                        f"INSANE 5 min ALERT 🚨\n"
                        f"{ticker}\n"
                        f"{signal}\n"
                        f"Time: {bar_time}\n"
                        f"Price: {last['Close']:.2f}"
                    )
                    send_sms(msg)
                    last_alert[ticker] = bar_time
                    print(f"[{dt.datetime.now()}] Alert sent → {ticker} {signal}")
            else:
                print("No signal detected!")

    except Exception as e:
        print("❌ Alert engine error:", e)
        time.sleep(60)
