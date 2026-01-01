import time
import pandas as pd
from build_dataset import build_feature_dataset
from model import spicy_sauce
from twilio.rest import Client
import datetime as dt
import os


# ==============================
# CONFIG
# ==============================
TICKERS = ["^GSPC", "TSLA"]
TIMEFRAME = "5m"
SLEEP_SECONDS = 330  # 5.5 minutes

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


# ==============================
# MAIN LOOP
# ==============================
print("🚨 INSANE Spicy Alert Engine started (5m)")

while True:
    try:
        for ticker in TICKERS:
            df = build_feature_dataset(
                ticker,
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
                signal = "TURN UP"
            elif last["Turn_Down"]:
                signal = "TURN DOWN"

            # ------------------------------
            # Alert (deduplicated)
            # ------------------------------
            if signal:
                if last_alert.get(ticker) != bar_time:
                    msg = (
                        f"INSANE ALERT 🚨\n"
                        f"{ticker}\n"
                        f"{signal}\n"
                        f"Time: {bar_time}\n"
                        f"Price: {last['Close']:.2f}"
                    )
                    send_sms(msg)
                    last_alert[ticker] = bar_time
                    print(f"[{dt.datetime.now()}] Alert sent → {ticker} {signal}")

        time.sleep(SLEEP_SECONDS)

    except Exception as e:
        print("❌ Alert engine error:", e)
        time.sleep(60)
