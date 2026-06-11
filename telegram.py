import os
import requests
from dotenv import load_dotenv

load_dotenv()

_CHANNELS = {
    "main":    (os.environ.get("TELEGRAM_BOT_TOKEN"),        os.environ.get("TELEGRAM_CHAT_ID")),
    "options": (os.environ.get("TELEGRAM_OPTION_BOT_TOKEN"), os.environ.get("TELEGRAM_OPTION_CHAT_ID")),
}


def send_alert(msg: str, channel: str = "main") -> None:
    if channel not in _CHANNELS:
        print(f"[telegram] unknown channel '{channel}'")
        return
    token, chat_id = _CHANNELS[channel]
    if not token or not chat_id:
        print(f"[telegram] channel '{channel}' not configured")
        return
    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[telegram] send failed ({channel}): {e}")
