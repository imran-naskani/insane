import os
import requests
from dotenv import load_dotenv

load_dotenv()

_CHANNEL_KEYS = {
    "main":    ("TELEGRAM_BOT_TOKEN",        "TELEGRAM_CHAT_ID"),
    "options": ("TELEGRAM_OPTION_BOT_TOKEN", "TELEGRAM_OPTION_CHAT_ID"),
}


def send_alert(msg: str, channel: str = "main") -> None:
    if channel not in _CHANNEL_KEYS:
        print(f"[telegram] unknown channel '{channel}'")
        return
    token_key, chat_key = _CHANNEL_KEYS[channel]
    token, chat_id = os.environ.get(token_key), os.environ.get(chat_key)
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
