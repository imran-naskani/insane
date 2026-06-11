import os
import pytest
from unittest.mock import patch, MagicMock

import telegram as tg

_MAIN_ENV = {
    "TELEGRAM_BOT_TOKEN":        "test-main-token",
    "TELEGRAM_CHAT_ID":           "test-main-chat",
    "TELEGRAM_OPTION_BOT_TOKEN": "test-opts-token",
    "TELEGRAM_OPTION_CHAT_ID":    "test-opts-chat",
}


def test_send_alert_main_channel_uses_main_token():
    with patch.dict(os.environ, _MAIN_ENV), patch("telegram.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()
        tg.send_alert("hello", channel="main")
        args, kwargs = mock_post.call_args
        assert "test-main-token" in args[0]
        assert kwargs["json"]["chat_id"] == "test-main-chat"
        assert kwargs["json"]["text"] == "hello"


def test_send_alert_options_channel_uses_options_token():
    with patch.dict(os.environ, _MAIN_ENV), patch("telegram.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()
        tg.send_alert("trade alert", channel="options")
        args, kwargs = mock_post.call_args
        assert "test-opts-token" in args[0]
        assert kwargs["json"]["chat_id"] == "test-opts-chat"


def test_send_alert_default_channel_is_main():
    with patch.dict(os.environ, _MAIN_ENV), patch("telegram.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()
        tg.send_alert("hello")
        args, _ = mock_post.call_args
        assert "test-main-token" in args[0]


def test_send_alert_swallows_network_exception():
    with patch.dict(os.environ, _MAIN_ENV), patch("telegram.requests.post", side_effect=Exception("timeout")):
        tg.send_alert("test", channel="main")  # must not raise
