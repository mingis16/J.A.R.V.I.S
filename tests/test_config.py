from __future__ import annotations

import pytest

from trading_bot.config import get_mt5_credentials, live_trading_confirmed


def _clear_mt5_env(monkeypatch):
    for var in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "MT5_TERMINAL_PATH"):
        monkeypatch.delenv(var, raising=False)


def test_get_mt5_credentials_raises_when_unset(tmp_path, monkeypatch):
    _clear_mt5_env(monkeypatch)
    monkeypatch.setattr("trading_bot.config.REPO_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="MT5_LOGIN"):
        get_mt5_credentials()


def test_get_mt5_credentials_reads_env_vars(tmp_path, monkeypatch):
    _clear_mt5_env(monkeypatch)
    monkeypatch.setattr("trading_bot.config.REPO_ROOT", tmp_path)
    monkeypatch.setenv("MT5_LOGIN", "12345")
    monkeypatch.setenv("MT5_PASSWORD", "hunter2")
    monkeypatch.setenv("MT5_SERVER", "Broker-Demo")

    creds = get_mt5_credentials()

    assert creds.login == 12345
    assert creds.password == "hunter2"
    assert creds.server == "Broker-Demo"
    assert creds.terminal_path is None


def test_live_trading_confirmed_requires_exact_match(monkeypatch):
    monkeypatch.delenv("JARVIS_CONFIRM_LIVE", raising=False)
    assert live_trading_confirmed() is False

    monkeypatch.setenv("JARVIS_CONFIRM_LIVE", "yes")
    assert live_trading_confirmed() is False

    monkeypatch.setenv("JARVIS_CONFIRM_LIVE", "YES_I_UNDERSTAND_THE_RISK")
    assert live_trading_confirmed() is True
