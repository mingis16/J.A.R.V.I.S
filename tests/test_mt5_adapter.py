from __future__ import annotations

from types import SimpleNamespace

from trading_bot import mt5_adapter
from trading_bot.config import MT5Credentials
from trading_bot.mt5_adapter import MT5Adapter

CREDS = MT5Credentials(login=123, password="pw", server="srv", terminal_path=None)


def _fake_mt5(monkeypatch, init_results, account_login=None):
    calls = {"initialize": [], "shutdown": 0}
    results = list(init_results)

    def initialize(*args, **kwargs):
        calls["initialize"].append(kwargs)
        return results.pop(0)

    monkeypatch.setattr(mt5_adapter.mt5, "initialize", initialize)
    monkeypatch.setattr(mt5_adapter.mt5, "shutdown", lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1))
    monkeypatch.setattr(mt5_adapter.mt5, "last_error", lambda: (-10005, "IPC timeout"))
    monkeypatch.setattr(
        mt5_adapter.mt5, "account_info", lambda: SimpleNamespace(login=account_login) if account_login else None
    )
    monkeypatch.setattr(mt5_adapter.time, "sleep", lambda s: None)
    return calls


def test_connect_reuses_terminal_already_on_the_account(monkeypatch):
    calls = _fake_mt5(monkeypatch, [True], account_login=123)
    MT5Adapter(CREDS).connect()
    assert calls["initialize"] == [{}]  # no credentials -> no forced re-login


def test_connect_retries_once_on_ipc_timeout(monkeypatch):
    # attach fails, first login times out, retry succeeds
    calls = _fake_mt5(monkeypatch, [False, False, True])
    MT5Adapter(CREDS).connect()
    assert len(calls["initialize"]) == 3 and calls["initialize"][-1]["login"] == 123


def test_connect_gives_up_after_one_retry(monkeypatch):
    _fake_mt5(monkeypatch, [False, False, False])
    try:
        MT5Adapter(CREDS).connect()
    except RuntimeError as exc:
        assert "IPC timeout" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
