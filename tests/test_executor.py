from __future__ import annotations

from dataclasses import dataclass

import pytest

from trading_bot.executor import Executor
from trading_bot.strategy import Signal
from trading_bot.trade_log import TradeLog


@dataclass
class FakeAccount:
    equity: float
    balance: float = 10_000.0
    margin: float = 0.0


@dataclass
class FakeSymbolInfo:
    trade_tick_value: float = 1.0
    trade_tick_size: float = 0.00001
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01


@dataclass
class FakePosition:
    ticket: int = 1
    symbol: str = "EURUSD"
    volume: float = 0.1
    type: int = 0
    magic: int = 0


@dataclass
class FakeOrderResult:
    success: bool = True
    retcode: int = 10009
    comment: str = "ok"
    order_id: int = 42
    price: float = 1.10001


class FakeAdapter:
    def __init__(self, equity: float = 10_000.0, open_positions: list | None = None):
        self._equity = equity
        self._open_positions = open_positions or []
        self.placed_orders: list[dict] = []

    def get_account_info(self):
        return FakeAccount(equity=self._equity)

    def get_open_positions(self, symbol=None):
        return self._open_positions

    def get_symbol_info(self, symbol):
        return FakeSymbolInfo()

    def place_market_order(self, **kwargs):
        self.placed_orders.append(kwargs)
        return FakeOrderResult()


TRADING_CFG = {
    "symbol": "EURUSD",
    "atr_sl_multiplier": 1.5,
    "atr_tp_multiplier": 3.0,
    "risk_per_trade_pct": 1.0,
    "max_daily_loss_pct": 3.0,
    "max_open_positions": 1,
}

EXECUTION_CFG_PAPER = {
    "live_trading": False,
    "magic_number": 123,
    "deviation_points": 20,
    "order_comment": "test",
}

EXECUTION_CFG_LIVE = {**EXECUTION_CFG_PAPER, "live_trading": True}


def _signal(action="BUY"):
    return Signal(action=action, reason="test signal", close_price=1.1000, atr_value=0.0010)


def test_hold_signal_is_a_noop(tmp_path):
    adapter = FakeAdapter()
    trade_log = TradeLog(tmp_path / "trades.jsonl")
    executor = Executor(adapter, trade_log, TRADING_CFG, EXECUTION_CFG_PAPER)

    outcome = executor.handle_signal(Signal("HOLD", "no cross", 1.1, 0.001))

    assert outcome.startswith("HOLD:")
    assert trade_log.tail(10) == []


def test_paper_buy_signal_logs_simulated_trade(tmp_path):
    adapter = FakeAdapter()
    trade_log = TradeLog(tmp_path / "trades.jsonl")
    executor = Executor(adapter, trade_log, TRADING_CFG, EXECUTION_CFG_PAPER)

    outcome = executor.handle_signal(_signal("BUY"))

    assert outcome.startswith("PAPER BUY")
    records = trade_log.tail(10)
    assert len(records) == 1
    assert records[0]["mode"] == "paper"
    assert records[0]["status"] == "simulated"
    assert adapter.placed_orders == []  # never touches the broker in paper mode


def test_max_open_positions_blocks_new_trade(tmp_path):
    adapter = FakeAdapter(open_positions=[FakePosition()])
    trade_log = TradeLog(tmp_path / "trades.jsonl")
    executor = Executor(adapter, trade_log, TRADING_CFG, EXECUTION_CFG_PAPER)

    outcome = executor.handle_signal(_signal("BUY"))

    assert outcome.startswith("BLOCKED: max_open_positions")
    assert trade_log.tail(10) == []


def test_daily_drawdown_guard_blocks_after_cap_exceeded(tmp_path):
    adapter = FakeAdapter(equity=10_000.0)
    trade_log = TradeLog(tmp_path / "trades.jsonl")
    executor = Executor(adapter, trade_log, TRADING_CFG, EXECUTION_CFG_PAPER)

    executor.handle_signal(_signal("BUY"))  # establishes today's baseline equity

    adapter._equity = 9_500.0  # -5%, exceeds 3% cap
    outcome = executor.handle_signal(_signal("BUY"))

    assert outcome.startswith("BLOCKED: daily loss")


def test_live_trading_without_env_confirmation_falls_back_to_paper(tmp_path, monkeypatch):
    monkeypatch.delenv("JARVIS_CONFIRM_LIVE", raising=False)
    adapter = FakeAdapter()
    trade_log = TradeLog(tmp_path / "trades.jsonl")
    executor = Executor(adapter, trade_log, TRADING_CFG, EXECUTION_CFG_LIVE)

    outcome = executor.handle_signal(_signal("BUY"))

    assert outcome.startswith("PAPER BUY")
    assert adapter.placed_orders == []


def test_live_trading_with_env_confirmation_places_real_order(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_CONFIRM_LIVE", "YES_I_UNDERSTAND_THE_RISK")
    adapter = FakeAdapter()
    trade_log = TradeLog(tmp_path / "trades.jsonl")
    executor = Executor(adapter, trade_log, TRADING_CFG, EXECUTION_CFG_LIVE)

    outcome = executor.handle_signal(_signal("BUY"))

    assert outcome.startswith("LIVE BUY")
    assert len(adapter.placed_orders) == 1
    records = trade_log.tail(10)
    assert records[0]["mode"] == "live"
    assert records[0]["status"] == "filled"
