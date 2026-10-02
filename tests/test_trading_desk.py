from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from assistant.daemon.scheduler import RoutineSpec, is_due
from assistant.telegram.bot import format_trade_message
from assistant.trading_desk.desk import TradingDesk
from assistant.trading_desk.limits import DeskLimits, Quote, check_new_trade, check_stop_move, daily_loss_pct
from assistant.trading_desk.state import DeskDay, DeskStateStore, roll_day
from trading_bot.trade_log import TradeLog

LIMITS = DeskLimits(
    risk_per_trade_pct=2.0, max_daily_loss_pct=6.0, max_open_positions=2, max_trades_per_day=5, min_reward_risk=1.0
)
QUOTE = Quote(bid=1.10000, ask=1.10010, point=0.00001, stops_level_points=0)
NOW = datetime(2026, 10, 1, 12, 2, tzinfo=timezone.utc)


def _check(**overrides: Any) -> str | None:
    kwargs: dict[str, Any] = dict(
        direction="buy",
        stop_loss=1.09810,
        take_profit=1.10410,
        risk_pct=2.0,
        quote=QUOTE,
        limits=LIMITS,
        symbol="EURUSDm",
        open_position_symbols=[],
        trades_today=0,
        loss_today_pct=0.0,
    )
    kwargs.update(overrides)
    return check_new_trade(**kwargs)


# ----- limits ---------------------------------------------------------------


def test_valid_buy_passes():
    assert _check() is None


def test_valid_sell_passes():
    assert _check(direction="sell", stop_loss=1.10200, take_profit=1.09600) is None


def test_daily_loss_stop_blocks_new_trades():
    assert "daily loss stop" in _check(loss_today_pct=6.0)


def test_max_trades_per_day_blocks():
    assert "trades today" in _check(trades_today=5)


def test_max_open_positions_blocks():
    assert "positions already open" in _check(open_position_symbols=["GBPUSDm", "USDJPYm"])


def test_one_position_per_symbol():
    assert "already have an open position" in _check(open_position_symbols=["EURUSDm"])


def test_risk_above_limit_rejected():
    assert "risk_pct" in _check(risk_pct=5.0)


def test_stop_on_wrong_side_rejected():
    assert "stop_loss must be below" in _check(stop_loss=1.10100)


def test_missing_take_profit_side_rejected():
    assert "stop_loss must be above" in _check(direction="sell", stop_loss=1.10200, take_profit=1.10300)


def test_stop_inside_three_spreads_rejected():
    # spread is 10 points -> minimum stop distance 30 points
    assert "minimum" in _check(stop_loss=1.09990, take_profit=1.10100)


def test_reward_below_risk_rejected():
    assert "reward:risk" in _check(stop_loss=1.09810, take_profit=1.10100)


def test_stop_can_only_tighten():
    assert check_stop_move(is_buy=True, current_stop=1.0980, new_stop=1.0970, bid=1.1000, ask=1.1001) is not None
    assert check_stop_move(is_buy=True, current_stop=1.0980, new_stop=1.0990, bid=1.1000, ask=1.1001) is None
    assert check_stop_move(is_buy=False, current_stop=1.1020, new_stop=1.1030, bid=1.1000, ask=1.1001) is not None
    assert check_stop_move(is_buy=False, current_stop=1.1020, new_stop=1.1010, bid=1.1000, ask=1.1001) is None


def test_daily_loss_pct_counts_only_losses():
    assert daily_loss_pct(1000, 940) == 6.0
    assert daily_loss_pct(1000, 1100) == 0.0
    assert daily_loss_pct(0, 500) == 0.0


def test_roll_day_resets_and_reanchors():
    old = DeskDay(day="2026-09-30", day_start_balance=1000, trades_today=3, cycles_today=9, mode="live")
    assert roll_day(old, "2026-10-01", 950, "live") == DeskDay(day="2026-10-01", day_start_balance=950, mode="live")

    unfunded = DeskDay(day="2026-10-01", day_start_balance=0, trades_today=0, cycles_today=2, mode="live")
    funded = roll_day(unfunded, "2026-10-01", 1000, "live")
    assert funded.day_start_balance == 1000 and funded.cycles_today == 2


def test_switching_paper_to_live_reanchors_the_daily_stop():
    paper = DeskDay(day="2026-10-02", day_start_balance=1000, trades_today=2, cycles_today=5, mode="paper")
    live = roll_day(paper, "2026-10-02", 11.10, "live")
    assert live == DeskDay(day="2026-10-02", day_start_balance=11.10, mode="live")
    assert daily_loss_pct(live.day_start_balance, 11.10) == 0.0


def test_hourly_routine_runs_once_per_hour_after_minute():
    spec = RoutineSpec("desk", lambda ctx: None, hourly_at_minute=2)
    assert is_due(spec, None, datetime(2026, 10, 1, 12, 1)) is False
    assert is_due(spec, None, datetime(2026, 10, 1, 12, 2)) is True
    assert is_due(spec, datetime(2026, 10, 1, 12, 2), datetime(2026, 10, 1, 12, 40)) is False
    assert is_due(spec, datetime(2026, 10, 1, 12, 2), datetime(2026, 10, 1, 13, 3)) is True


# ----- desk cycle (fake broker + fake Claude) ------------------------------


@dataclass
class FakeAccount:
    balance: float = 1000.0
    equity: float = 1000.0
    margin_free: float = 1000.0
    currency: str = "USD"
    trade_allowed: bool = True
    margin_level: float = 0.0


@dataclass
class FakeSymbolInfo:
    digits: int = 5
    point: float = 0.00001
    trade_stops_level: int = 0
    trade_tick_value: float = 1.0
    trade_tick_size: float = 0.00001
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01


@dataclass
class FakeTick:
    bid: float
    ask: float
    time: int


@dataclass
class FakeOrderResult:
    success: bool = True
    retcode: int = 10009
    comment: str = "done"
    order_id: int = 777
    price: float = 1.10010


class FakeAdapter:
    def __init__(self, tick_time: int | None = None, account: FakeAccount | None = None, closed=None):
        self.tick_time = tick_time if tick_time is not None else int(NOW.timestamp())
        self.account = account or FakeAccount()
        self.closed = closed or []
        self.orders: list[dict] = []
        self.connected = False

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def get_account_info(self):
        return self.account

    def resolve_symbol(self, base):
        return {"EURUSD": "EURUSDm"}.get(base)

    def get_tick(self, symbol):
        return FakeTick(bid=1.10000, ask=1.10010, time=self.tick_time)

    def get_symbol_info(self, symbol):
        return FakeSymbolInfo()

    def get_rates(self, symbol, timeframe, count):
        close = 1.10 + np.cumsum(np.full(count, 0.0001))
        return pd.DataFrame(
            {
                "time": pd.date_range(end=NOW, periods=count, freq="h"),
                "open": close - 0.0001,
                "high": close + 0.0005,
                "low": close - 0.0005,
                "close": close,
                "tick_volume": np.full(count, 100),
                "spread": np.full(count, 10),
            }
        )

    def closed_results(self, date_from, date_to, magic):
        return list(self.closed)

    def get_open_positions(self, symbol=None):
        return []

    def calc_margin(self, symbol, is_buy, volume, price):
        return 10.0

    def place_market_order(self, **kwargs):
        self.orders.append(kwargs)
        return FakeOrderResult()


@dataclass
class FakeUsage:
    input_tokens: int = 1000
    output_tokens: int = 200
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class FakeText:
    text: str
    type: str = "text"


@dataclass
class FakeToolUse:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class FakeResponse:
    content: list
    stop_reason: str
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeBetaMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.beta = type("Beta", (), {})()
        self.beta.messages = FakeBetaMessages(responses)


def _cfg(live: bool = False, firm: dict | None = None) -> dict:
    cfg = {
        "assistant": {"trade_log_path": "state/trades.jsonl"},
        "trading_desk": {
            "enabled": True,
            "live": live,
            "symbols": ["EURUSD", "XAUUSD"],
            "session_hours_utc": [7, 20],
            "max_cycles_per_day": 16,
            "paper_equity": 1000.0,
            "magic_number": 20261001,
            "model": "claude-opus-5-5",
            "effort": "low",
            "limits": {
                "risk_per_trade_pct": 2.0,
                "max_daily_loss_pct": 6.0,
                "max_open_positions": 2,
                "max_trades_per_day": 5,
                "min_reward_risk": 1.0,
            },
        },
    }
    if firm is not None:
        cfg["trading_desk"]["firm"] = firm
    return cfg


BUY = {
    "symbol": "EURUSD",
    "direction": "buy",
    "stop_loss": 1.09810,
    "take_profit": 1.10410,
    "reasoning": "H4 and H1 trend up, buying the pullback.",
}


def _desk(tmp_path, responses, live=False, adapter=None, firm=None):
    client = FakeClient(responses)
    desk = TradingDesk(tmp_path, _cfg(live, firm), client, adapter=adapter or FakeAdapter())
    return desk, client


def test_paper_cycle_places_code_sized_trade_and_journals(tmp_path):
    desk, client = _desk(
        tmp_path,
        [
            FakeResponse([FakeToolUse("t1", "place_trade", BUY)], "tool_use"),
            FakeResponse([FakeText("Bought EURUSD on the H1 pullback.")], "end_turn"),
        ],
    )

    summary = desk.run_cycle(NOW)

    assert summary == "Bought EURUSD on the H1 pullback."
    [trade] = TradeLog(tmp_path / "state" / "trades.jsonl").tail()
    # 2% of 1000 paper equity = 20; stop 0.002 away = 200/lot -> 0.10 lots
    assert trade["lots"] == 0.1
    assert trade["mode"] == "paper" and trade["status"] == "simulated"
    assert trade["extra"]["source"] == "alex_desk"
    assert desk.adapter.orders == []  # paper: nothing sent to the broker
    assert DeskStateStore(tmp_path / "state").day().trades_today == 1

    call = client.beta.messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["fallbacks"] == "default"
    assert "PAPER" in call["messages"][0]["content"]
    journal = [json.loads(line) for line in (tmp_path / "logs" / "trading_desk.jsonl").read_text().splitlines()]
    assert journal[-1]["summary"] == summary and journal[-1]["est_cost_usd"] > 0


def test_limit_breaking_trade_is_rejected_and_not_logged(tmp_path):
    bad = {**BUY, "take_profit": 1.10100}  # reward < risk
    desk, client = _desk(
        tmp_path,
        [
            FakeResponse([FakeToolUse("t1", "place_trade", bad)], "tool_use"),
            FakeResponse([FakeText("Stood aside.")], "end_turn"),
        ],
    )

    desk.run_cycle(NOW)

    tool_result = client.beta.messages.calls[1]["messages"][-1]["content"][0]["content"]
    assert tool_result.startswith("REJECTED")
    assert TradeLog(tmp_path / "state" / "trades.jsonl").tail() == []


def test_live_config_without_env_confirmation_stays_paper(tmp_path, monkeypatch):
    monkeypatch.delenv("JARVIS_CONFIRM_LIVE", raising=False)
    desk, _ = _desk(
        tmp_path,
        [FakeResponse([FakeToolUse("t1", "place_trade", BUY)], "tool_use"), FakeResponse([FakeText("ok")], "end_turn")],
        live=True,
    )
    desk.run_cycle(NOW)
    assert desk.adapter.orders == []


def test_live_with_both_gates_sends_order_with_desk_magic(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_CONFIRM_LIVE", "YES_I_UNDERSTAND_THE_RISK")
    desk, _ = _desk(
        tmp_path,
        [FakeResponse([FakeToolUse("t1", "place_trade", BUY)], "tool_use"), FakeResponse([FakeText("ok")], "end_turn")],
        live=True,
    )
    desk.run_cycle(NOW)
    [order] = desk.adapter.orders
    assert order["magic"] == 20261001
    assert order["volume"] == 0.1  # 2% of live equity 1000
    assert order["sl"] == 1.09810 and order["tp"] == 1.10410


def test_paused_desk_skips_without_calling_claude(tmp_path):
    desk, client = _desk(tmp_path, [])
    DeskStateStore(tmp_path / "state").update_control(paused=True)
    assert "paused" in desk.run_cycle(NOW)
    assert client.beta.messages.calls == []


def test_outside_session_skips(tmp_path):
    desk, client = _desk(tmp_path, [])
    assert "outside session" in desk.run_cycle(datetime(2026, 10, 1, 22, 2, tzinfo=timezone.utc))
    assert client.beta.messages.calls == []


def test_market_closed_skips_without_calling_claude(tmp_path):
    stale = FakeAdapter(tick_time=int(NOW.timestamp()) - 2 * 86400)
    desk, client = _desk(tmp_path, [], adapter=stale)
    assert "market closed" in desk.run_cycle(NOW)
    assert client.beta.messages.calls == []


def test_desk_trade_telegram_message_shows_entry_stop_target_and_reason():
    msg = format_trade_message(
        {
            "symbol": "EURUSDc", "action": "BUY", "mode": "live", "lots": 0.1, "price": 1.1001,
            "sl": 1.0981, "tp": 1.1041, "reason": "Trend up.", "status": "filled",
            "extra": {"source": "alex_desk", "risk_amount": 20.0, "risk_pct": 2.0, "reward_risk": 2.0, "currency": "USC"},
        }
    )
    assert "entry 1.1001" in msg and "SL 1.0981" in msg and "TP 1.1041" in msg
    assert "LIVE" in msg and "why: Trend up." in msg


# ----- the firm: Quant -> Analysts -> Strategy -> Risk -> Alex (CEO) -------

from assistant.trading_desk.firm import parse_count  # noqa: E402

FIRM_ON = {"enabled": True, "max_followups_per_cycle": 2}
CHECK = {k: v for k, v in BUY.items() if k != "reasoning"}


def _role(call: dict) -> str:
    system = call["system"]
    for marker, role in (
        ("You are the Director of Intelligence", "intelligence"),
        ("You are the Analyst team", "analyst"),
        ("You are the Market Strategy team", "strategist"),
        ("You are department 9, Risk Intelligence", "risk"),
        ("You are Alex, CEO", "ceo"),
    ):
        if marker in system:
            return role
    return "?"


def _tool_names(call: dict) -> set[str]:
    return {t["name"] for t in call["tools"]}


def _journal(tmp_path) -> dict:
    return json.loads((tmp_path / "logs" / "trading_desk.jsonl").read_text().splitlines()[-1])


def test_firm_full_chain_reports_up_and_only_ceo_trades(tmp_path):
    desk, client = _desk(
        tmp_path,
        [
            FakeResponse([FakeText("EURUSD aligned up on H4/H1/M15.\nCANDIDATES: 1")], "end_turn"),
            FakeResponse([FakeText("EURUSD long: STRONG, levels verified.\nVETTED: 1")], "end_turn"),
            FakeResponse([FakeText("Plan: buy EURUSD at market, SL 1.09810, TP 1.10410.")], "end_turn"),
            FakeResponse([FakeToolUse("r1", "check_trade", CHECK)], "tool_use"),
            FakeResponse([FakeText("Within limits.\nVERDICT: EURUSDm buy APPROVE 12")], "end_turn"),
            FakeResponse([FakeToolUse("c1", "place_trade", BUY)], "tool_use"),
            FakeResponse([FakeText("Bought EURUSD per the teams' plan.")], "end_turn"),
        ],
        firm=FIRM_ON,
    )

    summary = desk.run_cycle(NOW)

    calls = client.beta.messages.calls
    assert [_role(c) for c in calls] == ["intelligence", "analyst", "strategist", "risk", "risk", "ceo", "ceo"]
    # Division of labour is enforced by tool access: only the CEO can trade.
    for call in calls:
        trading = _tool_names(call) & {"place_trade", "close_position", "move_stop_loss"}
        assert bool(trading) == (_role(call) == "ceo")
    assert _tool_names(calls[3]) == {"check_trade"}
    # Each team sees the report before it; the CEO sees them all.
    assert "EURUSD aligned up" in calls[1]["messages"][0]["content"]
    assert "STRONG, levels verified" in calls[2]["messages"][0]["content"]
    assert "Plan: buy EURUSD" in calls[3]["messages"][0]["content"]
    ceo_input = calls[5]["messages"][0]["content"]
    assert all(s in ceo_input for s in ("CANDIDATES: 1", "VETTED: 1", "Plan: buy", "VERDICT: EURUSDm buy APPROVE"))
    # Risk's dry run used the real sizing code and placed nothing.
    assert "WOULD BE ACCEPTED: BUY 0.1 EURUSDm" in calls[4]["messages"][-1]["content"][0]["content"]

    assert summary == "Bought EURUSD per the teams' plan."
    assert len(TradeLog(tmp_path / "state" / "trades.jsonl").tail()) == 1
    entry = _journal(tmp_path)
    assert set(entry["reports"]) == {"intelligence", "analyst", "strategist", "risk"}
    assert set(entry["cost_by_role"]) == {"intelligence", "analyst", "strategist", "risk", "ceo"}
    assert entry["est_cost_usd"] > 0


def test_firm_quiet_hour_skips_middle_teams(tmp_path):
    desk, client = _desk(
        tmp_path,
        [
            FakeResponse([FakeText("No alignment anywhere.\nCANDIDATES: 0")], "end_turn"),
            FakeResponse([FakeText("Agree with the quants; standing aside.")], "end_turn"),
        ],
        firm=FIRM_ON,
    )

    desk.run_cycle(NOW)

    assert [_role(c) for c in client.beta.messages.calls] == ["intelligence", "ceo"]
    reports = _journal(tmp_path)["reports"]
    assert reports["analyst"].startswith("(skipped") and reports["risk"].startswith("(skipped")


def test_ceo_can_send_a_team_back(tmp_path):
    desk, client = _desk(
        tmp_path,
        [
            FakeResponse([FakeText("Nothing.\nCANDIDATES: 0")], "end_turn"),
            FakeResponse(
                [FakeToolUse("f1", "request_followup", {"team": "intelligence", "instructions": "Recheck USDJPY H4 swing low."})],
                "tool_use",
            ),
            FakeResponse([FakeText("Rechecked USDJPY: still no setup.\nCANDIDATES: 0")], "end_turn"),
            FakeResponse([FakeText("Quants confirmed on recheck; no trade.")], "end_turn"),
        ],
        firm=FIRM_ON,
    )

    desk.run_cycle(NOW)

    calls = client.beta.messages.calls
    assert [_role(c) for c in calls] == ["intelligence", "ceo", "intelligence", "ceo"]
    assert "Recheck USDJPY H4 swing low." in calls[2]["messages"][0]["content"]
    assert "Revised Intelligence Departments report" in calls[3]["messages"][-1]["content"][0]["content"]
    assert _journal(tmp_path)["followups"] == [{"team": "intelligence", "instructions": "Recheck USDJPY H4 swing low."}]


def test_followups_are_capped_per_cycle(tmp_path):
    desk, client = _desk(
        tmp_path,
        [
            FakeResponse([FakeText("CANDIDATES: 0")], "end_turn"),
            FakeResponse(
                [
                    FakeToolUse("f1", "request_followup", {"team": "intelligence", "instructions": "again"}),
                    FakeToolUse("f2", "request_followup", {"team": "analyst", "instructions": "run anyway"}),
                ],
                "tool_use",
            ),
            FakeResponse([FakeText("CANDIDATES: 0")], "end_turn"),  # the one allowed follow-up
            FakeResponse([FakeText("Done.")], "end_turn"),
        ],
        firm={"enabled": True, "max_followups_per_cycle": 1},
    )

    desk.run_cycle(NOW)

    results = client.beta.messages.calls[-1]["messages"][-1]["content"]
    assert "Revised Intelligence Departments report" in results[0]["content"]
    assert results[1]["content"].startswith("Error: follow-up limit reached")


def test_parse_count_reads_the_team_footer():
    assert parse_count("blah\nCANDIDATES: 2", "CANDIDATES") == 2
    assert parse_count("**CANDIDATES:** 0", "CANDIDATES") == 0
    assert parse_count("VETTED: 1\n...\nVETTED: 0", "VETTED") == 0
    assert parse_count("no footer at all", "CANDIDATES") is None


def test_only_one_decision_per_hour(tmp_path):
    desk, client = _desk(tmp_path, [FakeResponse([FakeText("Stood aside.")], "end_turn")])
    desk.run_cycle(NOW)  # 12:02
    assert "already decided this hour" in desk.run_cycle(NOW.replace(minute=50))
    assert len(client.beta.messages.calls) == 1


# ----- nine departments: Risk Intelligence gate, intel, briefing -----------

from assistant.trading_desk.briefing import DailyBriefing  # noqa: E402
from assistant.trading_desk.firm import department_findings, parse_verdicts  # noqa: E402
from assistant.trading_desk.intel import barrier_outcomes, base_rates  # noqa: E402
from assistant.trading_desk.limits import currency_concentration, losing_streak, pre_cycle_directive  # noqa: E402


def _directive(**overrides):
    kwargs = dict(
        limits=LIMITS, peak_equity=100.0, equity=100.0, closed_results_today=[],
        margin_level_pct=0.0, has_positions=False, drawdown_halted=False,
    )
    kwargs.update(overrides)
    return pre_cycle_directive(**kwargs)


def test_risk_gate_normal_by_default():
    d = _directive()
    assert d.state == "NORMAL" and d.max_risk_pct == LIMITS.risk_per_trade_pct


def test_risk_gate_halts_on_peak_drawdown_and_stays_halted():
    assert _directive(equity=69.0).state == "HALT_NEW"          # 31% below peak
    assert _directive(equity=95.0, drawdown_halted=True).state == "HALT_NEW"  # until re-armed


def test_risk_gate_losing_streak_reduces_then_halts():
    reduced = _directive(closed_results_today=[5.0, -1.0, -1.0, -1.0])
    assert reduced.state == "REDUCED" and reduced.max_risk_pct == LIMITS.risk_per_trade_pct / 2
    assert _directive(closed_results_today=[-1.0] * 5).state == "HALT_NEW"
    assert losing_streak([-1.0, -1.0, 2.0]) == 0


def test_risk_gate_low_margin_level_blocks_new_trades():
    assert _directive(margin_level_pct=300.0, has_positions=True).state == "HALT_NEW"
    assert _directive(margin_level_pct=300.0, has_positions=False).state == "NORMAL"


def test_currency_concentration_blocks_the_same_bet_twice():
    assert currency_concentration("GBPUSDm", True, [("EURUSDm", True)]) is not None   # short USD x2
    assert currency_concentration("XAUUSDm", True, [("EURUSDm", True)]) is not None   # short USD x2
    assert currency_concentration("USDJPYm", False, [("EURUSDm", True)]) is not None  # short USD x2
    assert currency_concentration("GBPUSDm", False, [("EURUSDm", True)]) is None      # EUR vs GBP spread
    assert currency_concentration("EURUSDm", True, []) is None


def test_parse_verdicts_and_department_findings():
    verdicts = parse_verdicts("blah\n**VERDICT:** EURUSDm buy APPROVE 8\nVERDICT: gbpusd sell REJECT\nVERDICT: NONE")
    assert verdicts == {("EURUSD", "buy"): ("APPROVE", 8.0), ("GBPUSD", "sell"): ("REJECT", None)}
    findings = department_findings("DEPARTMENT FINDINGS\n1. Macroeconomic: Fed on hold.\n**7. Order flow:** balanced\nCANDIDATES: 0")
    assert findings == ["1. Macroeconomic: Fed on hold.", "7. Order flow: balanced"]


def test_quant_base_rates_reflect_trend():
    n = 3000
    close = 1.0 + np.cumsum(np.full(n, 0.0002))  # steady uptrend
    df = pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=n, freq="h"),
        "open": close - 0.0001, "high": close + 0.0003, "low": close - 0.0003, "close": close,
    })
    long_win, short_win = barrier_outcomes(df)
    assert long_win[:-30].mean() > 0.9 and short_win[:-30].mean() < 0.05
    rates = base_rates(df)
    assert rates["current_state"] == "up" and rates["up"]["long_first"] > 0.9


def _firm_ceo_places(tmp_path, risk_text):
    return _desk(
        tmp_path,
        [
            FakeResponse([FakeText("CANDIDATES: 1")], "end_turn"),
            FakeResponse([FakeText("VETTED: 1")], "end_turn"),
            FakeResponse([FakeText("Plan: buy EURUSD.")], "end_turn"),
            FakeResponse([FakeText(risk_text)], "end_turn"),
            FakeResponse([FakeToolUse("c1", "place_trade", BUY)], "tool_use"),
            FakeResponse([FakeText("CONSENSUS VERDICT: no trade.")], "end_turn"),
        ],
        firm=FIRM_ON,
    )


def test_ceo_cannot_trade_without_risk_approval(tmp_path):
    desk, client = _firm_ceo_places(tmp_path, "Too risky.\nVERDICT: EURUSDm buy REJECT")
    desk.run_cycle(NOW)
    result = client.beta.messages.calls[-1]["messages"][-1]["content"][0]["content"]
    assert result.startswith("REJECTED: Risk Intelligence has not approved buy EURUSDm")
    assert TradeLog(tmp_path / "state" / "trades.jsonl").tail() == []


def test_ceo_cannot_exceed_risk_approved_size(tmp_path):
    desk, client = _firm_ceo_places(tmp_path, "OK at lower size.\nVERDICT: EURUSDm buy APPROVE 1")
    desk.run_cycle(NOW)  # CEO's place_trade defaults to the full 2% limit
    result = client.beta.messages.calls[-1]["messages"][-1]["content"][0]["content"]
    assert "approved at most 1% risk" in result


def test_trade_message_carries_department_findings(tmp_path):
    desk, _ = _firm_ceo_places(tmp_path, "VERDICT: EURUSDm buy APPROVE 12")
    desk.run_cycle(NOW)
    # The intelligence report had no numbered findings here; the record still carries the field.
    [trade] = TradeLog(tmp_path / "state" / "trades.jsonl").tail()
    assert trade["extra"]["departments"] == []
    msg = format_trade_message({**trade, "extra": {**trade["extra"], "departments": ["5. Technical: aligned up"]}})
    assert "Departments:" in msg and "5. Technical: aligned up" in msg


def test_wide_spread_blocks_new_entries(tmp_path):
    desk, client = _desk(
        tmp_path,
        [FakeResponse([FakeToolUse("t1", "place_trade", BUY)], "tool_use"), FakeResponse([FakeText("ok")], "end_turn")],
    )
    real = desk.adapter.get_rates

    def thin_history(symbol, timeframe, count):
        df = real(symbol, timeframe, count)
        if timeframe == "M15":
            df["spread"] = 2  # typical 2 points; live spread is 10 -> 5x
        return df

    desk.adapter.get_rates = thin_history
    desk.run_cycle(NOW)
    result = client.beta.messages.calls[-1]["messages"][-1]["content"][0]["content"]
    assert "spread is 5.0x normal" in result


def test_live_drawdown_halt_skips_cycle_without_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_CONFIRM_LIVE", "YES_I_UNDERSTAND_THE_RISK")
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "trading_desk_risk.json").write_text(json.dumps({"peak_equity": 2000.0}))
    desk, client = _desk(tmp_path, [], live=True)  # equity 1000 vs peak 2000 = 50% drawdown
    assert "Risk Intelligence halt" in desk.run_cycle(NOW)
    assert client.beta.messages.calls == []
    assert json.loads((tmp_path / "state" / "trading_desk_risk.json").read_text())["drawdown_halted"] is True


def test_daily_briefing_falls_back_without_web_and_caches(tmp_path):
    class Messages:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("tools"):
                raise RuntimeError("web search not enabled for this organization")
            return FakeResponse([FakeText("1. MACROECONOMIC: prices only")], "end_turn")

    client = type("C", (), {})()
    client.beta = type("B", (), {})()
    client.beta.messages = Messages()
    cfg = {"model": "claude-opus-5-5", "briefing": {"enabled": True}}
    briefing = DailyBriefing(tmp_path, cfg, client)

    text, cost = briefing.get(NOW, "snapshot")
    assert "PRICES ONLY" in text and cost > 0
    assert len(client.beta.messages.calls) == 2  # web attempt, then fallback
    text2, cost2 = briefing.get(NOW.replace(hour=15), "snapshot")
    assert cost2 == 0 and len(client.beta.messages.calls) == 2  # cached for the day
