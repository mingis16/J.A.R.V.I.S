"""Grades signals from a date range against what actually happened next in
the market — the answer to "how accurate were the signals."

Reads:
- state/trades.jsonl              (trading_bot paper/live trades — the volume source)
- signal_engine/live_checks.jsonl (calibrated engine's hourly checks — signal hits only)

For each, pulls subsequent price bars from MT5 and resolves TP/SL/timeout/
still-open using the same conservative ambiguous-bar-as-loss convention as
signal_engine/labeling.py, then reports hit rate AND mean R side by side —
never hit rate alone, per this project's own honesty rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from trading_bot.mt5_adapter import MT5Adapter
from trading_bot.trade_log import TradeLog

from signal_engine.config import SignalEngineConfig
from signal_engine.live_checks import LiveCheckLog

MAX_LOOKBACK_BARS = 20000  # matches the MT5 demo server's confirmed ceiling


@dataclass
class Graded:
    source: str
    pair: str
    direction: str
    entry_time: datetime
    entry: float
    stop: float
    take_profit: float
    outcome: str  # tp | sl | ambiguous | timeout | still_open
    r_multiple: float | None


def _is_buy(direction: str) -> bool:
    return direction.strip().lower() in ("long", "buy")


def resolve_from_bars(
    after: pd.DataFrame,
    direction: str,
    entry: float,
    stop: float,
    take_profit: float,
    horizon_bars: int | None,
) -> tuple[str, float | None]:
    """Pure resolution logic over bars strictly after entry_time — no I/O,
    unit-testable without MT5. Same conservative ambiguous-as-loss
    convention as signal_engine/labeling.py."""
    if len(after) == 0:
        return "still_open", None

    is_buy = _is_buy(direction)
    sl_dist = abs(entry - stop)
    if sl_dist <= 0:
        return "still_open", None

    bars_checked = 0
    for _, bar in after.iterrows():
        bars_checked += 1
        if is_buy:
            hit_tp, hit_sl = bar["high"] >= take_profit, bar["low"] <= stop
        else:
            hit_tp, hit_sl = bar["low"] <= take_profit, bar["high"] >= stop

        if hit_tp and hit_sl:
            return "ambiguous", -1.0
        if hit_tp:
            return "tp", abs(take_profit - entry) / sl_dist
        if hit_sl:
            return "sl", -1.0
        if horizon_bars and bars_checked >= horizon_bars:
            signed = 1 if is_buy else -1
            return "timeout", (bar["close"] - entry) / sl_dist * signed

    return "still_open", None


def resolve_outcome(
    adapter: MT5Adapter,
    pair: str,
    timeframe: str,
    entry_time: datetime,
    direction: str,
    entry: float,
    stop: float,
    take_profit: float,
    horizon_bars: int | None,
) -> tuple[str, float | None]:
    df = adapter.get_rates(pair, timeframe, MAX_LOOKBACK_BARS)
    df = df.sort_values("time").reset_index(drop=True)
    after = df[df["time"] > entry_time]
    return resolve_from_bars(after, direction, entry, stop, take_profit, horizon_bars)


def gather_trading_bot_entries(repo_root: Path, cfg: dict, since: datetime) -> list[dict]:
    trade_log = TradeLog(repo_root / cfg["assistant"]["trade_log_path"])
    entries = trade_log.tail(10**6)
    out = []
    for e in entries:
        ts = datetime.fromisoformat(e["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= since and e.get("sl") is not None and e.get("tp") is not None:
            out.append(e)
    return out


def gather_signal_engine_hits(se_cfg: SignalEngineConfig, since: datetime) -> list[dict]:
    log = LiveCheckLog(se_cfg.report_dir / "live_checks.jsonl")
    out = []
    for check in log.all():
        ts = datetime.fromisoformat(check["checked_at_utc"])
        if ts >= since and check.get("has_signal") and check.get("signal"):
            out.append(check["signal"])
    return out


def grade_all(adapter: MT5Adapter, repo_root: Path, cfg: dict, since: datetime) -> list[Graded]:
    graded: list[Graded] = []

    for e in gather_trading_bot_entries(repo_root, cfg, since):
        ts = datetime.fromisoformat(e["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        outcome, r = resolve_outcome(
            adapter, e["symbol"], cfg["trading"]["timeframe"], ts, e["action"], e["price"], e["sl"], e["tp"],
            horizon_bars=None,
        )
        graded.append(Graded("trading_bot", e["symbol"], e["action"], ts, e["price"], e["sl"], e["tp"], outcome, r))

    if "signal_engine" in cfg:
        se_cfg = SignalEngineConfig.from_yaml(cfg, repo_root)
        for s in gather_signal_engine_hits(se_cfg, since):
            ts = datetime.fromisoformat(s["timestamp_utc"])
            outcome, r = resolve_outcome(
                adapter, s["pair"], se_cfg.timeframe, ts, s["direction"], s["entry"], s["stop"], s["take_profit"],
                horizon_bars=se_cfg.horizon_bars,
            )
            graded.append(
                Graded("signal_engine", s["pair"], s["direction"], ts, s["entry"], s["stop"], s["take_profit"], outcome, r)
            )

    return graded


def format_report(graded: list[Graded]) -> str:
    lines: list[str] = []
    resolved = [g for g in graded if g.outcome in ("tp", "sl", "ambiguous", "timeout")]
    still_open = [g for g in graded if g.outcome == "still_open"]

    lines.append(f"\nTotal signals in range: {len(graded)}  (resolved: {len(resolved)}, still open: {len(still_open)})\n")

    if not resolved:
        lines.append("Nothing resolved yet — too early to say anything about accuracy.")
        return "\n".join(lines)

    if len(resolved) < 20:
        lines.append(
            f"NOTE: only {len(resolved)} resolved signals. This is a small sample — treat any "
            "hit rate/expectancy below as a rough signal, not a verdict.\n"
        )

    hits = sum(1 for g in resolved if g.outcome == "tp")
    hit_rate = hits / len(resolved)
    mean_r = sum(g.r_multiple for g in resolved if g.r_multiple is not None) / len(resolved)
    lines.append(f"Hit rate: {hit_rate:.1%}    Mean R: {mean_r:.3f}   <- both matter; mean R is the honest headline\n")

    lines.append("By source:")
    for source in sorted({g.source for g in resolved}):
        sub = [g for g in resolved if g.source == source]
        sub_hits = sum(1 for g in sub if g.outcome == "tp")
        sub_mean_r = sum(g.r_multiple for g in sub if g.r_multiple is not None) / len(sub)
        lines.append(f"  {source}: {len(sub)} resolved, hit rate {sub_hits / len(sub):.1%}, mean R {sub_mean_r:.3f}")

    lines.append("\nBy day:")
    by_day: dict[str, list[Graded]] = {}
    for g in resolved:
        by_day.setdefault(g.entry_time.date().isoformat(), []).append(g)
    for day in sorted(by_day):
        day_trades = by_day[day]
        day_hits = sum(1 for g in day_trades if g.outcome == "tp")
        day_mean_r = sum(g.r_multiple for g in day_trades if g.r_multiple is not None) / len(day_trades)
        lines.append(f"  {day}: {len(day_trades)} trades, hit rate {day_hits / len(day_trades):.1%}, mean R {day_mean_r:.3f}")

    outcome_counts = {o: sum(1 for g in resolved if g.outcome == o) for o in ("tp", "sl", "ambiguous", "timeout")}
    lines.append(f"\nOutcome breakdown: {outcome_counts}")
    if still_open:
        lines.append(f"\n{len(still_open)} signal(s) still open (entered too recently to resolve) — re-run later.")

    return "\n".join(lines)
