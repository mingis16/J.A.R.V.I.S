"""Code-computed inputs for the intelligence departments.

These departments do arithmetic, so they're code, not LLM calls — free to
run every cycle and impossible to "hallucinate":

- Quantitative (4): base rates from ~8 years of H1 history — how often price
  reached +1 ATR before -1 ATR within 24 hours in each trend state.
- Sentiment / risk appetite (6): cross-asset moves (equity indices, USDJPY,
  gold, BTC, dollar index, oil) scored risk-on / risk-off.
- Order flow (7): a PROXY. Retail FX on MT5 has no order book or real volume
  (market_book_add fails on this broker; real_volume is 0), so this uses
  tick volume on up vs down bars and where bars close in their range.
- Liquidity (8): current spread against what's normal for this hour, and
  tick activity against normal — also feeds the code risk gate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.strategy import atr, ema

# +1: rises in risk-on markets; -1: rises in risk-off markets.
RISK_ON_SIGN = {"US500": 1, "USTEC": 1, "DE30": 1, "JP225": 1, "BTCUSD": 1, "USDJPY": 1, "XAUUSD": -1}


def _pct(a: float, b: float) -> float:
    return (a / b - 1) * 100 if b else 0.0


# ----- Sentiment (6): cross-asset risk appetite ------------------------------


def cross_asset_sentiment(adapter: Any, names: dict[str, str]) -> str:
    """`names` maps base (e.g. "US500") -> broker symbol (e.g. "US500m")."""
    lines, score, counted = [], 0, 0
    for base, name in names.items():
        try:
            d1 = adapter.get_rates(name, "D1", 30)
            h4 = adapter.get_rates(name, "H4", 60)
        except Exception:
            continue
        close = d1["close"]
        day, week = _pct(close.iloc[-1], close.iloc[-2]), _pct(close.iloc[-1], close.iloc[-6])
        h4_close = h4["close"]
        trend = "up" if ema(h4_close, 20).iloc[-1] > ema(h4_close, 50).iloc[-1] else "down"
        lines.append(f"    {name}: 1d {day:+.2f}%, 5d {week:+.2f}%, H4 trend {trend}")
        sign = RISK_ON_SIGN.get(base.upper())
        if sign:
            score += sign * (1 if day > 0 else -1)
            counted += 1
    if not lines:
        return "  (no cross-asset data available on this account)"
    mood = "risk-on" if score >= 2 else "risk-off" if score <= -2 else "mixed"
    return f"  Risk appetite: {mood} (score {score:+d} of {counted} risk gauges)\n" + "\n".join(lines)


# ----- Order flow proxy (7) and liquidity (8) ---------------------------------


@dataclass
class Microstructure:
    text: str
    spread_ratio: float  # current spread / typical spread for this hour


def microstructure(adapter: Any, name: str, info: Any, tick: Any, now: datetime) -> Microstructure:
    m1 = adapter.get_rates(name, "M1", 60)
    up = float(m1.loc[m1["close"] > m1["open"], "tick_volume"].sum())
    down = float(m1.loc[m1["close"] < m1["open"], "tick_volume"].sum())
    imbalance = (up - down) / (up + down) * 100 if up + down else 0.0
    bar_range = (m1["high"] - m1["low"]).replace(0, np.nan)
    close_location = float(((m1["close"] - m1["low"]) / bar_range).mean())

    m15 = adapter.get_rates(name, "M15", 20 * 96)
    same_hour = m15[pd.to_datetime(m15["time"]).dt.hour == now.hour]
    typical_ticks = float(same_hour["tick_volume"].median()) if len(same_hour) else 0.0
    recent_ticks = float(m1["tick_volume"].tail(15).sum())
    activity = recent_ticks / typical_ticks if typical_ticks else 1.0
    typical_spread = float(same_hour["spread"].median()) if len(same_hour) else 0.0
    spread_now = round((tick.ask - tick.bid) / info.point)
    spread_ratio = spread_now / typical_spread if typical_spread > 0 else 1.0

    pressure = "buying" if imbalance > 15 else "selling" if imbalance < -15 else "balanced"
    text = (
        f"    Order flow (proxy, last 60 min): {pressure} — up-bar tick volume {imbalance:+.0f}% vs down-bars, "
        f"bars closing at {close_location:.0%} of their range; activity {activity:.1f}x normal for this hour\n"
        f"    Liquidity: spread {spread_now} points vs typical {typical_spread:.0f} for this hour ({spread_ratio:.1f}x)"
    )
    return Microstructure(text=text, spread_ratio=spread_ratio)


# ----- Quantitative (4): base rates from multi-year history ------------------


def barrier_outcomes(df: pd.DataFrame, k: float = 1.0, horizon: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Per bar: did price reach +k*ATR before -k*ATR within `horizon` bars
    (long_win), and the reverse (short_win)? A bar that touches both levels
    counts as a loss for both sides — the conservative reading."""
    close, high, low = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    a = atr(df, 14).to_numpy()
    n = len(df)
    up, down = close + k * a, close - k * a
    first_up = np.full(n, np.inf)
    first_down = np.full(n, np.inf)
    for j in range(1, horizon + 1):
        valid = np.arange(n) + j < n
        hi = np.where(valid, np.roll(high, -j), -np.inf)
        lo = np.where(valid, np.roll(low, -j), np.inf)
        hit_up = (hi >= up) & np.isinf(first_up)
        hit_down = (lo <= down) & np.isinf(first_down)
        first_up[hit_up] = j
        first_down[hit_down] = j
    return first_up < first_down, first_down < first_up


def trend_states(df: pd.DataFrame) -> np.ndarray:
    """'up' / 'down' / 'mixed' per H1 bar, using H1 EMA20/50 and the last
    *completed* H4 bar's EMA20/50 (no lookahead)."""
    h1_up = (ema(df["close"], 20) > ema(df["close"], 50)).to_numpy()
    indexed = df.set_index(pd.to_datetime(df["time"]))
    h4_close = indexed["close"].resample("4h").last().dropna()
    h4_up = (ema(h4_close, 20) > ema(h4_close, 50)).astype(float).shift(1)
    h4_up = (h4_up.reindex(indexed.index, method="ffill").fillna(0.0) > 0.5).to_numpy()
    return np.where(h1_up & h4_up, "up", np.where(~h1_up & ~h4_up, "down", "mixed"))


def base_rates(df: pd.DataFrame, horizon: int = 24) -> dict[str, Any]:
    long_win, short_win = barrier_outcomes(df, 1.0, horizon)
    states = trend_states(df)
    usable = np.arange(len(df)) < len(df) - horizon  # outcome fully observed
    result: dict[str, Any] = {
        "years": round((pd.to_datetime(df["time"]).iloc[-1] - pd.to_datetime(df["time"]).iloc[0]).days / 365.25, 1),
        "current_state": str(states[-1]),
    }
    for state in ("up", "down", "mixed", "all"):
        mask = usable & ((states == state) if state != "all" else True)
        n = int(mask.sum())
        result[state] = {
            "n": n,
            "long_first": round(float(long_win[mask].mean()), 3) if n else None,
            "short_first": round(float(short_win[mask].mean()), 3) if n else None,
        }
    return result


class QuantBaseRates:
    """Computed once per UTC day per symbol (35k H1 bars), cached in state/."""

    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def get(self, adapter: Any, name: str, today: str) -> dict[str, Any] | None:
        cache = self._load()
        if cache.get("date") != today:
            cache = {"date": today, "symbols": {}}
        if name not in cache["symbols"]:
            try:
                df = adapter.get_rates(name, "H1", 35000)
            except Exception:
                return None
            if len(df) < 2000:
                return None
            cache["symbols"][name] = base_rates(df)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(cache), encoding="utf-8")
        return cache["symbols"][name]


def describe_base_rates(rates: dict[str, Any] | None, spread_price: float, atr_h1: float) -> str:
    if not rates:
        return "    Quant base rates: not enough history"
    state = rates["current_state"]
    now, every = rates[state], rates["all"]
    # Spread is paid on every trade: at 1-ATR stops a 1:1 trade needs p > (1 + cost) / 2.
    cost_r = spread_price / atr_h1 if atr_h1 else 0.0
    needed = (1 + cost_r) / 2
    return (
        f"    Quant base rates ({rates['years']}y H1, +/-1 ATR within 24h): trend state now '{state}' "
        f"(n={now['n']}): long reached +1 ATR first {now['long_first']:.1%}, short {now['short_first']:.1%}; "
        f"all bars: long {every['long_first']:.1%}, short {every['short_first']:.1%}. "
        f"Spread costs {cost_r:.2f}R, so a 1:1 trade needs about {needed:.1%} to break even."
    )
