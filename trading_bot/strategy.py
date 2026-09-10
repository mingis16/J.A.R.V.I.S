from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Action = Literal["BUY", "SELL", "HOLD"]


@dataclass
class Signal:
    action: Action
    reason: str
    close_price: float
    atr_value: float


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    result = 100 - (100 / (1 + rs))
    return result.fillna(50.0)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def generate_signal(df: pd.DataFrame, params: dict) -> Signal:
    """EMA-crossover trend signal filtered by RSI, sized off ATR.

    `df` must be sorted oldest -> newest with columns open/high/low/close,
    at least `slow_ema_period + 2` rows.
    """
    required = max(params["slow_ema_period"], params["rsi_period"], params["atr_period"]) + 2
    if len(df) < required:
        return Signal("HOLD", f"insufficient history ({len(df)} < {required} bars)", float("nan"), float("nan"))

    close = df["close"]
    fast = ema(close, params["fast_ema_period"])
    slow = ema(close, params["slow_ema_period"])
    rsi_series = rsi(close, params["rsi_period"])
    atr_series = atr(df, params["atr_period"])

    prev_fast, prev_slow = fast.iloc[-2], slow.iloc[-2]
    last_fast, last_slow = fast.iloc[-1], slow.iloc[-1]
    last_rsi = rsi_series.iloc[-1]
    last_close = close.iloc[-1]
    last_atr = atr_series.iloc[-1]

    crossed_up = prev_fast <= prev_slow and last_fast > last_slow
    crossed_down = prev_fast >= prev_slow and last_fast < last_slow

    if crossed_up and last_rsi < params["rsi_overbought"]:
        return Signal("BUY", f"EMA bullish cross, RSI={last_rsi:.1f}", last_close, last_atr)
    if crossed_down and last_rsi > params["rsi_oversold"]:
        return Signal("SELL", f"EMA bearish cross, RSI={last_rsi:.1f}", last_close, last_atr)
    return Signal("HOLD", f"no fresh cross (RSI={last_rsi:.1f})", last_close, last_atr)
