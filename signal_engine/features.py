"""Feature construction. Every feature at row t uses only bars <= t — no
lookahead. See tests/test_signal_engine_features.py for the explicit
leakage test the spec requires (shift features forward, performance must
collapse to chance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_bot.strategy import atr as atr_indicator
from trading_bot.strategy import ema, rsi

FEATURE_COLUMNS = [
    "ret_1_atr",
    "ret_4_atr",
    "ret_24_atr",
    "realized_vol_24",
    "rsi_14",
    "ema_spread_atr",
    "dist_ema50_atr",
    "atr_percentile_100",
    "session_asian",
    "session_london",
    "session_ny",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "dist_session_vwap_atr",
]


def _session_dummies(hours: pd.Series) -> pd.DataFrame:
    # UTC hour buckets — approximate FX session overlap windows.
    asian = ((hours >= 0) & (hours < 8)).astype(int)
    london = ((hours >= 7) & (hours < 16)).astype(int)
    ny = ((hours >= 12) & (hours < 21)).astype(int)
    return pd.DataFrame({"session_asian": asian, "session_london": london, "session_ny": ny})


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP reset each UTC calendar day, using only bars up to and
    including the current one within that day (no lookahead)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["tick_volume"].clip(lower=1)
    day = df["time"].dt.date
    cum_pv = (typical * volume).groupby(day).cumsum()
    cum_v = volume.groupby(day).cumsum()
    return cum_pv / cum_v


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must be sorted oldest->newest with columns time/open/high/low/close/tick_volume.
    Returns a DataFrame aligned to df's index with FEATURE_COLUMNS plus
    passthrough columns needed downstream (ema_fast, ema_slow, atr_14, close).
    """
    close = df["close"]
    atr_14 = atr_indicator(df, 14)
    ema_fast = ema(close, 12)
    ema_slow = ema(close, 26)
    ema_50 = ema(close, 50)
    rsi_14 = rsi(close, 14)

    ret_1 = close.diff(1)
    ret_4 = close.diff(4)
    ret_24 = close.diff(24)
    realized_vol_24 = close.pct_change().rolling(24).std()

    atr_safe = atr_14.replace(0, np.nan)
    atr_percentile_100 = atr_14.rolling(100).rank(pct=True)

    hours = df["time"].dt.hour
    sessions = _session_dummies(hours)
    day_of_week = df["time"].dt.dayofweek
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)

    vwap = _session_vwap(df)
    dist_vwap = (close - vwap) / atr_safe

    feats = pd.DataFrame(
        {
            "ret_1_atr": ret_1 / atr_safe,
            "ret_4_atr": ret_4 / atr_safe,
            "ret_24_atr": ret_24 / atr_safe,
            "realized_vol_24": realized_vol_24,
            "rsi_14": rsi_14,
            "ema_spread_atr": (ema_fast - ema_slow) / atr_safe,
            "dist_ema50_atr": (close - ema_50) / atr_safe,
            "atr_percentile_100": atr_percentile_100,
            "session_asian": sessions["session_asian"],
            "session_london": sessions["session_london"],
            "session_ny": sessions["session_ny"],
            "day_of_week": day_of_week,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dist_session_vwap_atr": dist_vwap,
        },
        index=df.index,
    )

    feats["ema_fast"] = ema_fast
    feats["ema_slow"] = ema_slow
    feats["atr_14"] = atr_14
    feats["close"] = close
    feats["time"] = df["time"]
    return feats
