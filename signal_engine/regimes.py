"""Simple, transparent regime classification for the per-regime breakdown
the spec asks for, and the `regime` field in the output contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def classify_regime(atr_percentile: pd.Series, ema_spread_atr: pd.Series, trend_threshold: float = 0.5) -> pd.Series:
    """atr_percentile: rolling percentile rank of ATR (0-1), from features.py.
    ema_spread_atr: (ema_fast - ema_slow) / atr, from features.py.
    """
    vol = pd.cut(atr_percentile, bins=[-0.01, 0.33, 0.66, 1.01], labels=["low_vol", "mid_vol", "high_vol"])
    trending = ema_spread_atr.abs() >= trend_threshold
    trend_label = np.where(trending, "trending", "ranging")
    combined = pd.Series(
        [f"{t}_{v}" if pd.notna(v) else "unknown" for t, v in zip(trend_label, vol)],
        index=atr_percentile.index,
    )
    return combined
