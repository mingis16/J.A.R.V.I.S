"""Triple-barrier labeling.

For each candidate entry at bar i (direction given by `side`, +1 long /
-1 short), stop and take-profit are set at entry +/- ATR(i)*multiplier.
Barriers are checked against bar highs/lows (not closes) starting at bar
i+1 — the decision is made from information available at bar i's close,
and outcomes are only ever read from strictly later bars, so this cannot
leak.

Outcomes: 'tp' (label=1), 'sl' (label=0), 'timeout' (label=0, tracked
separately since it isn't a clean loss), 'ambiguous' (both barriers touched
within the same bar — resolved conservatively as a loss per spec).
'insufficient_horizon' for the last `horizon_bars` rows, which can't be
labeled at all (excluded from training).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LabelResult:
    label: np.ndarray  # 0/1, NaN (as -1 sentinel) where unlabeled
    outcome: list  # 'tp' | 'sl' | 'timeout' | 'ambiguous' | 'insufficient_horizon'
    r_multiple: np.ndarray
    bars_held: np.ndarray

    def ambiguity_rate(self) -> float:
        labeled = [o for o in self.outcome if o != "insufficient_horizon"]
        if not labeled:
            return 0.0
        return sum(1 for o in labeled if o == "ambiguous") / len(labeled)


def triple_barrier_labels(
    df: pd.DataFrame,
    side: np.ndarray,
    atr: np.ndarray,
    atr_sl_mult: float,
    atr_tp_mult: float,
    horizon_bars: int,
) -> LabelResult:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    label = np.full(n, -1, dtype=int)
    outcome: list = ["insufficient_horizon"] * n
    r_multiple = np.full(n, np.nan)
    bars_held = np.full(n, -1, dtype=int)
    risk_reward = atr_tp_mult / atr_sl_mult

    for i in range(n - horizon_bars):
        s = side[i]
        if s == 0 or np.isnan(atr[i]) or atr[i] <= 0:
            outcome[i] = "insufficient_horizon"
            continue

        entry = close[i]
        sl_dist = atr[i] * atr_sl_mult
        tp_dist = atr[i] * atr_tp_mult
        tp_price = entry + s * tp_dist
        sl_price = entry - s * sl_dist

        resolved = False
        for h in range(1, horizon_bars + 1):
            j = i + h
            if s > 0:
                hit_tp = high[j] >= tp_price
                hit_sl = low[j] <= sl_price
            else:
                hit_tp = low[j] <= tp_price
                hit_sl = high[j] >= sl_price

            if hit_tp and hit_sl:
                outcome[i] = "ambiguous"
                label[i] = 0
                r_multiple[i] = -1.0
                bars_held[i] = h
                resolved = True
                break
            if hit_tp:
                outcome[i] = "tp"
                label[i] = 1
                r_multiple[i] = risk_reward
                bars_held[i] = h
                resolved = True
                break
            if hit_sl:
                outcome[i] = "sl"
                label[i] = 0
                r_multiple[i] = -1.0
                bars_held[i] = h
                resolved = True
                break

        if not resolved:
            exit_price = close[i + horizon_bars]
            outcome[i] = "timeout"
            label[i] = 0
            r_multiple[i] = (s * (exit_price - entry)) / sl_dist
            bars_held[i] = horizon_bars

    return LabelResult(label=label, outcome=outcome, r_multiple=r_multiple, bars_held=bars_held)
