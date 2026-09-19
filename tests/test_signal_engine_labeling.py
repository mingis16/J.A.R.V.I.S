from __future__ import annotations

import math

import numpy as np
import pandas as pd

from signal_engine.labeling import triple_barrier_labels


def _bar(o, h, l, c):
    return {"open": o, "high": h, "low": l, "close": c}


def test_long_hits_take_profit_first():
    # entry close=100, atr=1, sl_mult=1 -> sl=99, tp_mult=2 -> tp=102
    df = pd.DataFrame(
        [
            _bar(100, 100, 100, 100),  # i=0, entry bar
            _bar(100, 101, 99.5, 100.5),  # i=1, no barrier touched
            _bar(100.5, 103, 100, 102.5),  # i=2, TP touched (high>=102)
        ]
    )
    side = np.array([1, 0, 0])
    atr = np.array([1.0, 1.0, 1.0])
    result = triple_barrier_labels(df, side, atr, atr_sl_mult=1.0, atr_tp_mult=2.0, horizon_bars=2)
    assert result.outcome[0] == "tp"
    assert result.label[0] == 1
    assert result.r_multiple[0] == 2.0
    assert result.bars_held[0] == 2


def test_long_hits_stop_loss_first():
    df = pd.DataFrame(
        [
            _bar(100, 100, 100, 100),
            _bar(100, 100.2, 98.5, 99),  # SL touched (low<=99)
            _bar(99, 101, 98, 100),
        ]
    )
    side = np.array([1, 0, 0])
    atr = np.array([1.0, 1.0, 1.0])
    result = triple_barrier_labels(df, side, atr, atr_sl_mult=1.0, atr_tp_mult=2.0, horizon_bars=2)
    assert result.outcome[0] == "sl"
    assert result.label[0] == 0
    assert result.r_multiple[0] == -1.0
    assert result.bars_held[0] == 1


def test_short_direction_flips_barriers():
    # short entry at 100: sl=101, tp=98
    df = pd.DataFrame(
        [
            _bar(100, 100, 100, 100),
            _bar(100, 100.5, 97.5, 98),  # low<=98 -> TP for short
        ]
    )
    side = np.array([-1, 0])
    atr = np.array([1.0, 1.0])
    result = triple_barrier_labels(df, side, atr, atr_sl_mult=1.0, atr_tp_mult=2.0, horizon_bars=1)
    assert result.outcome[0] == "tp"
    assert result.label[0] == 1


def test_ambiguous_bar_resolves_as_loss():
    df = pd.DataFrame(
        [
            _bar(100, 100, 100, 100),
            _bar(100, 103, 98, 100),  # both TP (>=102) and SL (<=99) touched
        ]
    )
    side = np.array([1, 0])
    atr = np.array([1.0, 1.0])
    result = triple_barrier_labels(df, side, atr, atr_sl_mult=1.0, atr_tp_mult=2.0, horizon_bars=1)
    assert result.outcome[0] == "ambiguous"
    assert result.label[0] == 0
    assert result.r_multiple[0] == -1.0
    assert result.ambiguity_rate() == 1.0


def test_timeout_when_neither_barrier_touched():
    df = pd.DataFrame(
        [
            _bar(100, 100, 100, 100),
            _bar(100, 100.5, 99.5, 100.2),  # neither barrier touched
        ]
    )
    side = np.array([1, 0])
    atr = np.array([1.0, 1.0])
    result = triple_barrier_labels(df, side, atr, atr_sl_mult=1.0, atr_tp_mult=2.0, horizon_bars=1)
    assert result.outcome[0] == "timeout"
    assert result.label[0] == 0
    # mark-to-market R = (100.2 - 100) / 1.0 = 0.2
    assert math.isclose(result.r_multiple[0], 0.2, abs_tol=1e-9)


def test_last_horizon_bars_are_unlabeled():
    df = pd.DataFrame([_bar(100, 100, 100, 100)] * 5)
    side = np.array([1, 1, 1, 1, 1])
    atr = np.array([1.0] * 5)
    result = triple_barrier_labels(df, side, atr, atr_sl_mult=1.0, atr_tp_mult=2.0, horizon_bars=2)
    assert result.outcome[3] == "insufficient_horizon"
    assert result.outcome[4] == "insufficient_horizon"
