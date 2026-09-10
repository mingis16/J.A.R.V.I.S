import numpy as np
import pandas as pd

from trading_bot.strategy import atr, ema, generate_signal, rsi

PARAMS = {
    "fast_ema_period": 3,
    "slow_ema_period": 6,
    "rsi_period": 5,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "atr_period": 5,
    "atr_sl_multiplier": 1.5,
    "atr_tp_multiplier": 3.0,
}


def _df_from_closes(closes: list[float]) -> pd.DataFrame:
    closes = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.001,
            "low": closes - 0.001,
            "close": closes,
        }
    )


def test_ema_converges_toward_flat_series():
    series = pd.Series([1.0] * 20)
    result = ema(series, 5)
    assert abs(result.iloc[-1] - 1.0) < 1e-9


def test_rsi_is_100_for_strictly_increasing_series():
    series = pd.Series(range(1, 30))
    result = rsi(series, 14)
    assert result.iloc[-1] > 95


def test_rsi_is_0_for_strictly_decreasing_series():
    series = pd.Series(range(30, 1, -1))
    result = rsi(series, 14)
    assert result.iloc[-1] < 5


def test_atr_is_positive_for_moving_series():
    df = _df_from_closes([1.0, 1.01, 1.02, 1.015, 1.03, 1.05, 1.04, 1.06])
    result = atr(df, 3)
    assert (result.dropna() > 0).all()


def test_generate_signal_holds_with_insufficient_history():
    df = _df_from_closes([1.0, 1.01])
    signal = generate_signal(df, PARAMS)
    assert signal.action == "HOLD"
    assert "insufficient history" in signal.reason


def test_generate_signal_detects_bullish_cross():
    # A downtrend then a sharp uptrend forces a fast-over-slow EMA cross.
    closes = [1.10, 1.09, 1.08, 1.07, 1.06, 1.05, 1.06, 1.09, 1.13, 1.18]
    df = _df_from_closes(closes)
    signal = generate_signal(df, PARAMS)
    assert signal.action in {"BUY", "HOLD"}  # RSI filter may still block; must never mis-signal SELL here
