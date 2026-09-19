"""signal_emitter.latest_signal's gating logic (threshold + positive
expectancy) is exercised naturally by the real MT5 data producing no
signal — this test uses synthetic data engineered to clear both gates, so
the actual output-contract JSON-building branch gets covered too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signal_engine.config import SignalEngineConfig
from signal_engine.signal_emitter import latest_signal


def _strongly_trending_df(n_bars: int, seed: int = 11) -> pd.DataFrame:
    """One long, clean uptrend — the EMA-cross "long" candidate should hit
    TP overwhelmingly often, giving the model a strong, easy-to-learn,
    genuinely-high-probability setup at the most recent bar."""
    rng = np.random.default_rng(seed)
    drift = 0.06
    noise = rng.normal(0, 0.03, size=n_bars)
    # Occasional pullback shocks so some long candidates genuinely hit SL —
    # otherwise every row is a win and LogisticRegression has only one class.
    shocks = np.where(rng.random(n_bars) < 0.08, rng.normal(-0.4, 0.1, size=n_bars), 0.0)
    close = 100.0 + np.cumsum(drift + noise + shocks)
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) + 0.02
    low = np.minimum(open_, close) - 0.02
    times = pd.date_range("2020-01-01", periods=n_bars, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": rng.integers(50, 200, size=n_bars),
        }
    )


def _cfg(tmp_path) -> SignalEngineConfig:
    return SignalEngineConfig(
        pairs=["TEST"],
        timeframe="H1",
        history_bars=3000,
        horizon_bars=10,
        atr_sl_mult=1.0,
        atr_tp_mult=2.0,
        probability_threshold=0.4,
        spread_slippage_mult=1.0,
        commission_r=0.0,
        n_folds=3,
        embargo_bars=10,
        min_train_bars=500,
        cache_dir=tmp_path / "cache",
        report_dir=tmp_path / "report",
    )


def test_latest_signal_emits_output_contract_fields(tmp_path):
    df = _strongly_trending_df(3000)
    cfg = _cfg(tmp_path)

    signal = latest_signal(df, cfg, "TEST", spread_price=0.001)

    assert signal is not None, "expected a strong, clean uptrend to clear both the probability and expectancy gates"
    for key in (
        "timestamp_utc",
        "pair",
        "direction",
        "entry",
        "stop",
        "take_profit",
        "atr_at_signal",
        "risk_reward",
        "p_tp_calibrated",
        "expectancy_R_after_costs",
        "features_top",
        "sample_size_in_bucket",
        "regime",
    ):
        assert key in signal, f"missing output contract field: {key}"

    assert signal["direction"] == "long"
    assert signal["pair"] == "TEST"
    assert signal["p_tp_calibrated"] > cfg.probability_threshold
    assert signal["expectancy_R_after_costs"] > 0
    assert signal["sample_size_in_bucket"] > 0
    assert len(signal["features_top"]) == 3
    assert signal["take_profit"] > signal["entry"] > signal["stop"]  # long: TP above entry above stop


def test_latest_signal_none_when_insufficient_history(tmp_path):
    df = _strongly_trending_df(50)  # far below MIN_LABELED_BARS
    cfg = _cfg(tmp_path)

    assert latest_signal(df, cfg, "TEST", spread_price=0.001) is None
