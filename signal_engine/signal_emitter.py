"""Produces a live signal (or None) for the most recent bar, matching the
spec's output contract exactly. Trains on all available labeled history,
calibrates on a held-out tail slice — same discipline as the backtest, just
scored on the newest bar instead of a historical test fold.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from signal_engine.backtest import VALIDATION_FRACTION, prepare_dataset, usable_rows
from signal_engine.calibration import fit_calibrator, reliability_diagram_data
from signal_engine.config import SignalEngineConfig
from signal_engine.costs import cost_in_r, expectancy_r
from signal_engine.features import FEATURE_COLUMNS
from signal_engine.model import train_baseline

MIN_LABELED_BARS = 200


def latest_signal(
    df: pd.DataFrame,
    cfg: SignalEngineConfig,
    symbol: str,
    spread_price: float,
    calibration_method: str = "sigmoid",
) -> dict[str, Any] | None:
    feats, _ = prepare_dataset(df, cfg)
    labeled = usable_rows(feats)
    if len(labeled) < MIN_LABELED_BARS:
        return None

    last = feats.iloc[-1]
    if last[FEATURE_COLUMNS].isna().any():
        return None

    split = int(len(labeled) * (1 - VALIDATION_FRACTION))
    train, val = labeled.iloc[:split], labeled.iloc[split:]
    if len(val) < 20:
        return None

    X_train = train[FEATURE_COLUMNS].to_numpy()
    model = train_baseline(X_train, train["label"].to_numpy(), FEATURE_COLUMNS)

    raw_p_val = model.predict_proba(val[FEATURE_COLUMNS].to_numpy())
    y_val = val["label"].to_numpy()
    calibrator = fit_calibrator(raw_p_val, y_val, method=calibration_method)
    p_val_calibrated = calibrator.calibrate(raw_p_val)

    X_last = last[FEATURE_COLUMNS].to_numpy().reshape(1, -1)
    raw_p_last = model.predict_proba(X_last)[0]
    p_calibrated = float(calibrator.calibrate(np.array([raw_p_last]))[0])

    risk_reward = cfg.atr_tp_mult / cfg.atr_sl_mult
    sl_distance_price = float(last["atr_14"]) * cfg.atr_sl_mult
    cost = cost_in_r(sl_distance_price, spread_price, spread_price * (cfg.spread_slippage_mult - 1), cfg.commission_r)
    expectancy = expectancy_r(p_calibrated, risk_reward, cost)

    if not (p_calibrated > cfg.probability_threshold and expectancy > 0):
        return None

    side = int(last["side"])
    entry = float(last["close"])
    atr_val = float(last["atr_14"])
    sl_dist = atr_val * cfg.atr_sl_mult
    tp_dist = atr_val * cfg.atr_tp_mult
    stop = entry - side * sl_dist
    take_profit = entry + side * tp_dist

    coefs = model.classifier.coef_[0]
    scaled_val = model.scaler.transform(X_last)[0]
    contributions = coefs * scaled_val
    top_idx = np.argsort(-np.abs(contributions))[:3]
    features_top = {FEATURE_COLUMNS[i]: round(float(last[FEATURE_COLUMNS[i]]), 4) for i in top_idx}

    buckets = reliability_diagram_data(y_val, p_val_calibrated, n_buckets=10)
    bucket_idx = min(int(p_calibrated * 10), 9)
    sample_size_in_bucket = buckets[bucket_idx].count

    return {
        "timestamp_utc": last["time"].isoformat(),
        "pair": symbol,
        "direction": "long" if side > 0 else "short",
        "entry": round(entry, 5),
        "stop": round(stop, 5),
        "take_profit": round(take_profit, 5),
        "atr_at_signal": round(atr_val, 5),
        "risk_reward": round(risk_reward, 2),
        "p_tp_calibrated": round(p_calibrated, 4),
        "expectancy_R_after_costs": round(float(expectancy), 4),
        "features_top": features_top,
        "sample_size_in_bucket": int(sample_size_in_bucket),
        "regime": str(last["regime"]),
    }
