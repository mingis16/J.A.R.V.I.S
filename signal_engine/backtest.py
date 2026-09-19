"""Walk-forward backtest: trains + calibrates per fold, evaluates on that
fold's held-out test block, and aggregates the metrics the spec requires.
Also runs the null test (shuffled labels) and breaks results out by year
and by regime.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from signal_engine.calibration import (
    Calibrator,
    brier_score,
    expected_calibration_error,
    fit_calibrator,
    reliability_diagram_data,
)
from signal_engine.config import SignalEngineConfig
from signal_engine.costs import cost_in_r, expectancy_r
from signal_engine.features import FEATURE_COLUMNS, build_features
from signal_engine.labeling import triple_barrier_labels
from signal_engine.model import BaselineModel, train_baseline
from signal_engine.regimes import classify_regime
from signal_engine.splits import Fold, purged_walk_forward_folds

VALIDATION_FRACTION = 0.2  # tail slice of each fold's train set, held out for calibration only
_LABELED_OUTCOMES = ["tp", "sl", "ambiguous", "timeout"]


def usable_rows(feats_slice: pd.DataFrame) -> pd.DataFrame:
    """Rows with a real label AND finite features — drops the rolling-window
    warmup period (first ~100 bars have NaN features) and unlabeled tail."""
    has_label = feats_slice["outcome"].isin(_LABELED_OUTCOMES)
    has_features = feats_slice[FEATURE_COLUMNS].notna().all(axis=1)
    return feats_slice[has_label & has_features]


@dataclass
class FoldResult:
    fold_index: int
    test_start_time: pd.Timestamp
    test_end_time: pd.Timestamp
    n_signals: int
    hit_rate: float
    mean_r: float
    expectancy_r: float
    max_drawdown_r: float
    sharpe: float
    brier: float
    ece: float
    ambiguity_rate: float
    trade_r: list[float] = field(default_factory=list)
    per_year: dict = field(default_factory=dict)
    per_regime: dict = field(default_factory=dict)
    reliability: list = field(default_factory=list)


def _max_drawdown_r(trade_r: np.ndarray) -> float:
    if len(trade_r) == 0:
        return 0.0
    equity = np.cumsum(trade_r)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    return float(drawdown.min())


def _sharpe(trade_r: np.ndarray) -> float:
    if len(trade_r) < 2 or trade_r.std() == 0:
        return 0.0
    return float(trade_r.mean() / trade_r.std() * np.sqrt(len(trade_r)))


def prepare_dataset(df: pd.DataFrame, cfg: SignalEngineConfig):
    feats = build_features(df)
    side = np.where(feats["ema_fast"] > feats["ema_slow"], 1, -1)
    labels = triple_barrier_labels(df, side, feats["atr_14"].to_numpy(), cfg.atr_sl_mult, cfg.atr_tp_mult, cfg.horizon_bars)
    feats["side"] = side
    feats["label"] = labels.label
    feats["outcome"] = labels.outcome
    feats["r_multiple"] = labels.r_multiple
    feats["regime"] = classify_regime(feats["atr_percentile_100"], feats["ema_spread_atr"])
    return feats, labels


def _score_fold(fold: Fold, feats: pd.DataFrame, cfg: SignalEngineConfig, symbol: str, calibration_method: str, spread_price: float, fold_index: int, shuffle_labels: bool = False) -> FoldResult | None:
    train_full = usable_rows(feats.iloc[fold.train_idx])
    if len(train_full) < 50:
        return None

    split_point = int(len(train_full) * (1 - VALIDATION_FRACTION))
    train = train_full.iloc[:split_point]
    val = train_full.iloc[split_point:]
    if len(val) < 20:
        return None

    test = usable_rows(feats.iloc[fold.test_idx])
    if len(test) == 0:
        return None

    y_train = train["label"].to_numpy()
    if shuffle_labels:
        rng = np.random.default_rng(42 + fold_index)
        y_train = rng.permutation(y_train)

    X_train = train[FEATURE_COLUMNS].to_numpy()
    model: BaselineModel = train_baseline(X_train, y_train, FEATURE_COLUMNS)

    X_val = val[FEATURE_COLUMNS].to_numpy()
    raw_p_val = model.predict_proba(X_val)
    y_val = val["label"].to_numpy()
    if shuffle_labels:
        rng = np.random.default_rng(142 + fold_index)
        y_val = rng.permutation(y_val)
    calibrator: Calibrator = fit_calibrator(raw_p_val, y_val, method=calibration_method)

    X_test = test[FEATURE_COLUMNS].to_numpy()
    raw_p_test = model.predict_proba(X_test)
    p_test = calibrator.calibrate(raw_p_test)
    y_test = test["label"].to_numpy()

    risk_reward = cfg.atr_tp_mult / cfg.atr_sl_mult
    sl_distance_price = test["atr_14"].to_numpy() * cfg.atr_sl_mult
    costs = np.array(
        [cost_in_r(d, spread_price, spread_price * (cfg.spread_slippage_mult - 1), cfg.commission_r) for d in sl_distance_price]
    )
    expectancy = np.array([expectancy_r(p, risk_reward, c) for p, c in zip(p_test, costs)])

    take = (p_test > cfg.probability_threshold) & (expectancy > 0)
    n_signals = int(take.sum())
    if n_signals == 0:
        trade_r = np.array([])
        hit_rate = float("nan")
        mean_r = float("nan")
        mean_expectancy = float("nan")
    else:
        trade_r = test["r_multiple"].to_numpy()[take]
        hit_rate = float((y_test[take] == 1).mean())
        mean_r = float(trade_r.mean())
        mean_expectancy = float(expectancy[take].mean())

    per_year: dict = {}
    per_regime: dict = {}
    if n_signals > 0:
        years = test["time"].dt.year.to_numpy()[take]
        regimes = test["regime"].to_numpy()[take]
        for yr in np.unique(years):
            m = years == yr
            per_year[int(yr)] = {
                "n_signals": int(m.sum()),
                "hit_rate": float((y_test[take][m] == 1).mean()),
                "mean_r": float(trade_r[m].mean()),
            }
        for reg in np.unique(regimes):
            m = regimes == reg
            per_regime[str(reg)] = {
                "n_signals": int(m.sum()),
                "hit_rate": float((y_test[take][m] == 1).mean()),
                "mean_r": float(trade_r[m].mean()),
            }

    return FoldResult(
        fold_index=fold_index,
        test_start_time=test["time"].iloc[0],
        test_end_time=test["time"].iloc[-1],
        n_signals=n_signals,
        hit_rate=hit_rate,
        mean_r=mean_r,
        expectancy_r=mean_expectancy,
        max_drawdown_r=_max_drawdown_r(trade_r),
        sharpe=_sharpe(trade_r),
        brier=brier_score(y_test, p_test),
        ece=expected_calibration_error(y_test, p_test),
        ambiguity_rate=float((test["outcome"] == "ambiguous").mean()),
        trade_r=trade_r.tolist(),
        per_year=per_year,
        per_regime=per_regime,
        reliability=reliability_diagram_data(y_test, p_test),
    )


@dataclass
class BacktestReport:
    symbol: str
    folds: list[FoldResult]
    null_test_expectancy: float
    cost_sensitivity: dict  # {multiplier: mean_expectancy}
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    n_bars: int


def run_walk_forward(df: pd.DataFrame, cfg: SignalEngineConfig, symbol: str, spread_price: float, calibration_method: str = "sigmoid") -> BacktestReport:
    feats, _ = prepare_dataset(df, cfg)
    folds_spec = purged_walk_forward_folds(len(df), cfg.n_folds, cfg.horizon_bars, cfg.embargo_bars, cfg.min_train_bars)

    fold_results = []
    for i, fold in enumerate(folds_spec):
        result = _score_fold(fold, feats, cfg, symbol, calibration_method, spread_price, i)
        if result is not None:
            fold_results.append(result)

    # Null test: shuffle labels on the last (largest-train) fold only — cheap and sufficient to confirm no edge.
    null_expectancy = float("nan")
    if folds_spec:
        null_result = _score_fold(folds_spec[-1], feats, cfg, symbol, calibration_method, spread_price, len(folds_spec) - 1, shuffle_labels=True)
        if null_result is not None:
            null_expectancy = null_result.expectancy_r

    # Cost sensitivity on the last fold across multipliers.
    cost_sensitivity = {}
    if folds_spec:
        for mult in (0.5, 1.0, 2.0):
            saved = cfg.spread_slippage_mult
            cfg.spread_slippage_mult = 1.0 + (saved - 1.0) * mult if saved > 1.0 else mult
            r = _score_fold(folds_spec[-1], feats, cfg, symbol, calibration_method, spread_price * mult, len(folds_spec) - 1)
            cfg.spread_slippage_mult = saved
            cost_sensitivity[mult] = r.expectancy_r if r is not None else float("nan")

    return BacktestReport(
        symbol=symbol,
        folds=fold_results,
        null_test_expectancy=null_expectancy,
        cost_sensitivity=cost_sensitivity,
        data_start=df["time"].iloc[0],
        data_end=df["time"].iloc[-1],
        n_bars=len(df),
    )
