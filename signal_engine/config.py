from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trading_bot.config import REPO_ROOT


@dataclass
class SignalEngineConfig:
    pairs: list[str]
    timeframe: str
    history_bars: int
    horizon_bars: int
    atr_sl_mult: float
    atr_tp_mult: float
    probability_threshold: float
    spread_slippage_mult: float
    commission_r: float
    n_folds: int
    embargo_bars: int
    min_train_bars: int
    cache_dir: Path
    report_dir: Path

    @classmethod
    def from_yaml(cls, cfg: dict, repo_root: Path = REPO_ROOT) -> "SignalEngineConfig":
        se = cfg["signal_engine"]
        return cls(
            pairs=list(se["pairs"]),
            timeframe=se["timeframe"],
            history_bars=se["history_bars"],
            horizon_bars=se["horizon_bars"],
            atr_sl_mult=se["atr_sl_mult"],
            atr_tp_mult=se["atr_tp_mult"],
            probability_threshold=se["probability_threshold"],
            spread_slippage_mult=se["spread_slippage_mult"],
            commission_r=se["commission_r"],
            n_folds=se["n_folds"],
            embargo_bars=se["embargo_bars"],
            min_train_bars=se["min_train_bars"],
            cache_dir=repo_root / se["cache_dir"],
            report_dir=repo_root / se["report_dir"],
        )
