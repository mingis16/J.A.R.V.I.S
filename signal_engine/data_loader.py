"""Pulls H1 history from the MT5 demo account and caches it locally so
repeated runs (feature tuning, tests, report regeneration) don't re-hit the
terminal every time.

Known limitation (see REPORT.md Limitations section): this broker/demo
server caps copy_rates_from_pos at 20,000 H1 bars regardless of the count
requested, i.e. ~3.2 years — short of the spec's 5-year target.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from trading_bot.config import get_mt5_credentials
from trading_bot.mt5_adapter import MT5Adapter

logger = logging.getLogger("signal_engine.data_loader")


def _cache_path(cache_dir: Path, symbol: str, timeframe: str) -> Path:
    return cache_dir / f"{symbol}_{timeframe}.csv"


def pull_history(symbol: str, timeframe: str, count: int, cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Returns a DataFrame sorted oldest->newest with columns:
    time (UTC tz-aware), open, high, low, close, tick_volume, spread, real_volume.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, symbol, timeframe)

    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["time"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        logger.info("Loaded %d cached bars for %s from %s", len(df), symbol, path)
        return df

    creds = get_mt5_credentials()
    adapter = MT5Adapter(creds)
    adapter.connect()
    try:
        df = adapter.get_rates(symbol, timeframe, count)
    finally:
        adapter.disconnect()

    df = df.sort_values("time").reset_index(drop=True)
    df.to_csv(path, index=False)
    logger.info("Pulled %d bars for %s (%s -> %s), cached to %s", len(df), symbol, df["time"].min(), df["time"].max(), path)
    return df
