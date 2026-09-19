"""Calibrated FX signal engine.

    python scripts/run_signal_engine.py --train   # walk-forward backtest, writes REPORT.md
    python scripts/run_signal_engine.py --emit     # print today's signal JSON per pair, if any
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_bot.config import REPO_ROOT, get_mt5_credentials, load_env, load_yaml_config
from trading_bot.mt5_adapter import MT5Adapter

from signal_engine.backtest import run_walk_forward
from signal_engine.config import SignalEngineConfig
from signal_engine.data_loader import pull_history
from signal_engine.report import write_report
from signal_engine.signal_emitter import latest_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("signal_engine.cli")


def _current_spread_price(symbol: str) -> float:
    creds = get_mt5_credentials()
    adapter = MT5Adapter(creds)
    adapter.connect()
    try:
        info = adapter.get_symbol_info(symbol)
        return float(info.spread) * float(info.point)
    finally:
        adapter.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrated FX signal engine")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Run the walk-forward backtest and write REPORT.md")
    group.add_argument("--emit", action="store_true", help="Print today's signal (if any) for each configured pair")
    parser.add_argument("--refresh-data", action="store_true", help="Re-pull history from MT5 instead of using the cache")
    args = parser.parse_args()

    load_env()
    cfg_yaml = load_yaml_config()
    cfg = SignalEngineConfig.from_yaml(cfg_yaml, REPO_ROOT)

    if args.train:
        reports = []
        for symbol in cfg.pairs:
            logger.info("Pulling history for %s...", symbol)
            df = pull_history(symbol, cfg.timeframe, cfg.history_bars, cfg.cache_dir, refresh=args.refresh_data)
            spread_price = _current_spread_price(symbol)
            logger.info("Running walk-forward backtest for %s (spread proxy=%s)...", symbol, spread_price)
            report = run_walk_forward(df, cfg, symbol, spread_price)
            reports.append(report)
        path = write_report(reports, cfg)
        logger.info("Wrote %s", path)
        return 0

    if args.emit:
        for symbol in cfg.pairs:
            df = pull_history(symbol, cfg.timeframe, cfg.history_bars, cfg.cache_dir, refresh=args.refresh_data)
            spread_price = _current_spread_price(symbol)
            signal = latest_signal(df, cfg, symbol, spread_price)
            if signal:
                print(json.dumps(signal, indent=2))
            else:
                print(f"{symbol}: no signal (below probability/expectancy threshold or insufficient data)")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
