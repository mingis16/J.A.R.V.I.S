from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from trading_bot.config import REPO_ROOT, get_mt5_credentials, load_yaml_config
from trading_bot.executor import Executor
from trading_bot.mt5_adapter import MT5Adapter
from trading_bot.strategy import generate_signal
from trading_bot.trade_log import TradeLog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("trading_bot.main")

_shutdown_requested = False


def _handle_sigint(signum, frame) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Shutdown requested, will stop after current cycle.")


def run_cycle(adapter: MT5Adapter, executor: Executor, trading_cfg: dict) -> str:
    df = adapter.get_rates(trading_cfg["symbol"], trading_cfg["timeframe"], trading_cfg["history_bars"])
    signal_obj = generate_signal(df, trading_cfg)
    outcome = executor.handle_signal(signal_obj)
    logger.info("%s %s -> %s", trading_cfg["symbol"], signal_obj.action, outcome)
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. forex trading bot (MetaTrader 5)")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit.")
    parser.add_argument("--status", action="store_true", help="Print account/position status and exit.")
    args = parser.parse_args()

    cfg = load_yaml_config()
    trading_cfg = cfg["trading"]
    execution_cfg = cfg["execution"]

    try:
        creds = get_mt5_credentials()
        adapter = MT5Adapter(creds)
        adapter.connect()
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    try:
        if args.status:
            account = adapter.get_account_info()
            positions = adapter.get_open_positions(trading_cfg["symbol"])
            print(f"Balance: {account.balance}  Equity: {account.equity}  Margin: {account.margin}")
            print(f"Open positions on {trading_cfg['symbol']}: {len(positions)}")
            for p in positions:
                print(f"  #{p.ticket} type={p.type} volume={p.volume} price_open={p.price_open}")
            return 0

        trade_log = TradeLog(REPO_ROOT / cfg["assistant"]["trade_log_path"])
        executor = Executor(adapter, trade_log, trading_cfg, execution_cfg)

        if execution_cfg["live_trading"]:
            logger.warning(
                "live_trading=true — this bot WILL attempt real orders if JARVIS_CONFIRM_LIVE "
                "is set correctly in the environment."
            )
        else:
            logger.info("Running in PAPER mode (execution.live_trading=false). No real orders will be sent.")

        if args.once:
            run_cycle(adapter, executor, trading_cfg)
            return 0

        signal.signal(signal.SIGINT, _handle_sigint)
        logger.info(
            "Starting poll loop for %s every %ss. Press Ctrl+C to stop.",
            trading_cfg["symbol"],
            trading_cfg["poll_seconds"],
        )
        while not _shutdown_requested:
            try:
                run_cycle(adapter, executor, trading_cfg)
            except Exception:
                logger.exception("Cycle failed; will retry next interval.")
            for _ in range(trading_cfg["poll_seconds"]):
                if _shutdown_requested:
                    break
                time.sleep(1)
        return 0
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    sys.exit(main())
