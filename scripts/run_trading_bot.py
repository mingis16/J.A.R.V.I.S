"""Thin entrypoint: `python scripts/run_trading_bot.py` == `python -m trading_bot.main`."""

from trading_bot.main import main

if __name__ == "__main__":
    raise SystemExit(main())
