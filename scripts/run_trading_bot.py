"""Thin entrypoint: `python scripts/run_trading_bot.py` == `python -m trading_bot.main`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_bot.main import main

if __name__ == "__main__":
    raise SystemExit(main())
