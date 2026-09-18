"""Thin entrypoint: `python scripts/run_telegram_bot.py` starts the Telegram
front-end (chat with Alex + push trading-signal notifications) via polling.
No public server/webhook needed. See README for setup (BotFather token +
finding your Telegram user ID).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.telegram.bot import main

if __name__ == "__main__":
    raise SystemExit(main())
