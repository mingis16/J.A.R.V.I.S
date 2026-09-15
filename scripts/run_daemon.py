"""Thin entrypoint: `python scripts/run_daemon.py` starts the 24/7 background
scheduler. Pass --once to run every enabled routine a single time immediately
(for manual testing) instead of looping forever.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.daemon.scheduler import main

if __name__ == "__main__":
    raise SystemExit(main())
