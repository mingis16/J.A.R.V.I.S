"""Thin entrypoint: `python scripts/run_assistant.py` == `python -m assistant.cli`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
