"""Thin entrypoint: `python scripts/run_voice_assistant.py` starts the wake-word voice loop."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.voice.voice_assistant import main

if __name__ == "__main__":
    raise SystemExit(main())
