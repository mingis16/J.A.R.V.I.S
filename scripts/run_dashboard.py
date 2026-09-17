"""Thin entrypoint: `python scripts/run_dashboard.py` starts the local web
dashboard at http://127.0.0.1:5000 (port configurable via dashboard.port in
config/config.yaml).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.dashboard.app import main

if __name__ == "__main__":
    raise SystemExit(main())
