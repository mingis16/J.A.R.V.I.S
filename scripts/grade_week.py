"""Grades signals from a date range against what actually happened next in
the market — the answer to "how accurate were the signals."

    python scripts/grade_week.py                    # this week (Mon 00:00 UTC -> now)
    python scripts/grade_week.py --since 2026-09-15  # custom start date (UTC)

See signal_engine/grading.py for the actual logic.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_bot.config import REPO_ROOT, get_mt5_credentials, load_env, load_yaml_config
from trading_bot.mt5_adapter import MT5Adapter

from signal_engine.grading import format_report, grade_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Grade signals against what actually happened.")
    parser.add_argument("--since", help="UTC date (YYYY-MM-DD) to grade from. Default: this week's Monday.")
    args = parser.parse_args()

    load_env()
    cfg = load_yaml_config()

    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        now = datetime.now(timezone.utc)
        since = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"Grading signals since {since.isoformat()}")

    creds = get_mt5_credentials()
    adapter = MT5Adapter(creds)
    adapter.connect()
    try:
        graded = grade_all(adapter, REPO_ROOT, cfg, since)
    finally:
        adapter.disconnect()

    print(format_report(graded))
    return 0


if __name__ == "__main__":
    sys.exit(main())
