from __future__ import annotations

import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import anthropic

from assistant.daemon.proposals import ProposalLog
from assistant.daemon.routines import ROUTINES, RoutineContext, RoutineFn
from assistant.memory import Memory

logger = logging.getLogger("assistant.daemon")


@dataclass
class RoutineSpec:
    name: str
    fn: RoutineFn
    interval_s: float | None = None
    daily_at: str | None = None  # "HH:MM", local time


def is_due(spec: RoutineSpec, last_run: Optional[datetime], now: datetime) -> bool:
    """Pure scheduling decision — no I/O, easy to unit test."""
    if spec.interval_s is not None:
        return last_run is None or (now - last_run).total_seconds() >= spec.interval_s
    if spec.daily_at is not None:
        hh, mm = (int(x) for x in spec.daily_at.split(":"))
        scheduled_today = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if now < scheduled_today:
            return False
        return last_run is None or last_run.date() < now.date()
    return False


def build_specs(cfg: dict) -> list[RoutineSpec]:
    daemon_cfg = cfg.get("daemon", {})
    return [
        RoutineSpec(
            "trading_bot_health_check",
            ROUTINES["trading_bot_health_check"],
            interval_s=daemon_cfg.get("trading_bot_health_check_interval_s", 900),
        ),
        RoutineSpec(
            "dev_agent_routine",
            ROUTINES["dev_agent_routine"],
            interval_s=daemon_cfg.get("dev_agent_interval_s", 21600),
        ),
        RoutineSpec(
            "overnight_summary",
            ROUTINES["overnight_summary"],
            daily_at=daemon_cfg.get("overnight_summary_time", "07:00"),
        ),
    ]


class Scheduler:
    def __init__(self, specs: list[RoutineSpec], ctx: RoutineContext):
        self.specs = specs
        self.ctx = ctx
        self.last_run: dict[str, datetime] = {}
        self._shutdown = False

    def _handle_sigint(self, signum, frame) -> None:
        self._shutdown = True
        logger.info("Shutdown requested, will stop after this tick.")

    def _run_routine(self, spec: RoutineSpec, now: datetime) -> None:
        logger.info("Running routine: %s", spec.name)
        try:
            spec.fn(self.ctx)
        except Exception:
            logger.exception("Routine %s failed; will retry next interval.", spec.name)
        self.last_run[spec.name] = now

    def run_once(self) -> None:
        now = datetime.now()
        for spec in self.specs:
            self._run_routine(spec, now)

    def run_forever(self, tick_seconds: int = 5) -> None:
        signal.signal(signal.SIGINT, self._handle_sigint)
        logger.info("Daemon started with routines: %s", ", ".join(s.name for s in self.specs))
        while not self._shutdown:
            now = datetime.now()
            for spec in self.specs:
                if is_due(spec, self.last_run.get(spec.name), now):
                    self._run_routine(spec, now)
            for _ in range(tick_seconds):
                if self._shutdown:
                    break
                time.sleep(1)
        logger.info("Daemon stopped.")


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("assistant.daemon")
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)


def main() -> int:
    import argparse

    from trading_bot.config import REPO_ROOT, load_env, load_yaml_config

    parser = argparse.ArgumentParser(description="Alex background daemon")
    parser.add_argument(
        "--once", action="store_true", help="Run every enabled routine once immediately, then exit."
    )
    args = parser.parse_args()

    load_env()
    cfg = load_yaml_config()
    daemon_cfg = cfg.get("daemon", {})

    setup_logging(REPO_ROOT / daemon_cfg.get("log_path", "logs/daemon.log"))

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning(
            "ANTHROPIC_API_KEY is not set — trading_bot_health_check will still run, but "
            "dev_agent_routine and overnight_summary (which call Claude) will fail per-cycle."
        )

    memory = Memory(REPO_ROOT / cfg["assistant"]["memory_path"])
    proposal_log = ProposalLog(REPO_ROOT / daemon_cfg.get("proposals_path", "logs/proposals.jsonl"))
    client = anthropic.Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
    ctx = RoutineContext(
        repo_root=REPO_ROOT,
        cfg=cfg,
        memory=memory,
        proposal_log=proposal_log,
        client=client,
    )

    scheduler = Scheduler(build_specs(cfg), ctx)
    if args.once:
        scheduler.run_once()
    else:
        scheduler.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
