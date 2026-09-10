from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

TIMEFRAME_NAMES = (
    "M1", "M2", "M3", "M4", "M5", "M6", "M10", "M12", "M15", "M20", "M30",
    "H1", "H2", "H3", "H4", "H6", "H8", "H12", "D1", "W1", "MN1",
)


@dataclass
class MT5Credentials:
    login: int
    password: str
    server: str
    terminal_path: str | None


def load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    path = path or (REPO_ROOT / "config" / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env() -> None:
    load_dotenv(REPO_ROOT / ".env")


def get_mt5_credentials() -> MT5Credentials:
    load_env()
    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")
    if not login or not password or not server:
        raise RuntimeError(
            "MT5_LOGIN, MT5_PASSWORD and MT5_SERVER must be set in .env "
            "before the trading bot can connect to MetaTrader 5."
        )
    return MT5Credentials(
        login=int(login),
        password=password,
        server=server,
        terminal_path=os.environ.get("MT5_TERMINAL_PATH") or None,
    )


def live_trading_confirmed() -> bool:
    """Hard safety gate: real orders require this exact env var value."""
    return os.environ.get("JARVIS_CONFIRM_LIVE") == "YES_I_UNDERSTAND_THE_RISK"
