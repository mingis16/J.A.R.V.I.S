from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import MetaTrader5 as mt5
import pandas as pd

from trading_bot.config import MT5Credentials, TIMEFRAME_NAMES

logger = logging.getLogger("trading_bot.mt5_adapter")

_TIMEFRAME_MAP = {name: getattr(mt5, f"TIMEFRAME_{name}") for name in TIMEFRAME_NAMES}

# Broker filling-mode support varies; try in this order until one is accepted.
_FILLING_MODES = (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN)


@dataclass
class OrderResult:
    success: bool
    retcode: int | None
    comment: str
    order_id: int | None
    price: float | None


class MT5Adapter:
    """Thin, explicit wrapper around the MetaTrader5 terminal API.

    Every method raises RuntimeError with mt5.last_error() context on
    failure rather than silently returning None, so callers fail loud.
    """

    def __init__(self, creds: MT5Credentials):
        self._creds = creds
        self._connected = False

    def connect(self) -> None:
        kwargs: dict[str, Any] = dict(
            login=self._creds.login,
            password=self._creds.password,
            server=self._creds.server,
        )
        if self._creds.terminal_path:
            ok = mt5.initialize(self._creds.terminal_path, **kwargs)
        else:
            ok = mt5.initialize(**kwargs)
        if not ok:
            raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
        self._connected = True
        logger.info("Connected to MT5 account %s on %s", self._creds.login, self._creds.server)

    def disconnect(self) -> None:
        if self._connected:
            mt5.shutdown()
            self._connected = False

    def __enter__(self) -> "MT5Adapter":
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.disconnect()

    def get_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        if timeframe not in _TIMEFRAME_MAP:
            raise ValueError(f"Unknown timeframe '{timeframe}'. Valid: {sorted(_TIMEFRAME_MAP)}")
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"symbol_select('{symbol}') failed: {mt5.last_error()}")
        rates = mt5.copy_rates_from_pos(symbol, _TIMEFRAME_MAP[timeframe], 0, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates_from_pos returned no data: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df

    def get_symbol_info(self, symbol: str) -> Any:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbol_info('{symbol}') failed: {mt5.last_error()}")
        return info

    def get_tick(self, symbol: str) -> Any:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick('{symbol}') failed: {mt5.last_error()}")
        return tick

    def get_account_info(self) -> Any:
        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"account_info() failed: {mt5.last_error()}")
        return info

    def get_open_positions(self, symbol: str | None = None) -> list[Any]:
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        return list(positions) if positions is not None else []

    def close_position(self, position: Any, deviation: int = 20) -> OrderResult:
        symbol = position.symbol
        volume = position.volume
        is_buy = position.type == mt5.ORDER_TYPE_BUY
        tick = self.get_tick(symbol)
        price = tick.bid if is_buy else tick.ask
        order_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "deviation": deviation,
            "magic": position.magic,
            "comment": "jarvis-close",
            "type_time": mt5.ORDER_TIME_GTC,
        }
        return self._send_with_filling_fallback(request)

    def place_market_order(
        self,
        symbol: str,
        is_buy: bool,
        volume: float,
        sl: float | None,
        tp: float | None,
        deviation: int,
        magic: int,
        comment: str,
    ) -> OrderResult:
        tick = self.get_tick(symbol)
        price = tick.ask if is_buy else tick.bid
        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        return self._send_with_filling_fallback(request)

    def _send_with_filling_fallback(self, request: dict[str, Any]) -> OrderResult:
        last_result = None
        for filling in _FILLING_MODES:
            request["type_filling"] = filling
            result = mt5.order_send(request)
            if result is None:
                last_result = OrderResult(False, None, str(mt5.last_error()), None, None)
                continue
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return OrderResult(True, result.retcode, result.comment, result.order, result.price)
            last_result = OrderResult(False, result.retcode, result.comment, None, None)
            # Only retry with a different filling mode on an "unsupported filling" style error.
            if result.retcode not in (mt5.TRADE_RETCODE_INVALID_FILL,):
                break
        return last_result or OrderResult(False, None, "order_send returned no result", None, None)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
