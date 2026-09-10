from __future__ import annotations

import logging

from trading_bot.config import live_trading_confirmed
from trading_bot.mt5_adapter import MT5Adapter
from trading_bot.risk_manager import DailyDrawdownGuard, compute_position_size
from trading_bot.strategy import Signal
from trading_bot.trade_log import TradeLog

logger = logging.getLogger("trading_bot.executor")


class Executor:
    def __init__(
        self,
        adapter: MT5Adapter,
        trade_log: TradeLog,
        trading_cfg: dict,
        execution_cfg: dict,
    ):
        self.adapter = adapter
        self.trade_log = trade_log
        self.trading_cfg = trading_cfg
        self.execution_cfg = execution_cfg
        self.drawdown_guard = DailyDrawdownGuard(trading_cfg["max_daily_loss_pct"])

    def handle_signal(self, signal: Signal) -> str:
        if signal.action == "HOLD":
            return f"HOLD: {signal.reason}"

        symbol = self.trading_cfg["symbol"]
        account = self.adapter.get_account_info()
        day_key = self._today_key()
        ok, block_reason = self.drawdown_guard.check(account.equity, day_key)
        if not ok:
            logger.warning("Daily drawdown circuit breaker tripped: %s", block_reason)
            return f"BLOCKED: {block_reason}"

        open_positions = self.adapter.get_open_positions(symbol)
        if len(open_positions) >= self.trading_cfg["max_open_positions"]:
            return f"BLOCKED: max_open_positions ({self.trading_cfg['max_open_positions']}) reached"

        symbol_info = self.adapter.get_symbol_info(symbol)
        sl_distance = signal.atr_value * self.trading_cfg["atr_sl_multiplier"]
        tp_distance = signal.atr_value * self.trading_cfg["atr_tp_multiplier"]
        is_buy = signal.action == "BUY"
        sl_price = signal.close_price - sl_distance if is_buy else signal.close_price + sl_distance
        tp_price = signal.close_price + tp_distance if is_buy else signal.close_price - tp_distance

        sizing = compute_position_size(
            equity=account.equity,
            risk_per_trade_pct=self.trading_cfg["risk_per_trade_pct"],
            sl_distance_price=sl_distance,
            tick_value=symbol_info.trade_tick_value,
            tick_size=symbol_info.trade_tick_size,
            volume_min=symbol_info.volume_min,
            volume_max=symbol_info.volume_max,
            volume_step=symbol_info.volume_step,
        )
        if sizing.blocked_reason:
            return f"BLOCKED: position sizing failed: {sizing.blocked_reason}"

        live = self.execution_cfg["live_trading"]
        if live and not live_trading_confirmed():
            logger.warning(
                "live_trading=true in config but JARVIS_CONFIRM_LIVE is not set correctly; "
                "falling back to paper mode for this signal."
            )
            live = False

        if not live:
            record = self.trade_log.new_record(
                symbol=symbol,
                action=signal.action,
                mode="paper",
                lots=sizing.lots,
                price=signal.close_price,
                sl=sl_price,
                tp=tp_price,
                reason=signal.reason,
                status="simulated",
            )
            self.trade_log.record(record)
            return f"PAPER {signal.action} {sizing.lots} lots {symbol} @ {signal.close_price:.5f} (sl={sl_price:.5f} tp={tp_price:.5f})"

        result = self.adapter.place_market_order(
            symbol=symbol,
            is_buy=is_buy,
            volume=sizing.lots,
            sl=sl_price,
            tp=tp_price,
            deviation=self.execution_cfg["deviation_points"],
            magic=self.execution_cfg["magic_number"],
            comment=self.execution_cfg["order_comment"],
        )
        record = self.trade_log.new_record(
            symbol=symbol,
            action=signal.action,
            mode="live",
            lots=sizing.lots,
            price=result.price or signal.close_price,
            sl=sl_price,
            tp=tp_price,
            reason=signal.reason,
            order_id=result.order_id,
            status="filled" if result.success else f"failed:{result.retcode}:{result.comment}",
        )
        self.trade_log.record(record)
        if not result.success:
            logger.error("Live order failed: %s", result.comment)
            return f"LIVE ORDER FAILED: {result.comment}"
        return f"LIVE {signal.action} {sizing.lots} lots {symbol} @ {result.price} (order #{result.order_id})"

    @staticmethod
    def _today_key() -> str:
        from trading_bot.mt5_adapter import utc_now

        return utc_now().strftime("%Y-%m-%d")
