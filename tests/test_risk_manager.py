from trading_bot.risk_manager import DailyDrawdownGuard, compute_position_size, round_to_step


def test_round_to_step():
    assert round_to_step(0.137, 0.01) == 0.13
    assert round_to_step(1.0, 0.1) == 1.0
    assert round_to_step(0.05, 0.0) == 0.05


def test_compute_position_size_basic():
    # EURUSD-like symbol: tick_size=0.00001, tick_value≈$1 per lot per tick at 5 decimals... use $10 per pip typical
    sizing = compute_position_size(
        equity=10_000,
        risk_per_trade_pct=1.0,
        sl_distance_price=0.0020,  # 20 pips
        tick_value=1.0,
        tick_size=0.00001,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    # risk_amount = 100; value_per_price_unit_per_lot = 1.0/0.00001 = 100_000
    # loss_per_lot = 0.0020 * 100_000 = 200; raw_lots = 100/200 = 0.5
    assert sizing.blocked_reason is None
    assert sizing.lots == 0.5
    assert sizing.risk_amount == 100.0


def test_compute_position_size_below_min_is_blocked():
    sizing = compute_position_size(
        equity=100,
        risk_per_trade_pct=0.1,
        sl_distance_price=0.0020,
        tick_value=1.0,
        tick_size=0.00001,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    assert sizing.lots == 0.0
    assert sizing.blocked_reason is not None


def test_compute_position_size_invalid_sl_distance():
    sizing = compute_position_size(
        equity=10_000,
        risk_per_trade_pct=1.0,
        sl_distance_price=0.0,
        tick_value=1.0,
        tick_size=0.00001,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    assert sizing.lots == 0.0
    assert "stop-loss" in sizing.blocked_reason


def test_daily_drawdown_guard_trips():
    guard = DailyDrawdownGuard(max_daily_loss_pct=3.0)
    ok, reason = guard.check(10_000, "2026-09-10")
    assert ok and reason is None

    ok, reason = guard.check(9_800, "2026-09-10")  # -2%, within cap
    assert ok and reason is None

    ok, reason = guard.check(9_600, "2026-09-10")  # -4%, exceeds 3% cap
    assert not ok and reason is not None


def test_daily_drawdown_guard_resets_on_new_day():
    guard = DailyDrawdownGuard(max_daily_loss_pct=3.0)
    guard.check(10_000, "2026-09-10")
    ok, _ = guard.check(9_000, "2026-09-10")  # -10%, blocked
    assert not ok

    ok, reason = guard.check(9_000, "2026-09-11")  # new day resets baseline
    assert ok and reason is None
