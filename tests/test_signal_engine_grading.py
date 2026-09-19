from __future__ import annotations

import math

import pandas as pd

from signal_engine.grading import Graded, format_report, resolve_from_bars


def _bars(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"high": h, "low": l, "close": c} for h, l, c in rows])


def test_resolve_from_bars_no_bars_yet_is_still_open():
    outcome, r = resolve_from_bars(_bars([]), "long", 100, 99, 102, horizon_bars=5)
    assert outcome == "still_open"
    assert r is None


def test_resolve_from_bars_long_hits_tp():
    bars = _bars([(100.5, 99.8, 100.2), (102.5, 100, 102)])
    outcome, r = resolve_from_bars(bars, "long", 100, 99, 102, horizon_bars=5)
    assert outcome == "tp"
    assert math.isclose(r, 2.0)


def test_resolve_from_bars_long_hits_sl():
    bars = _bars([(100.2, 98.5, 99)])
    outcome, r = resolve_from_bars(bars, "long", 100, 99, 102, horizon_bars=5)
    assert outcome == "sl"
    assert r == -1.0


def test_resolve_from_bars_short_flips_direction():
    bars = _bars([(100.5, 97.5, 98)])  # low<=98 -> TP for a short; high stays below the 101 stop
    outcome, r = resolve_from_bars(bars, "short", 100, 101, 98, horizon_bars=5)
    assert outcome == "tp"


def test_resolve_from_bars_ambiguous_bar_is_a_loss():
    bars = _bars([(103, 98, 100)])  # both tp(>=102) and sl(<=99) touched
    outcome, r = resolve_from_bars(bars, "long", 100, 99, 102, horizon_bars=5)
    assert outcome == "ambiguous"
    assert r == -1.0


def test_resolve_from_bars_timeout_after_horizon():
    bars = _bars([(100.3, 99.5, 100.1), (100.4, 99.6, 100.2)])
    outcome, r = resolve_from_bars(bars, "long", 100, 99, 102, horizon_bars=2)
    assert outcome == "timeout"
    assert math.isclose(r, (100.2 - 100) / 1.0)


def test_resolve_from_bars_no_horizon_never_times_out_stays_open():
    bars = _bars([(100.3, 99.5, 100.1)])
    outcome, r = resolve_from_bars(bars, "long", 100, 99, 102, horizon_bars=None)
    assert outcome == "still_open"


def test_format_report_handles_empty_input():
    report = format_report([])
    assert "Nothing resolved yet" in report


def test_format_report_includes_hit_rate_and_mean_r():
    graded = [
        Graded("trading_bot", "EURUSD", "BUY", pd.Timestamp("2026-01-01", tz="UTC"), 1.1, 1.09, 1.12, "tp", 2.0),
        Graded("trading_bot", "EURUSD", "BUY", pd.Timestamp("2026-01-02", tz="UTC"), 1.1, 1.09, 1.12, "sl", -1.0),
    ]
    report = format_report(graded)
    assert "Hit rate: 50.0%" in report
    assert "Mean R: 0.500" in report
