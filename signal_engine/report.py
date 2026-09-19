"""Writes REPORT.md — reliability diagram + equity curve PNGs, cost
sensitivity table, per-year/regime breakdown, and the Limitations section
the spec requires as a first-class deliverable, not an afterthought.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from signal_engine.backtest import BacktestReport
from signal_engine.config import SignalEngineConfig

DATA_YEARS_AVAILABLE = 3.2  # see data_loader.py docstring — MT5 demo history ceiling


def _plot_reliability(report: BacktestReport, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    for fold in report.folds:
        xs = [b.mean_predicted for b in fold.reliability if b.count > 0]
        ys = [b.observed_rate for b in fold.reliability if b.count > 0]
        ax.plot(xs, ys, marker="o", alpha=0.6, label=f"fold {fold.fold_index}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed hit rate")
    ax.set_title(f"{report.symbol} reliability diagram")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_equity_curve(report: BacktestReport, out_path: Path) -> None:
    all_trades: list[float] = []
    for fold in report.folds:
        all_trades.extend(fold.trade_r)
    fig, ax = plt.subplots(figsize=(7, 4))
    if all_trades:
        equity = np.cumsum(all_trades)
        ax.plot(equity)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative R")
    ax.set_title(f"{report.symbol} equity curve (walk-forward, all folds concatenated)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fold_table(report: BacktestReport) -> str:
    lines = [
        "| fold | test start | test end | signals | hit rate | mean R | expectancy R | max DD (R) | Sharpe | Brier | ECE | ambiguity rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for f in report.folds:
        lines.append(
            f"| {f.fold_index} | {f.test_start_time:%Y-%m-%d} | {f.test_end_time:%Y-%m-%d} | {f.n_signals} | "
            f"{f.hit_rate:.2%} | {f.mean_r:.3f} | {f.expectancy_r:.3f} | {f.max_drawdown_r:.2f} | {f.sharpe:.2f} | "
            f"{f.brier:.4f} | {f.ece:.4f} | {f.ambiguity_rate:.2%} |"
        )
    return "\n".join(lines)


def _aggregate(report: BacktestReport) -> dict:
    signals = sum(f.n_signals for f in report.folds)
    all_r = [r for f in report.folds for r in f.trade_r]
    hits = sum(1 for f in report.folds for r in f.trade_r if r > 0)
    return {
        "total_signals": signals,
        "hit_rate": (hits / len(all_r)) if all_r else float("nan"),
        "mean_r": float(np.mean(all_r)) if all_r else float("nan"),
        "expectancy": float(np.mean([f.expectancy_r for f in report.folds if f.n_signals > 0])) if signals else float("nan"),
        "max_drawdown_r": min((f.max_drawdown_r for f in report.folds), default=0.0),
    }


def _cost_table(report: BacktestReport) -> str:
    lines = ["| cost multiplier | mean expectancy (R) |", "|---|---|"]
    for mult, exp in sorted(report.cost_sensitivity.items()):
        val = "no signals gated in (n/a)" if exp != exp else f"{exp:.4f}"  # NaN check
        lines.append(f"| {mult}x | {val} |")
    return "\n".join(lines)


def _year_regime_tables(report: BacktestReport) -> tuple[str, str]:
    per_year: dict = {}
    per_regime: dict = {}
    for f in report.folds:
        for yr, stats in f.per_year.items():
            per_year.setdefault(yr, []).append(stats)
        for reg, stats in f.per_regime.items():
            per_regime.setdefault(reg, []).append(stats)

    year_lines = ["| year | signals | hit rate | mean R |", "|---|---|---|---|"]
    for yr in sorted(per_year):
        stats_list = per_year[yr]
        n = sum(s["n_signals"] for s in stats_list)
        hr = np.average([s["hit_rate"] for s in stats_list], weights=[s["n_signals"] for s in stats_list])
        mr = np.average([s["mean_r"] for s in stats_list], weights=[s["n_signals"] for s in stats_list])
        year_lines.append(f"| {yr} | {n} | {hr:.2%} | {mr:.3f} |")

    regime_lines = ["| regime | signals | hit rate | mean R |", "|---|---|---|---|"]
    for reg in sorted(per_regime):
        stats_list = per_regime[reg]
        n = sum(s["n_signals"] for s in stats_list)
        hr = np.average([s["hit_rate"] for s in stats_list], weights=[s["n_signals"] for s in stats_list])
        mr = np.average([s["mean_r"] for s in stats_list], weights=[s["n_signals"] for s in stats_list])
        regime_lines.append(f"| {reg} | {n} | {hr:.2%} | {mr:.3f} |")

    n_regimes_with_signal = len(per_regime)
    flag = ""
    if n_regimes_with_signal <= 1 and per_regime:
        flag = "\n\n**Flag:** all signals came from a single regime bucket — this strategy has not been shown to generalize across market conditions.\n"

    return "\n".join(year_lines) + flag, "\n".join(regime_lines)


def write_report(reports: list[BacktestReport], cfg: SignalEngineConfig) -> Path:
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = cfg.report_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    sections = ["# Signal Engine Report\n"]
    sections.append(
        "This report is generated by `scripts/run_signal_engine.py --train`. "
        "Read the Limitations section before treating any number here as tradeable.\n"
    )

    for report in reports:
        agg = _aggregate(report)
        reliability_path = assets_dir / f"{report.symbol}_reliability.png"
        equity_path = assets_dir / f"{report.symbol}_equity.png"
        _plot_reliability(report, reliability_path)
        _plot_equity_curve(report, equity_path)
        year_table, regime_table = _year_regime_tables(report)

        sections.append(f"## {report.symbol}\n")
        sections.append(
            f"Data: {report.n_bars} H1 bars, {report.data_start:%Y-%m-%d} to {report.data_end:%Y-%m-%d} "
            f"(~{(report.data_end - report.data_start).days / 365.25:.1f} years).\n"
        )
        sections.append("### Headline (aggregate across all walk-forward folds)\n")
        sections.append(
            f"- Total signals: **{agg['total_signals']}**\n"
            f"- Hit rate: **{agg['hit_rate']:.2%}**\n"
            f"- Mean R per trade: **{agg['mean_r']:.3f}**\n"
            f"- Mean expectancy after costs: **{agg['expectancy']:.3f} R**\n"
            f"- Worst fold max drawdown: **{agg['max_drawdown_r']:.2f} R**\n"
        )
        if agg["total_signals"] == 0:
            sections.append(
                "\n**No fold produced any signal above the probability/expectancy threshold. "
                "This is the model reporting no tradeable edge was found — a valid and honest "
                "outcome, not a bug to be tuned away.**\n"
            )
        elif not (agg["expectancy"] > 0):
            sections.append(
                "\n**Aggregate expectancy after costs is not positive. Per the spec's honesty "
                "requirement: the honest calibrated edge here is zero or negative. Do not trade "
                "this configuration.**\n"
            )

        sections.append("### Per-fold detail\n")
        sections.append(_fold_table(report) + "\n")

        sections.append("### Calibration\n")
        sections.append(f"![reliability diagram](report_assets/{report.symbol}_reliability.png)\n")

        sections.append("### Equity curve\n")
        sections.append(f"![equity curve](report_assets/{report.symbol}_equity.png)\n")

        sections.append("### Cost sensitivity\n")
        sections.append(_cost_table(report) + "\n")
        cost_exps = list(report.cost_sensitivity.values())
        if cost_exps and all(e != e for e in cost_exps):  # all NaN
            sections.append(
                "\nNo signals were gated in at *any* cost multiplier, including 0.5x (lower costs). "
                "That means the **probability threshold** is what's binding here, not the cost "
                "assumptions — even cutting costs in half didn't unlock a single signal.\n"
            )

        sections.append("### Per-year breakdown\n")
        sections.append(year_table + "\n")

        sections.append("### Per-regime breakdown\n")
        sections.append(regime_table + "\n")

        sections.append("### Null test (shuffled labels)\n")
        if report.null_test_expectancy != report.null_test_expectancy:  # NaN check
            sections.append(
                "Shuffled-label model gated in **zero signals** on the null-test fold — nothing to "
                "evaluate here, but consistent with (not contradicting) the real model also gating in "
                "no signals on that fold: the probability threshold, not the null test, is what's "
                "binding.\n"
            )
        else:
            sections.append(
                f"Expectancy after shuffling training labels: **{report.null_test_expectancy:.4f} R**. "
                "This should be near zero; if it isn't, the pipeline has a leak.\n"
            )

    sections.append("## Limitations\n")
    sections.append(
        f"- **Data depth: {DATA_YEARS_AVAILABLE:.1f} years, not the 5+ years the spec asked for.** "
        "The MT5 demo server (`MetaQuotes-Demo`) caps `copy_rates_from_pos` at 20,000 H1 bars "
        "regardless of the count requested — a broker/server-side history retention limit, "
        "confirmed identical across all three pairs (all start 2023-06-29/30). This likely means "
        "fewer full volatility/rate-cycle regimes are represented than the spec intended, and any "
        "per-year/per-regime breakdown above has fewer independent years to draw on than is ideal. "
        "To close this gap, pull deeper history from a dedicated data vendor (e.g. Dukascopy tick "
        "data resampled to H1, or a paid provider) rather than a demo trading account.\n"
        "- **Spread/slippage cost model is a live snapshot, not historical.** H1 OHLC bars carry no "
        "historical bid/ask spread; the cost model uses the current MT5 spread as a proxy for every "
        "historical bar. Spreads widen materially around news and session opens/closes, so realized "
        "live costs will likely exceed this estimate at times — treat the reported expectancy as an "
        "upper bound, not a guarantee.\n"
        "- **Ambiguous-bar resolution.** When both TP and SL fall inside the same future bar, the "
        "true intra-bar order is unknowable from OHLC alone; this is resolved conservatively as a "
        "loss. See the per-fold ambiguity rate above — if it's high for a given pair/config, results "
        "are more uncertain than the headline numbers suggest.\n"
        "- **Regime classification is a simple heuristic** (ATR percentile x EMA-spread trend "
        "threshold), not a validated market-regime model. Treat per-regime breakdowns as descriptive, "
        "not as a rigorous regime-switching claim.\n"
        "- **Data volume needed to detect calibration degradation:** a reliability bucket needs "
        "roughly 30-50 outcomes before its observed rate is a meaningful estimate (versus noise). At "
        "this system's typical signal rate, that implies waiting for at least several dozen live "
        "signals per pair — likely 2-4 months of paper trading — before comparing live calibration "
        "against the backtest numbers above with any confidence.\n"
        "- **Backtest results do not predict live results.** Before risking real capital: run this "
        "system in paper mode long enough to accumulate the sample size above, recompute the "
        "reliability diagram and Brier score on live paper trades, and confirm they land in the same "
        "range as backtest. If live calibration is materially worse, the model has degraded or the "
        "backtest overstated the edge — do not proceed to live capital until that's resolved.\n"
    )

    report_path = cfg.report_dir / "REPORT.md"
    report_path.write_text("\n".join(sections), encoding="utf-8")
    return report_path
