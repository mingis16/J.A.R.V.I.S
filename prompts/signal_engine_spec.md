# Prompt: Build a calibrated FX signal engine

You are a quantitative developer. Build a forex signal engine in Python. Read this
entire spec before writing code, and push back in writing on anything you believe is
statistically unsound rather than silently complying.

## The goal, stated correctly

The naive ask is "give me signals with an 80% chance of hitting take profit." That
target is misleading on its own: for a roughly driftless price process,

    P(TP hit before SL) ~= SL_distance / (TP_distance + SL_distance)

so an 80% hit rate is achievable simply by setting TP at one quarter of the stop
distance, and it carries zero edge. Do not optimize for hit rate.

Optimize for this instead:

1. **Calibration.** When the model says 80%, the realized hit rate on unseen data
   must be 75-85%. Measure with a reliability diagram, Brier score, and expected
   calibration error (ECE) in 10 buckets.
2. **Positive expectancy after costs.** Expectancy per trade in R units:
   `E = p*TP_R - (1-p)*1 - costs_R`, where costs include spread, commission and
   realistic slippage. Report expectancy, not win rate, as the headline metric.
3. Only emit a signal when BOTH the calibrated probability is above threshold AND
   expectancy after costs is positive.

## Data and hygiene

- Pairs and timeframe: make configurable; default EURUSD, GBPUSD, USDJPY on H1.
- Minimum 5 years of history. State the data source in the README.
- Bars must be timestamped in UTC and aligned; document how you handle the Sunday
  open, rollover gaps, and missing bars.
- **No lookahead.** Every feature at bar t uses only data closed at or before t.
  Write an explicit unit test that shifts all features and shows performance
  collapses to chance; if it does not, you have leakage.
- Labeling: triple-barrier method. For each entry candidate, label 1 if TP is touched
  before SL within the horizon, 0 if SL first, and exclude or separately handle
  timeouts. Use bar highs/lows, not closes. When both barriers fall inside one bar,
  resolve conservatively as a loss and note the ambiguity rate.

## Model

- Start with a transparent baseline: logistic regression or gradient boosting over
  ~10-30 features (ATR-normalized returns, realized vol, session, trend/range
  classifier, distance from session VWAP, day-of-week, spread regime). No deep
  learning until the baseline is beaten.
- Fit a probability calibrator (Platt or isotonic) on a validation split that is
  separate from both train and test.
- Volatility-scale all barriers: TP and SL expressed in ATR multiples, not fixed pips.

## Validation (this is the part that decides if it is real)

- **Walk-forward / purged k-fold with embargo.** Train on a window, test on the next
  unseen window, roll forward. Embargo bars around each split boundary so overlapping
  labels cannot leak.
- Report per-fold and aggregate: number of signals, hit rate, mean R, expectancy,
  max drawdown in R, Sharpe on the trade series, and calibration error.
- Include a cost sensitivity table at 0.5x, 1x and 2x assumed spread/slippage.
- Run a null test: shuffle labels and confirm the pipeline reports no edge.
- Report the strategy's performance separately per year and per volatility regime.
  A strategy that only worked in one regime must be flagged as such.

## Output contract

Each signal is a JSON object:

```json
{
  "timestamp_utc": "...", "pair": "EURUSD", "direction": "long",
  "entry": 1.0850, "stop": 1.0820, "take_profit": 1.0910,
  "atr_at_signal": 0.0031, "risk_reward": 2.0,
  "p_tp_calibrated": 0.62, "expectancy_R_after_costs": 0.19,
  "features_top": {"...": 0.0},
  "sample_size_in_bucket": 412,
  "regime": "trending_high_vol"
}
```

Never emit a probability without the out-of-sample sample size behind that bucket.

## Deliverables

1. Runnable code: data loader, feature builder, labeler, trainer, calibrator,
   walk-forward backtester, signal emitter.
2. A `REPORT.md` with the reliability diagram, equity curve in R, the cost
   sensitivity table, and per-year breakdown.
3. An explicit **Limitations** section: what would break this, how much data it needs
   to detect degradation, and the live-vs-backtest checks to run before risking money.

## Honesty requirements

- If the honest calibrated edge is small or zero, say so plainly in `REPORT.md`. A
  correct negative result is the successful outcome of this task.
- Do not tune thresholds on the test set. If you look at test results and change the
  model, that set is burned; say so and carve out a fresh one.
- Do not report a win rate without the accompanying average R and expectancy.
- State clearly that backtested results are not predictive of live results, and that
  the system must run in paper mode long enough to compare live calibration against
  backtest calibration before any real capital is used.
