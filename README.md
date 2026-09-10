# J.A.R.V.I.S.

Two things, sharing one repo:

1. **`trading_bot/`** — a forex trading bot for MetaTrader 5. EMA-crossover + RSI-filtered
   signals, ATR-based stop loss / take profit, equity-based position sizing, and a daily
   drawdown circuit breaker. Defaults to **paper (simulated) mode**.
2. **`assistant/`** — a Claude-powered personal assistant/orchestrator with real tools
   (files, shell commands, memory, control of the trading bot) and the ability to spawn
   focused subagents for bounded tasks.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Fill in `.env`:

- `ANTHROPIC_API_KEY` — required for the assistant.
- `MT5_LOGIN` / `MT5_PASSWORD` / `MT5_SERVER` — required for the trading bot. Get these from
  your broker's MT5 account (a **demo account works identically** for paper-mode testing).
- `MT5_TERMINAL_PATH` — only needed if the `MetaTrader5` Python package can't auto-locate your
  installed terminal.

## Running the assistant

```bash
python scripts/run_assistant.py                # interactive REPL
python scripts/run_assistant.py "what's my trading bot's status?"   # one-shot
```

It can read/write files, run shell commands, remember facts across sessions, start/stop/check
the trading bot, and delegate subtasks to `researcher` / `coder` / `general` subagents.

**`run_command` executes real shell commands on this machine.** There's a denylist for
obviously catastrophic whole-drive commands and every command is logged to
`state/command_audit.log`, but this is a safety net, not a sandbox — it's exactly as
powerful as you typing the command yourself. Review the audit log periodically.

## Running the trading bot

```bash
python scripts/run_trading_bot.py --status   # account + open positions, then exit
python scripts/run_trading_bot.py --once     # one signal-check cycle, then exit
python scripts/run_trading_bot.py            # poll loop (Ctrl+C to stop)
```

Strategy and risk parameters live in `config/config.yaml`.

### Live trading is gated on purpose

By default `execution.live_trading: false` in `config/config.yaml` — every signal is logged to
`state/trades.jsonl` as a simulated trade, no order ever reaches your broker.

To arm real order execution, **two independent things** must both be true:

1. `execution.live_trading: true` in `config/config.yaml`
2. The environment variable `JARVIS_CONFIRM_LIVE=YES_I_UNDERSTAND_THE_RISK` set in `.env`

Either one alone is not enough — this is intentional, so a single accidental config change or
leaked env var can't put real money at risk by itself. The assistant cannot set the env var for
you; it's outside its tool surface by design.

Before ever going live: run against a **demo MT5 account** for a meaningful stretch first, and
review `risk_per_trade_pct` / `max_daily_loss_pct` / `max_open_positions` in
`config/config.yaml` — they are the only things standing between a bad signal and real losses.

## Tests

```bash
pytest
```

Covers position sizing math, the daily drawdown circuit breaker, indicator/signal logic, the
paper/live trade execution flow, the assistant's tool registry (including the command
denylist) and memory persistence, and the hand-rolled tool-use loop in the orchestrator and
subagents (via a fake Anthropic client — no API key or network access needed). Runs on every
push via GitHub Actions (`.github/workflows/tests.yml`).

## Architecture notes

- `trading_bot/mt5_adapter.py` wraps the raw `MetaTrader5` package calls (connect, fetch rates,
  place/close orders) and fails loudly instead of returning `None` on error.
- `trading_bot/risk_manager.py` sizes positions off account equity and the broker's own
  tick value/size for the symbol, so lot sizes stay correct across different symbols and
  brokers without hardcoded pip values.
- `assistant/orchestrator.py` runs a manual Claude tool-use loop (see the Anthropic Messages
  API docs) rather than the beta tool-runner helper, so the tool surface (files, shell,
  trading control, subagents) is fully explicit and easy to audit.
- `assistant/subagents.py` caps delegation at depth 1 — a spawned subagent cannot itself spawn
  subagents — to keep cost and runaway-loop risk bounded.
