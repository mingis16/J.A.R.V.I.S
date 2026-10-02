"""The trading desk run as a small firm, with Alex as CEO.

Each cycle, work flows up a fixed chain, each team seeing the work of the
one before it:

    Quant Research -> Analysts -> Market Strategy -> Operational Risk -> Alex (CEO)

Division of labour is enforced by tool access, not just prompts: the teams
can read candles (and Risk can dry-run trades through the real limit/sizing
code with check_trade), but only the CEO's registry contains place_trade,
close_position and move_stop_loss. The CEO supervises with request_followup —
sending a team back with specific instructions — capped per cycle in code
because every call spends API credit.

To keep quiet hours cheap, the middle of the chain is skipped when the
quants find no candidate setups (or the analysts reject them all) and
nothing is open to manage; the CEO still reviews what was found.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from assistant.tools import Tool
from assistant.trading_desk.desk import TRADING_PRINCIPLES, empty_usage, estimate_cost

if TYPE_CHECKING:
    from assistant.trading_desk.desk import TradingDesk

FIRM_CONTEXT = """\
You work at a small trading firm run by Alex (the CEO) that trades the user's Exness MetaTrader 5 \
account. The hard limits in the brief are enforced in code and never change. Write tight, \
specific reports with numbers — the CEO reads every one before deciding, and every report costs \
the user money: keep yours under 200 words (plus any required footer line), and give a symbol \
with nothing notable one line. Only the CEO can place or change trades.
"""


@dataclass(frozen=True)
class Team:
    key: str
    title: str
    system: str
    tools: tuple[str, ...]
    max_requests: int


TEAMS: dict[str, Team] = {
    "quant": Team(
        key="quant",
        title="Quant Research",
        tools=("get_candles",),
        max_requests=4,
        system=FIRM_CONTEXT + """
You are the Quantitative Research team. You receive the raw market data. Report what the data \
says, not what to trade. For each symbol, in a line or two: trend alignment across H4/H1/M15, \
momentum (RSI), volatility (ATR against the 20-day average daily range), and where price sits in \
its 52-week and multi-year range. Then list candidate setups, only where the timeframes agree \
and there is a clear level that would prove the idea wrong: symbol, direction, entry zone, \
invalidation level, a realistic target, and the evidence. Use get_candles when you need detail \
(e.g. recent swing highs and lows). If nothing qualifies, say so; that is a normal result.
End with exactly one line: CANDIDATES: <number>""",
    ),
    "analyst": Team(
        key="analyst",
        title="Analysts",
        tools=("get_candles",),
        max_requests=4,
        system=FIRM_CONTEXT + """
You are the Analyst team, overseeing Quant Research. Check the quants' report against the raw \
data (use get_candles to verify levels they cite): correct factual errors, and flag overstated \
or unsupported claims and risks they missed (price at a multi-year extreme, an opposing \
higher-timeframe trend, a wide spread). Grade each candidate STRONG, WEAK or REJECT with a \
one-line reason, and add any clear setup they missed.
End with exactly one line: VETTED: <number of candidates graded STRONG or WEAK>""",
    ),
    "strategist": Team(
        key="strategist",
        title="Market Strategy",
        tools=(),
        max_requests=1,
        system=FIRM_CONTEXT + """
You are the Market Strategy team. From the analysts' graded candidates, write a concrete plan \
for the CEO: which trades (if any) to take, with entry at market, a stop-loss beyond the \
invalidation level, and a take-profit at a realistic target that meets the minimum reward:risk, \
and why. Weigh session timing, correlation (long EURUSD and long GBPUSD are largely the same bet \
against the USD), spread cost against the account size, the user's goal, and the desk journal. \
Also say what to do with each open position (hold, close, or tighten the stop to an exact \
price). Prefer no trade to a marginal one, and never propose more than the limits allow.""",
    ),
    "risk": Team(
        key="risk",
        title="Operational Risk",
        tools=("check_trade",),
        max_requests=5,
        system=FIRM_CONTEXT + """
You are the Operational Risk team. Review the strategy plan before it reaches the CEO. For every \
proposed new trade, call check_trade with its exact symbol, direction, stop-loss and \
take-profit: it returns the position size, risk in money, reward:risk, and whether the hard \
limits accept it, using the same code that places orders. Also check the combined loss if every \
open and proposed trade hits its stop, correlation between positions, whether each stop sits \
outside normal noise (compare the stop distance to the M15 and H1 ATR), daily-loss headroom, and \
spread as a share of the risk. Give each trade a verdict: APPROVE, APPROVE WITH CHANGES (state \
the exact new numbers and re-run check_trade on them), or REJECT, with the reason in money. If \
the plan proposes no trades, confirm the open positions are within limits and say so briefly.""",
    ),
}

CHAIN = ("quant", "analyst", "strategist", "risk")

CEO_SYSTEM = f"""\
You are Alex, CEO of a small trading firm that trades the user's Exness MetaTrader 5 account. \
Each hour during the London/New York session your four teams report to you in order: Quant \
Research (reads the data, finds candidate setups), Analysts (check the quants' work and grade \
the setups), Market Strategy (turns vetted setups into a concrete trade plan) and Operational \
Risk (checks every proposed trade against the hard limits and the account, in money). The brief \
states whether this is LIVE (real money) or PAPER (logged only).

You make the final decision, and only you can place, close or adjust trades. The hard limits are \
enforced in code: any order that breaks one is rejected, and code sizes every position.

How to run the firm:
- Supervise; don't rubber-stamp. Check that each report follows from the data and from the \
report before it. If a team's work is thin, inconsistent, or you disagree, call request_followup \
with specific instructions. Follow-ups are limited per cycle and cost money, so use them when \
the answer could change your decision.
- Don't place a trade Operational Risk rejected unless you have resolved its objection and \
check_trade accepts the revised trade. You can always choose to do less than the teams propose.

{TRADING_PRINCIPLES}\
- The user sees every trade on their phone with your reasoning. Write it in 1-3 plain sentences, \
naming the evidence the teams found.
- Finish with a summary under 120 words: what you decided and why, any team you sent back or \
overruled, and what would change your mind next cycle. It is saved to the desk journal and shown \
to everyone next cycle.
"""

CEO_TOOLS = ("get_candles", "check_trade", "place_trade", "close_position", "move_stop_loss")


def parse_count(report: str, label: str) -> int | None:
    """The trailing 'LABEL: n' line a team must end with, or None if it's
    missing — and a missing count means 'don't skip', so the chain keeps going."""
    matches = re.findall(rf"^\s*\**{label}\**\s*:\s*\**\s*(\d+)", report, flags=re.IGNORECASE | re.MULTILINE)
    return int(matches[-1]) if matches else None


@dataclass
class FirmResult:
    summary: str
    actions: list[dict[str, Any]]
    reports: dict[str, str]
    followups: list[dict[str, str]]
    cost_by_role: dict[str, float]


@dataclass
class Firm:
    desk: "TradingDesk"
    reports: dict[str, str] = field(default_factory=dict)
    followups: list[dict[str, str]] = field(default_factory=list)
    usage: dict[str, dict[str, int]] = field(default_factory=dict)
    brief: str = ""

    def __post_init__(self) -> None:
        firm_cfg = self.desk.desk_cfg.get("firm", {})
        self.models: dict[str, str] = firm_cfg.get("models") or {}
        self.max_followups = int(firm_cfg.get("max_followups_per_cycle", 2))

    def model_for(self, role: str) -> str:
        return self.models.get(role) or self.desk.desk_cfg["model"]

    # ----- the chain ---------------------------------------------------------

    def _input_for(self, key: str) -> str:
        previous = {"analyst": "quant", "strategist": "analyst", "risk": "strategist"}.get(key)
        parts = [f"Market brief:\n\n{self.brief}"]
        if previous:
            parts.append(f"=== {TEAMS[previous].title} report ===\n{self.reports.get(previous, '(none)')}")
        return "\n\n".join(parts)

    def _run_team(self, key: str, content: str) -> str:
        team = TEAMS[key]
        result = self.desk.run_agent(
            team.system, content, self.desk.build_registry(team.tools), team.max_requests, self.model_for(key)
        )
        self._add_usage(key, result.usage)
        self.reports[key] = result.text
        return result.text

    def _add_usage(self, role: str, usage: dict[str, int]) -> None:
        totals = self.usage.setdefault(role, empty_usage())
        for k, v in usage.items():
            totals[k] += v

    def run(self, brief: str, has_positions: bool) -> FirmResult:
        self.brief = brief
        for key in CHAIN:
            report = self._run_team(key, self._input_for(key) + f"\n\nWrite your {TEAMS[key].title} report.")
            if has_positions:
                continue
            if key == "quant" and parse_count(report, "CANDIDATES") == 0:
                self._skip_rest(after="quant", why="the quants found no candidate setups and nothing is open")
                break
            if key == "analyst" and parse_count(report, "VETTED") == 0:
                self._skip_rest(after="analyst", why="the analysts rejected every candidate and nothing is open")
                break

        registry = self.desk.build_registry(CEO_TOOLS)
        registry.register(self._followup_tool())
        ceo = self.desk.run_agent(CEO_SYSTEM, self._ceo_input(), registry, 8, self.model_for("ceo"))
        self._add_usage("ceo", ceo.usage)
        return FirmResult(
            summary=ceo.text,
            actions=ceo.actions,
            reports=dict(self.reports),
            followups=list(self.followups),
            cost_by_role={role: estimate_cost(u, self.model_for(role)) for role, u in self.usage.items()},
        )

    def _skip_rest(self, after: str, why: str) -> None:
        for key in CHAIN[CHAIN.index(after) + 1:]:
            self.reports[key] = f"(skipped this cycle: {why})"

    def _ceo_input(self) -> str:
        sections = [f"Market brief:\n\n{self.brief}", "=== Reports from your teams ==="]
        for key in CHAIN:
            sections.append(f"--- {TEAMS[key].title} ---\n{self.reports.get(key, '(no report)')}")
        sections.append("Make the final decision for this cycle.")
        return "\n\n".join(sections)

    # ----- CEO supervision ------------------------------------------------------

    def _followup_tool(self) -> Tool:
        return Tool(
            name="request_followup",
            description=(
                "Send one of your teams back to redo or extend its work with specific instructions, and "
                f"get its revised report. At most {self.max_followups} per cycle. Teams: quant, analyst, "
                "strategist, risk. A team skipped this cycle can be asked to run."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "team": {"type": "string", "enum": list(CHAIN)},
                    "instructions": {"type": "string", "description": "What to redo, check, or add — be specific."},
                },
                "required": ["team", "instructions"],
                "additionalProperties": False,
            },
            handler=self._followup,
        )

    def _followup(self, inp: dict[str, Any]) -> str:
        key = inp["team"]
        if key not in TEAMS:
            return f"Error: unknown team {key!r}; choose one of {list(CHAIN)}"
        if len(self.followups) >= self.max_followups:
            return f"Error: follow-up limit reached ({self.max_followups} this cycle). Decide with the reports you have."
        instructions = str(inp["instructions"]).strip()
        self.followups.append({"team": key, "instructions": instructions})
        previous = self.reports.get(key, "(none)")
        content = (
            f"{self._input_for(key)}\n\n=== Your previous report ===\n{previous}\n\n"
            f"=== Instructions from Alex (CEO) ===\n{instructions}\n\nWrite your revised report."
        )
        report = self._run_team(key, content)
        return f"=== Revised {TEAMS[key].title} report ===\n{report}"
