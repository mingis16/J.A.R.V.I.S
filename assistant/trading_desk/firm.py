"""The trading desk run as a firm, with Alex as CEO over nine intelligence
departments.

    Departments 1-8 (data)  ->  Director of Intelligence  ->  Analysts
        ->  Market Strategy  ->  9 Risk Intelligence  ->  Alex (CEO)  ->  execution

Where each department's input comes from (see desk.build_brief):
  1 Macro, 2 Geopolitical, 3 Fundamental, 6 Positioning: a daily web-search
    briefing (briefing.py), reused all day.
  4 Quant, 6 Sentiment (risk appetite), 7 Order flow (proxy), 8 Liquidity:
    computed by code every cycle (intel.py) — arithmetic, not opinion.
  5 Technical: multi-timeframe indicators and long-term ranges (desk.py).
  9 Risk: a code gate that decides before any LLM is paid (limits.py), plus
    the Risk Intelligence team, whose approval place_trade requires.

Division of labour is enforced by tool access and code, not just prompts:
teams can read candles, Risk Intelligence can dry-run trades through the
real limit/sizing code (check_trade), and only the CEO's registry contains
place_trade / close_position / move_stop_loss — and even then place_trade
refuses any trade Risk Intelligence didn't APPROVE this cycle. The CEO
supervises with request_followup, capped per cycle because every call costs.

Quiet hours stay cheap: when the Director finds no candidates (or the
analysts reject them all) and nothing is open, the middle of the chain is
skipped and the CEO reviews the intelligence report alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from assistant.tools import Tool
from assistant.trading_desk.desk import TRADING_PRINCIPLES, empty_usage, estimate_cost

if TYPE_CHECKING:
    from assistant.trading_desk.desk import TradingDesk

DEPARTMENTS = (
    "Macroeconomic", "Geopolitical", "Fundamental", "Quantitative",
    "Technical", "Sentiment", "Order flow", "Liquidity", "Risk",
)

FIRM_CONTEXT = """\
You work at a small trading firm run by Alex (the CEO) that trades the user's Exness MetaTrader 5 \
account. The hard limits and the Risk Intelligence code gate in the brief are enforced in code \
and never change. Write tight, specific reports with numbers — the CEO reads every one before \
deciding, and every report costs the user money. Only the CEO can place or change trades.
"""


@dataclass(frozen=True)
class Team:
    key: str
    title: str
    system: str
    tools: tuple[str, ...]
    max_requests: int


TEAMS: dict[str, Team] = {
    "intelligence": Team(
        key="intelligence",
        title="Intelligence Departments",
        tools=("get_candles",),
        max_requests=4,
        system=FIRM_CONTEXT + """
You are the Director of Intelligence. The brief already contains each department's data: the \
daily briefing (1 Macroeconomic, 2 Geopolitical, 3 Fundamental, 6 Positioning), cross-asset risk \
appetite (6 Sentiment), quant base rates from years of history (4 Quantitative), multi-timeframe \
technicals and long-term ranges (5 Technical), the order-flow proxy (7 — retail FX has no real \
order book, so treat it as weak evidence) and liquidity (8). Turn it into the departments' \
findings. Never invent data a department doesn't have; say "no view" instead.

Write, under 300 words:
DEPARTMENT FINDINGS
1. Macroeconomic: ...
2. Geopolitical: ...
3. Fundamental: ...
4. Quantitative: ... (if the base rate for the current trend state is near 50% or below the \
break-even, say plainly that trend alone is not an edge)
5. Technical: ...
6. Sentiment: ...
7. Order flow: ...
8. Liquidity: ...
CANDIDATE SETUPS — only where several departments agree and there is a clear invalidation \
level: symbol, direction, entry zone, invalidation level, realistic target, supporting \
departments, dissenting departments. Use get_candles if you need recent swing levels.
End with exactly one line: CANDIDATES: <number>""",
    ),
    "analyst": Team(
        key="analyst",
        title="Analysts",
        tools=("get_candles",),
        max_requests=4,
        system=FIRM_CONTEXT + """
You are the Analyst team, overseeing the Director of Intelligence. Check the intelligence report \
against the brief (use get_candles to verify levels it cites): correct factual errors, and flag \
overstated or unsupported claims and missed risks (a department contradicting the setup, event \
risk in the daily briefing, price at a multi-year extreme, a wide spread). Grade each candidate \
STRONG, WEAK or REJECT with a one-line reason; add any clear setup they missed. Under 200 words.
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
and why. Weigh event risk and timing from the daily briefing, correlation (long EURUSD and long \
GBPUSD are largely the same bet against the USD — the code blocks doubling a currency anyway), \
spread cost against the account size, the user's goal, and the desk journal. Also say what to do \
with each open position (hold, close, or tighten the stop to an exact price). Prefer no trade to \
a marginal one. Under 200 words.""",
    ),
    "risk": Team(
        key="risk",
        title="Risk Intelligence",
        tools=("check_trade",),
        max_requests=5,
        system=FIRM_CONTEXT + """
You are department 9, Risk Intelligence. You override every other department: no trade reaches \
the market without your APPROVE, and code enforces that. For every proposed new trade, call \
check_trade with its exact symbol, direction, stop-loss and take-profit — it returns the size, \
risk in money, reward:risk and whether the hard limits and code gate accept it, using the same \
code that places orders. Then judge what code can't: the combined loss if every open and \
proposed trade hits its stop, whether each stop sits outside normal noise (compare it to the \
M15 and H1 ATR), event risk from the daily briefing, spread as a share of the risk, and whether \
the evidence justifies the risk. If you'd approve with changes, re-run check_trade on the new \
numbers and approve those. Under 200 words.
End with one line per proposed trade:
VERDICT: <SYMBOL> <buy|sell> APPROVE <max risk_pct>   or   VERDICT: <SYMBOL> <buy|sell> REJECT
If the plan proposes no trades, end with: VERDICT: NONE""",
    ),
}

CHAIN = ("intelligence", "analyst", "strategist", "risk")

CEO_SYSTEM = f"""\
You are Alex, CEO of a trading firm that trades the user's Exness MetaTrader 5 account. Each hour \
during the London/New York session your organisation reports to you: the Director of \
Intelligence (findings from eight departments — Macroeconomic, Geopolitical, Fundamental, \
Quantitative, Technical, Sentiment, Order flow, Liquidity — plus candidate setups), the \
Analysts (who check that report and grade the setups), Market Strategy (a concrete trade plan), \
and department 9, Risk Intelligence (checks every proposed trade in money and issues the \
APPROVE/REJECT verdicts). The brief states whether this is LIVE (real money) or PAPER.

You make the final decision; only you can place, close or adjust trades. Two things outrank you, \
both enforced in code: the hard limits with the Risk Intelligence code gate, and Risk \
Intelligence's verdicts — place_trade refuses any trade it didn't APPROVE this cycle, or above \
the risk it approved. To trade something it didn't review, send it back with request_followup \
naming the exact trade.

How to run the firm:
- Supervise; don't rubber-stamp. Check each report follows from the data and from the report \
before it. If work is thin, inconsistent, or you disagree, call request_followup with specific \
instructions. Follow-ups are limited per cycle and cost money; use them when the answer could \
change your decision.
- Weigh departments by the quality of their evidence, not by headcount: computed data (quant \
base rates, liquidity) outranks narrative, and departments with "no view" don't count.

{TRADING_PRINCIPLES}\
- The user sees every trade on their phone with your reasoning: 1-3 plain sentences naming the \
departments' evidence.
- Finish with, under 150 words:
CONSENSUS VERDICT: the trade (symbol, direction, entry, stop, target, risk) or "no trade"
Departments for / against, and any team you sent back or overruled.
What would change your mind next cycle.
It is saved to the desk journal and shown to everyone next cycle.
"""

CEO_TOOLS = ("get_candles", "check_trade", "place_trade", "close_position", "move_stop_loss")


def parse_count(report: str, label: str) -> int | None:
    """The trailing 'LABEL: n' line a team must end with, or None if it's
    missing — and a missing count means 'don't skip', so the chain keeps going."""
    matches = re.findall(rf"^\s*\**{label}\**\s*:\s*\**\s*(\d+)", report, flags=re.IGNORECASE | re.MULTILINE)
    return int(matches[-1]) if matches else None


def symbol_key(symbol: str) -> str:
    """'EURUSDm' / 'eurusd' / 'EURUSDc' -> 'EURUSD' so verdicts match broker names."""
    return re.sub(r"[^A-Z]", "", symbol.upper())[:6]


def parse_verdicts(report: str) -> dict[tuple[str, str], tuple[str, float | None]]:
    verdicts: dict[tuple[str, str], tuple[str, float | None]] = {}
    pattern = r"^\W*VERDICT\W*:\s*\**\s*([A-Za-z0-9_.]+)\s+(buy|sell)\s+(APPROVE|REJECT)\b[^\d\n]*([\d.]+)?"
    for symbol, direction, verdict, risk in re.findall(pattern, report, flags=re.IGNORECASE | re.MULTILINE):
        verdicts[(symbol_key(symbol), direction.lower())] = (verdict.upper(), float(risk) if risk else None)
    return verdicts


def department_findings(report: str) -> list[str]:
    """The numbered 'N. Department: finding' lines of the intelligence report,
    for the Telegram trade message."""
    lines = []
    for line in report.splitlines():
        match = re.match(r"^\s*\**\s*([1-8])[.)]\s*\**\s*([A-Za-z ]+?)\**\s*:\s*\**\s*(.+)$", line)
        if match:
            text = match.group(3).strip()
            lines.append(f"{match.group(1)}. {match.group(2).strip()}: {text[:160]}{'…' if len(text) > 160 else ''}")
    return lines


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
    approvals: dict[tuple[str, str], tuple[str, float | None]] = field(default_factory=dict)
    brief: str = ""

    def __post_init__(self) -> None:
        firm_cfg = self.desk.desk_cfg.get("firm", {})
        self.models: dict[str, str] = firm_cfg.get("models") or {}
        self.max_followups = int(firm_cfg.get("max_followups_per_cycle", 2))

    def model_for(self, role: str) -> str:
        return self.models.get(role) or self.desk.desk_cfg["model"]

    # ----- the chain ---------------------------------------------------------

    def _input_for(self, key: str) -> str:
        previous = {"analyst": "intelligence", "strategist": "analyst", "risk": "strategist"}.get(key)
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
        if key == "risk":
            self.approvals = parse_verdicts(result.text)
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
            if key == "intelligence" and parse_count(report, "CANDIDATES") == 0:
                self._skip_rest(after=key, why="the intelligence departments found no candidate setups and nothing is open")
                break
            if key == "analyst" and parse_count(report, "VETTED") == 0:
                self._skip_rest(after=key, why="the analysts rejected every candidate and nothing is open")
                break

        registry = self.desk.build_registry(CEO_TOOLS)
        registry.register(self._followup_tool())
        self.desk.trade_guard = self._risk_guard
        self.desk.trade_context = {"departments": department_findings(self.reports.get("intelligence", ""))}
        try:
            ceo = self.desk.run_agent(CEO_SYSTEM, self._ceo_input(), registry, 8, self.model_for("ceo"))
        finally:
            self.desk.trade_guard = None
            self.desk.trade_context = None
        self._add_usage("ceo", ceo.usage)
        return FirmResult(
            summary=ceo.text,
            actions=ceo.actions,
            reports=dict(self.reports),
            followups=list(self.followups),
            cost_by_role={role: estimate_cost(u, self.model_for(role)) for role, u in self.usage.items()},
        )

    def _risk_guard(self, symbol: str, direction: str, risk_pct: float) -> str | None:
        """Department 9 overrides everyone, the CEO included: no APPROVE, no trade."""
        verdict, max_risk = self.approvals.get((symbol_key(symbol), direction), (None, None))
        if verdict != "APPROVE":
            return (
                f"Risk Intelligence has not approved {direction} {symbol} this cycle"
                + (" (it rejected it)" if verdict == "REJECT" else "")
                + ". Use request_followup(team='risk') with the exact trade if you want it reviewed."
            )
        if max_risk is not None and risk_pct > max_risk + 1e-9:
            return f"Risk Intelligence approved at most {max_risk:g}% risk for this trade; use risk_pct <= {max_risk:g}."
        return None

    def _skip_rest(self, after: str, why: str) -> None:
        for key in CHAIN[CHAIN.index(after) + 1:]:
            self.reports[key] = f"(skipped this cycle: {why})"

    def _ceo_input(self) -> str:
        sections = [f"Market brief:\n\n{self.brief}", "=== Reports from your organisation ==="]
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
                f"get its revised report. At most {self.max_followups} per cycle. Teams: intelligence, "
                "analyst, strategist, risk. A team skipped this cycle can be asked to run; to get a trade "
                "approved, send 'risk' the exact symbol, direction, stop-loss and take-profit."
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
