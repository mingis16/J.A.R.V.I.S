"""Daily intelligence briefing: departments 1 (Macroeconomic), 2 (Geopolitical),
3 (Fundamental) and the positioning half of 6 (Sentiment).

Their inputs — central-bank decisions, data releases, conflicts, sanctions,
rate differentials, CFTC positioning, today's event calendar — change daily,
not hourly, and need live news the broker doesn't provide. So this runs once
per UTC day with Claude's web search tool, is cached in state/, and every
hourly cycle reuses it. If web search is unavailable, it falls back to a
prices-only briefing that says so plainly rather than presenting
remembered facts as current.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from assistant.trading_desk.desk import FALLBACK_BETA, empty_usage, estimate_cost

logger = logging.getLogger("assistant.trading_desk.briefing")

BRIEFING_SYSTEM = """\
You are four intelligence departments of a small trading firm that trades EURUSD, GBPUSD, \
USDJPY and gold (XAUUSD) for one client: Macroeconomic, Geopolitical, Fundamental, and \
Positioning. Use web_search to establish today's facts — don't rely on memory for anything \
that changes (rates, latest data, events, positioning). Prefer primary and reputable sources, \
and give the date of each key fact. If you can't verify something, say "unverified".

Searches are limited, so make each one broad and follow this plan (one search each):
1. the latest Fed, ECB, BoE and BoJ rate decisions and guidance
2. the high-impact economic calendar for today and tomorrow
3. the latest CFTC Commitments of Traders positioning for EUR, GBP, JPY and gold
4. geopolitical risks moving currency and gold markets this week
5. this week's outlook for EURUSD, GBPUSD, USDJPY and gold
Use any remaining searches only to fill a gap that changes the bias. The forex market is open \
from Sunday 22:00 UTC to Friday 21:00 UTC; use the date and time in the message to know where \
in the week you are.

Write the briefing in exactly this structure, under 350 words in total:
1. MACROECONOMIC: policy rates and the latest decision or guidance for the Fed, ECB, BoE and BoJ; \
latest inflation, jobs and growth surprises; what the market expects next.
2. GEOPOLITICAL: conflicts, sanctions, trade policy and elections that could move these markets \
now, and in which direction.
3. FUNDAMENTAL: rate differentials (USD vs EUR, GBP, JPY), real-yield direction for gold, and \
which currencies look fundamentally supported or weak.
6. POSITIONING: latest CFTC Commitments of Traders positioning in EUR, GBP, JPY and gold \
(crowded longs or shorts), plus any widely reported retail-sentiment extremes.
EVENT RISK (next 24h, UTC): one line per high-impact release or speech, as "HH:MM UTC | currency | \
event", or "none found".
BIAS: one line per instrument: bullish, bearish or neutral from these four departments, with \
the main reason.
"""

NO_WEB_NOTE = """
Web search is unavailable for this run. Base the briefing ONLY on the market data in the \
message, label every section "prices only, no live news", and do not present remembered facts \
about rates, data, or events as current.
"""


class DailyBriefing:
    def __init__(self, repo_root: Path, desk_cfg: dict, client: Any):
        self.path = repo_root / "state" / "daily_briefing.json"
        self.cfg = desk_cfg.get("briefing", {})
        self.model = self.cfg.get("model") or desk_cfg["model"]
        self.effort = desk_cfg.get("effort", "low")
        self.client = client

    def cached(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def get(self, now: datetime, market_snapshot: str) -> tuple[str, float]:
        """Today's briefing text and what generating it cost now (0 if cached)."""
        if not self.cfg.get("enabled"):
            return "(daily briefing disabled)", 0.0
        cached = self.cached()
        if cached and cached.get("date") == now.date().isoformat():
            return self._label(cached), 0.0
        record = self._generate(now, market_snapshot)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return self._label(record), record["est_cost_usd"]

    @staticmethod
    def _label(record: dict[str, Any]) -> str:
        source = "with live web search" if record.get("web") else "PRICES ONLY — no live news this time"
        return f"(written {record['created_utc'][11:16]} UTC, {source})\n{record['text']}"

    def _generate(self, now: datetime, market_snapshot: str) -> dict[str, Any]:
        tomorrow = now + timedelta(days=1)
        content = (
            f"Today is {now:%A %d %B %Y}, {now:%H:%M} UTC (tomorrow is {tomorrow:%A %d %B %Y}).\n\n"
            f"Market data right now:\n{market_snapshot}"
        )
        usage = empty_usage()
        searches = 0
        try:
            text, searches = self._run(content, BRIEFING_SYSTEM, usage, with_web=True)
            web = True
        except Exception as exc:  # web search disabled for the org, network, etc.
            logger.warning("Briefing web search failed (%s: %s); falling back to prices only.", type(exc).__name__, exc)
            text, _ = self._run(content, BRIEFING_SYSTEM + NO_WEB_NOTE, usage, with_web=False)
            web = False
        return {
            "date": now.date().isoformat(),
            "created_utc": now.isoformat(),
            "web": web,
            "web_searches": searches,
            "text": text,
            "usage": usage,
            # Token cost only; web searches are billed separately per search.
            "est_cost_usd": estimate_cost(usage, self.model),
        }

    def _run(self, content: str, system: str, usage: dict[str, int], with_web: bool) -> tuple[str, int]:
        tools = (
            [{"type": "web_search_20260209", "name": "web_search", "max_uses": int(self.cfg.get("web_search_max_uses", 5))}]
            if with_web
            else []
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        searches = 0
        for _ in range(4):
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=8000,
                system=system,
                messages=messages,
                output_config={"effort": self.effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
                **({"tools": tools} if tools else {}),
            )
            for key in usage:
                usage[key] += int(getattr(response.usage, key, 0) or 0)
            server_use = getattr(response.usage, "server_tool_use", None)
            searches += int(getattr(server_use, "web_search_requests", 0) or 0)
            if response.stop_reason == "pause_turn":
                # Server-side search loop hit its iteration cap: re-send so it resumes.
                messages = [{"role": "user", "content": content}, {"role": "assistant", "content": response.content}]
                continue
            if response.stop_reason == "refusal":
                raise RuntimeError("briefing request was declined")
            # Cited answers arrive as many text blocks split around each citation;
            # joining with "" restores the sentences (a newline join broke them up).
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            if not text:
                raise RuntimeError("briefing came back empty")
            return text, searches
        raise RuntimeError("briefing did not finish within 4 continuations")
