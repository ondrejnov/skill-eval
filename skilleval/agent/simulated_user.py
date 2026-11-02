"""Simulovaný uživatel (rozšíření `simulated_user` ze skilleval.yaml): odpovědi na koncovou otázku agenta a na
AskUserQuestion. Repliky `replies[i]` popořadě, potom (nebo místo nich) judge model podle instrukce `agent`.
Rozpočet `max_replies` je společný pro oba způsoby, když runner předá stejný objekt hooku i `run_agent`."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from skilleval.models import Judge, SimulatedUserSpec
from skilleval.trace import TraceWriter

NOT_AVAILABLE = ("User is not available in this non-interactive session; proceed with your best judgement or stop "
                 "and explain.")

SYSTEM = """\
You are simulating the human user of an AI assistant in an automated test. Reply with the user's next message \
only – plain text, no preamble, no quotes, in the language of the conversation. Keep it short and natural.

How the user behaves:
{instructions}"""

# Otázka na volbu bez otazníku na konci ("Dejte vědět, kterou variantu…", "Let me know which …")
_CHOICE_RE = re.compile(
    r"(would you like|do you want|should i|shall i|which (one|option)|let me know (which|whether|if)|please (choose|"
    r"confirm|select)|chcete|\bmám (to|ji|je|ho|pokračovat)\b|přejete si|preferujete|vyberte|zvolte|dejte (mi )?vědět|"
    r"potvrďte|kterou (variantu|možnost)|kterou z)", re.I)
_TRAILING = " \t\r\n*_`\"')]>.:!»“”"


def looks_like_question(text: str) -> bool:
    """Odpověď agenta končí otázkou uživateli (poslední neprázdné řádky končí „?“) nebo žádá o volbu."""
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    if not lines:
        return False
    if any(ln.rstrip(_TRAILING).endswith("?") for ln in lines[-3:]):
        return True
    tail = "\n".join(lines[-4:])[-500:]
    return bool(_CHOICE_RE.search(tail))


def transcript(conversation: list[dict[str, str]]) -> str:
    who = {"user": "USER", "assistant": "ASSISTANT"}
    return "\n\n".join(f"{who.get(m['role'], m['role'].upper())}: {m['content']}" for m in conversation)


class SimulatedUser:
    """Stav simulovaného uživatele jednoho runu. Instance je zároveň hook pro AskUserQuestion (`await sim(input)`)."""

    def __init__(self, spec: SimulatedUserSpec, judge: Judge | None = None):
        self.spec = spec
        self.judge = judge
        self.used = 0
        self.cost_usd = 0.0
        self.conversation: list[dict[str, str]] = []  # zjednodušená konverzace (plní run_agent)
        self.trace: TraceWriter | None = None  # nastaví run_agent

    @property
    def remaining(self) -> int:
        return max(0, self.spec.max_replies - self.used)

    def can_reply(self) -> bool:
        if self.remaining <= 0:
            return False
        return self.used < len(self.spec.replies) or bool(self.spec.agent and self.judge)

    async def reply(self, conversation: list[dict[str, str]], question: str | None = None) -> str | None:
        """Další replika uživatele, nebo None (rozpočet vyčerpán / není z čeho odpovědět)."""
        if not self.can_reply():
            return None
        i = self.used
        if i < len(self.spec.replies):
            text = self.spec.replies[i]
        else:
            assert self.judge is not None and self.spec.agent
            body = transcript(conversation)
            if question:
                body += f"\n\nThe assistant now asks the user:\n{question}"
            body += "\n\nWrite the user's reply."
            text, cost = await self.judge.complete(system=SYSTEM.format(instructions=self.spec.agent.strip()),
                                                   messages=[{"role": "user", "content": body}])
            self.cost_usd += cost
            text = text.strip()
        self.used += 1
        return text

    async def __call__(self, tool_input: dict[str, Any]) -> str:
        """Hook AskUserQuestion: vstup `{questions: [{question, header, options: [{label, description}], …}]}`."""
        questions = [q for q in (tool_input.get("questions") or []) if isinstance(q, dict)]
        asked = format_questions(questions)
        answer = await self.reply(self.conversation, question=asked)
        if answer is None:
            return NOT_AVAILABLE
        if self.trace is not None:
            self.trace.system("skilleval_simulated_user", reply=answer, via="AskUserQuestion")
        labels = ", ".join(str(q.get("question", "")) for q in questions) or "question"
        return f'User has answered your questions: "{labels}"="{answer}". You can now continue with the user\'s answers in mind.'


def format_questions(questions: list[dict[str, Any]]) -> str:
    out = []
    for q in questions:
        line = str(q.get("question", ""))
        opts = [o for o in (q.get("options") or []) if isinstance(o, dict)]
        if opts:
            line += "\n" + "\n".join(f"  - {o.get('label', '')}" + (f": {o['description']}" if o.get("description") else "")
                                     for o in opts)
        out.append(line)
    return "\n".join(out)


def make_ask_user_hook(spec: SimulatedUserSpec, judge: Judge | None) -> Callable[[dict[str, Any]], Awaitable[str]]:
    """Hook pro BuiltinTools (AskUserQuestion). Vrací `SimulatedUser` – předej ho i `run_agent(simulated_user=…)`,
    pak sdílí rozpočet replik, vidí konverzaci, zapisuje do trace a jeho náklady jdou do `cost_usd` runu."""
    return SimulatedUser(spec, judge)


__all__ = ["NOT_AVAILABLE", "SimulatedUser", "looks_like_question", "make_ask_user_hook"]
