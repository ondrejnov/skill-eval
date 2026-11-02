"""Grader `llm`: soudce hlasuje 3× (paralelně) nad rubrikou, PASS při většině (≥ 2 ze 3)."""

from __future__ import annotations

import asyncio
from typing import Any

from skilleval.grading.targets import TargetError, target_label, target_text
from skilleval.models import GraderSpec, Judge, JudgeVote, RunArtifacts

EVIDENCE_MAX = 20_000  # kolik z evidence se ukládá do výsledku
JUDGE_INPUT_MAX = 200_000  # pojistka proti obřím vstupům soudce (zkrácení uprostřed)
DISSENT_MAX = 300

JUDGE_SYSTEM = """\
You are a strict, impartial grader of the output of an AI agent that was run non-interactively on a task.
You evaluate exactly ONE criterion against the evidence you are given and nothing else.

Rules:
- Judge only the criterion. Ignore other qualities or problems of the output that the criterion does not mention.
- The criterion may be written in any language (often Czech); apply it literally as written.
- If the criterion is conditional ("If…", "When…", "Pokud…", "Když…") and the condition did not occur in the \
evidence, the verdict is PASS.
- Formatting, wording, language and length do not decide the verdict unless the criterion explicitly requires them; \
an answer that is correct but formatted differently than the criterion describes still PASSes.
- Base the verdict only on the evidence. If the evidence does not show that the criterion is satisfied, the verdict \
is FAIL.
- The evidence may be a session transcript in JSON lines (Claude Code stream-json): "assistant" messages contain the \
agent's text and tool_use calls (name + input), "user" messages contain the prompt and tool_result outputs. A line \
like "[…N messages elided…]" means messages were omitted from the middle; do not assume anything about them.

Respond with ONLY a JSON object, no other text:
{"verdict": "PASS" | "FAIL", "explanation": "<one or two sentences explaining the verdict>"}"""

FOCUS_DESCRIPTIONS = {
    "last_message": "the agent's final response text",
    "trace": ("the session transcript as JSON lines, one message per line (only the first 12 and the last 12 "
              "messages are shown when it is longer)"),
    "files": "the list of file paths the agent created during the run, one per line",
    "mock_calls": ("every call the agent made to a mocked MCP tool, one JSON object per line with server, tool, "
                   "input, the mock's output and isError"),
}


def focus_description(focus: Any) -> str:
    if isinstance(focus, dict):
        return f"the contents of the file `{focus.get('path')}` in the agent's workspace after the run"
    return FOCUS_DESCRIPTIONS.get(focus or "last_message", target_label(focus))


def clip_middle(text: str, limit: int = JUDGE_INPUT_MAX) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n[…{len(text) - limit} characters elided…]\n" + text[-half:]


def clip_evidence(text: str) -> str:
    return text if len(text) <= EVIDENCE_MAX else text[:EVIDENCE_MAX] + "…"


def criteria_of(spec: GraderSpec) -> str:
    criteria = spec.config.get("criteria")
    if criteria is None:
        criteria = spec.markdown
    return str(criteria or "").strip()


def build_user_message(criteria: str, focus: Any, evidence: str, task_prompt: str | None) -> str:
    parts = []
    if task_prompt:
        parts.append(f"## Task given to the agent (context only, do not grade it)\n<task>\n{task_prompt.strip()}\n"
                     "</task>")
    parts.append(f"## Criterion\n<criterion>\n{criteria}\n</criterion>")
    parts.append(f"## Evidence\nThe evidence is {focus_description(focus)}.\n<evidence>\n{evidence}\n</evidence>")
    parts.append('Does the evidence satisfy the criterion? Answer with the JSON object {"verdict": "PASS" | "FAIL", '
                 '"explanation": "…"}.')
    return "\n\n".join(parts)


async def run_votes(judge: Judge, system: str, user: str, images: list[dict] | None,
                    votes: int) -> tuple[list[JudgeVote], float]:
    """`votes` hlasů paralelně. Chyba některého hlasu → výjimka (náklady úspěšných hlasů jsou v ex.judge_cost)."""
    results = await asyncio.gather(*(judge.vote(system, user, images or None) for _ in range(votes)),
                                   return_exceptions=True)
    cost = sum(r[1] for r in results if not isinstance(r, BaseException))
    for r in results:
        if isinstance(r, BaseException):
            r.judge_cost = cost  # type: ignore[attr-defined]
            raise r
    return [r[0] for r in results], cost  # type: ignore[index]


def summarize_votes(votes: list[JudgeVote]) -> tuple[bool, str]:
    """PASS při většině; explanation jako Claude Code + vysvětlení prvního FAIL hlasu."""
    passes = sum(v.passed for v in votes)
    passed = passes * 2 > len(votes)
    explanation = "judge votes: " + " ".join("PASS" if v.passed else "FAIL" for v in votes)
    dissent = next((v for v in votes if not v.passed and v.explanation.strip()), None)
    if dissent:
        text = " ".join(dissent.explanation.split())
        explanation += " — " + (text if len(text) <= DISSENT_MAX else text[:DISSENT_MAX] + "…")
    return passed, explanation


async def grade_llm(spec: GraderSpec, art: RunArtifacts, judge: Judge, *, votes: int = 3,
                    task_prompt: str | None = None) -> tuple[bool, str, str | None, list[bool], float]:
    """→ (passed, explanation, evidence, judge_votes, judge_cost)."""
    criteria = criteria_of(spec)
    if not criteria:
        raise ValueError("llm grader requires `criteria` (file body)")
    focus = spec.config.get("focus")
    try:
        text, images = target_text(focus, art, for_llm=True)
    except TargetError as ex:
        return False, str(ex), None, [], 0.0
    user = build_user_message(criteria, focus, clip_middle(text), task_prompt)
    vs, cost = await run_votes(judge, JUDGE_SYSTEM, user, images, votes)
    passed, explanation = summarize_votes(vs)
    return passed, explanation, clip_evidence(text), [v.passed for v in vs], cost
