"""Grader `baseline`: soudce porovná trace runu s referenčním transcriptem (`baseline_file`, .jsonl v case)."""

from __future__ import annotations

from skilleval.grading.llm import clip_evidence, clip_middle, criteria_of, run_votes, summarize_votes
from skilleval.grading.targets import TargetError, elide_lines, target_text
from skilleval.models import CaseSpec, GraderSpec, Judge, RunArtifacts

BASELINE_SYSTEM = """\
You are a strict, impartial grader comparing a run of an AI agent against a reference transcript of the same task.
Both are session transcripts in JSON lines (Claude Code stream-json): "assistant" messages contain the agent's text \
and tool_use calls (name + input), "user" messages contain the prompt and tool_result outputs. A line like \
"[…N messages elided…]" means messages were omitted from the middle; do not assume anything about them.

Decide whether the RUN satisfies the criteria AT LEAST AS WELL AS the REFERENCE does.
Rules:
- Judge only the criteria; they may be written in any language (often Czech). If no specific criteria are given, \
judge how well the task was accomplished.
- PASS when the run is as good as or better than the reference on the criteria; FAIL when it is clearly worse.
- Differences in wording, formatting, ordering of independent steps, ids or timestamps do not matter unless the \
criteria say so.
- If a criterion is conditional ("If…", "Pokud…") and the condition did not occur in the run, treat it as satisfied.

Respond with ONLY a JSON object, no other text:
{"verdict": "PASS" | "FAIL", "explanation": "<one or two sentences explaining the verdict>"}"""

DEFAULT_CRITERIA = "Accomplish the user's task correctly and completely."


def load_baseline(case: CaseSpec, spec: GraderSpec) -> list[str]:
    name = spec.config.get("baseline_file")
    if not isinstance(name, str) or not name:
        raise ValueError("baseline grader requires `baseline_file`")
    case_dir = case.dir.resolve()
    path = (case_dir / name).resolve()
    if case_dir not in path.parents:
        raise ValueError(f"baseline_file {name!r} must be inside the case directory")
    if path.suffix != ".jsonl":
        raise ValueError(f"baseline_file {name!r} must be a .jsonl transcript")
    if not path.is_file():
        raise TargetError(f"baseline_file not found: {name}")
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def build_user_message(criteria: str, reference: str, run: str, task_prompt: str | None) -> str:
    parts = []
    if task_prompt:
        parts.append(f"## Task given to the agent\n<task>\n{task_prompt.strip()}\n</task>")
    parts.append(f"## Criteria\n<criteria>\n{criteria}\n</criteria>")
    parts.append(f"## Reference transcript\n<reference>\n{reference}\n</reference>")
    parts.append(f"## Run transcript (being graded)\n<run>\n{run}\n</run>")
    parts.append('Does the run satisfy the criteria at least as well as the reference? Answer with the JSON object '
                 '{"verdict": "PASS" | "FAIL", "explanation": "…"}.')
    return "\n\n".join(parts)


async def grade_baseline(case: CaseSpec, spec: GraderSpec, art: RunArtifacts, judge: Judge, *, votes: int = 3,
                         task_prompt: str | None = None) -> tuple[bool, str, str | None, list[bool], float]:
    """→ (passed, explanation, evidence, judge_votes, judge_cost)."""
    try:
        reference_lines = load_baseline(case, spec)
    except TargetError as ex:
        return False, str(ex), None, [], 0.0
    criteria = criteria_of(spec) or DEFAULT_CRITERIA
    run_text, _ = target_text("trace", art, for_llm=True)
    reference = elide_lines(reference_lines)
    user = build_user_message(criteria, clip_middle(reference, 100_000), clip_middle(run_text, 100_000),
                              task_prompt)
    vs, cost = await run_votes(judge, BASELINE_SYSTEM, user, None, votes)
    passed, explanation = summarize_votes(vs)
    return passed, explanation, clip_evidence(run_text), [v.passed for v in vs], cost

