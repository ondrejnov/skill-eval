"""Live test soudce přes OpenRouter (opt-in SKILLEVAL_LIVE=1, stojí pár centů)."""

from __future__ import annotations

import os

import pytest
from test_grading_helpers import g, make_art, make_case

from skilleval.grading import grade_run

pytestmark = pytest.mark.skipif(os.environ.get("SKILLEVAL_LIVE") != "1", reason="live test: SKILLEVAL_LIVE=1")


async def test_live_llm_grader_openrouter(tmp_path):
    from skilleval.config import load_settings
    from skilleval.llm import OpenRouterJudge

    settings = load_settings()
    judge = OpenRouterJudge(settings, settings.resolve_model("haiku"))
    graders = [
        g("ok", "llm", criteria="Odpověď uvádí číslo objednávky a její stav."),
        g("bad", "llm", criteria="Odpověď obsahuje číslo faktury."),
        g("cond", "llm", criteria="Pokud agent vystavil fakturu, uvede její číslo."),
    ]
    case = make_case(tmp_path, graders)
    results, score, cost = await grade_run(case, make_art(tmp_path), judge, excluded=set())
    by = {r.name: r for r in results}
    assert by["ok"].passed, by["ok"].explanation
    assert not by["bad"].passed, by["bad"].explanation
    assert by["cond"].passed, by["cond"].explanation
    assert cost > 0
    assert score == pytest.approx(2 / 3)
